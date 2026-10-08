"""Tools: declare a Python function, get an OpenAI-shaped schema and a safe caller.

Only registered tools can be invoked, arguments are validated against the schema
before the function runs, and a tool may demand confirmation before it executes.

    @tool(description="Current weather for a city.")
    def get_weather(city: str, units: str = "celsius") -> dict:
        return {"city": city, "temp": 31}

    box = Toolbox([get_weather])
    box.schemas()            # -> [{"type": "function", "function": {...}}]
    box.call("get_weather", {"city": "Lagos"})

Schemas come from type hints and the docstring, or you can pass `parameters=`
explicitly, or annotate the single argument with a pydantic model.
"""

from __future__ import annotations

import inspect
import json
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Literal, Sequence, get_args, get_origin

__all__ = ["Tool", "Toolbox", "tool", "ToolError", "ToolResult"]

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolError(Exception):
    """Raised for a problem the MODEL should see and can recover from: unknown
    tool, bad arguments, a tool that raised. The agent turns it into a
    <tool_response> rather than crashing the loop."""


@dataclass(frozen=True)
class ToolResult:
    name: str
    value: Any = None
    error: str | None = None
    declined: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.declined

    def payload(self) -> Any:
        """What goes back into the model's context."""
        if self.declined:
            return {"error": "declined", "detail": self.error or "the user declined this call"}
        if self.error:
            return {"error": self.error}
        return self.value


def _schema_for_annotation(ann: Any) -> dict[str, Any]:
    """JSON Schema for one parameter annotation. Unknown types fall back to
    string, which keeps a usable schema rather than refusing to build one."""
    if ann is inspect.Parameter.empty or ann is Any:
        return {"type": "string"}
    origin = get_origin(ann)
    if origin is Literal:
        options = list(get_args(ann))
        kind = _JSON_TYPES.get(type(options[0]), "string") if options else "string"
        return {"type": kind, "enum": options}
    if origin in (list, Sequence, tuple):
        args = get_args(ann)
        item = _schema_for_annotation(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if origin is dict:
        return {"type": "object"}
    if origin is typing.Union:  # includes `X | None`
        inner = [a for a in get_args(ann) if a is not type(None)]
        return _schema_for_annotation(inner[0]) if len(inner) == 1 else {"type": "string"}
    if isinstance(ann, type) and issubclass(ann, Enum):
        return {"type": "string", "enum": [m.value for m in ann]}
    if isinstance(ann, type) and ann in _JSON_TYPES:
        return {"type": _JSON_TYPES[ann]}
    # A pydantic model (or anything exposing a JSON schema) used as a type.
    schema_fn = getattr(ann, "model_json_schema", None)
    if callable(schema_fn):
        return schema_fn()
    return {"type": "string"}


def _doc_summary(fn: Callable[..., Any]) -> str:
    doc = inspect.getdoc(fn) or ""
    return doc.strip().split("\n\n")[0].replace("\n", " ").strip()


def _hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """Resolved type hints, not raw __annotations__.

    Under `from __future__ import annotations` (used throughout this codebase)
    every annotation is a STRING, so reading param.annotation directly yields
    'float' instead of float and every parameter silently degrades to
    {"type": "string"} -- which then rejects correct arguments at validation
    time, e.g. a float argument refused as "must be string".

    get_type_hints evaluates them against the function's module globals, which
    fails outright if any one name is not importable there (a type imported
    inside the function body, or declared in a local scope). One unresolvable
    name should not cost the schema every OTHER parameter, so the fallback
    resolves annotations one at a time against the module globals plus typing's
    own namespace, and gives up only on the individual names it cannot see.
    """
    try:
        return typing.get_type_hints(fn)
    except Exception:
        pass
    namespace = {**vars(typing), **getattr(fn, "__globals__", {})}
    out: dict[str, Any] = {}
    for key, raw in getattr(fn, "__annotations__", {}).items():
        if not isinstance(raw, str):
            out[key] = raw
            continue
        try:
            out[key] = eval(raw, namespace)  # noqa: S307 - annotation text from our own source
        except Exception:
            continue
    return out


def _parameters_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    hints = _hints(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        props[name] = _schema_for_annotation(hints.get(name, param.annotation))
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def _validate(schema: dict[str, Any], args: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """Check arguments against the schema. Deliberately small: presence, unknown
    keys, and shallow type checks with int->float widening. The point is to stop
    a hallucinated argument reaching a real function, not to be a JSON Schema
    implementation."""
    if not isinstance(args, dict):
        raise ToolError(f"{tool_name}: arguments must be an object, got {type(args).__name__}")
    props: dict[str, Any] = schema.get("properties", {}) or {}
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        raise ToolError(f"{tool_name}: missing required argument(s): {', '.join(missing)}")
    if schema.get("additionalProperties") is False:
        unknown = [k for k in args if k not in props]
        if unknown:
            raise ToolError(
                f"{tool_name}: unknown argument(s): {', '.join(unknown)}. "
                f"Valid arguments: {', '.join(props) or 'none'}"
            )
    checkers: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "array": (list, tuple),
        "object": (dict,),
    }
    for key, value in args.items():
        spec = props.get(key)
        if not spec:
            continue
        want = spec.get("type")
        allowed = checkers.get(want)
        # bool is an int subclass; a bool where a number belongs is a real error.
        if allowed and (not isinstance(value, allowed) or (want in ("integer", "number") and isinstance(value, bool))):
            raise ToolError(f"{tool_name}: argument {key!r} must be {want}, got {type(value).__name__}")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"{tool_name}: argument {key!r} must be one of {spec['enum']}, got {value!r}")
    return args


@dataclass
class Tool:
    """A callable exposed to the model."""

    fn: Callable[..., Any]
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    timeout: float | None = None
    # Set when the function takes a single pydantic model, so arguments are
    # validated and passed as that model rather than as loose kwargs.
    model: Any = None

    def __post_init__(self) -> None:
        self.name = self.name or getattr(self.fn, "__name__", "tool")
        self.description = self.description or _doc_summary(self.fn)
        if not self.parameters:
            self.parameters = self._infer_parameters()

    def _infer_parameters(self) -> dict[str, Any]:
        sig = inspect.signature(self.fn)
        hints = _hints(self.fn)
        named = [(n, p) for n, p in sig.parameters.items() if n not in ("self", "cls")]
        if len(named) == 1:
            name, param = named[0]
            ann = hints.get(name, param.annotation)
            schema_fn = getattr(ann, "model_json_schema", None)
            if callable(schema_fn):  # single pydantic-model argument
                self.model = ann
                schema = dict(schema_fn())
                schema.pop("title", None)
                schema.setdefault("additionalProperties", False)
                return schema
        return _parameters_from_signature(self.fn)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        return _validate(self.parameters, args, self.name)

    def __call__(self, args: dict[str, Any]) -> Any:
        args = self.validate(args)
        if self.model is not None:
            return self.fn(self.model(**args))
        if self.timeout is None:
            return self.fn(**args)
        return _call_with_timeout(self.fn, args, self.timeout, self.name)


def _call_with_timeout(fn: Callable[..., Any], args: dict[str, Any], timeout: float, name: str) -> Any:
    """Run fn in a worker thread and give up waiting after `timeout`.

    A timed-out thread is NOT killed -- Python cannot safely do that -- so the
    call keeps running in the background and only the wait is abandoned. Use
    timeouts to keep the agent responsive, not as a resource guarantee.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: fn(**args))
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            raise ToolError(f"{name}: timed out after {timeout:g}s") from None
        finally:
            pool.shutdown(wait=False)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str = "",
    description: str = "",
    parameters: dict[str, Any] | None = None,
    requires_confirmation: bool = False,
    timeout: float | None = None,
    model: Any = None,
) -> Any:
    """Decorator turning a function into a Tool. Usable bare or with arguments.

    Schemas are inferred from type hints and the docstring. Pass `parameters=`
    to supply JSON Schema directly, or `model=` to name the pydantic model the
    single argument expects -- needed only when the model class cannot be
    resolved from the function's module globals (one declared inside another
    function, say), since postponed annotations leave only its name behind.
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        t = Tool(
            fn=f,
            name=name,
            description=description,
            parameters=dict(parameters) if parameters else {},
            requires_confirmation=requires_confirmation,
            timeout=timeout,
        )
        if model is not None:
            t.model = model
            if not parameters:
                schema = dict(model.model_json_schema())
                schema.pop("title", None)
                schema.setdefault("additionalProperties", False)
                t.parameters = schema
        return t

    return wrap(fn) if fn is not None else wrap


class Toolbox:
    """A registry; the allowlist is simply what has been registered."""

    def __init__(self, tools: Iterable[Tool | Callable[..., Any]] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.add(t)

    def add(self, t: Tool | Callable[..., Any]) -> Tool:
        resolved = t if isinstance(t, Tool) else Tool(fn=t)
        self._tools[resolved.name] = resolved
        return resolved

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(sorted(self._tools)) or "none registered"
            raise ToolError(f"unknown tool {name!r}. Available tools: {known}") from None

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Validate and run, converting every failure into a ToolResult so the
        agent can hand the model something to recover from."""
        try:
            t = self[name]
        except ToolError as exc:
            return ToolResult(name=name, error=str(exc))
        try:
            return ToolResult(name=name, value=t(args))
        except ToolError as exc:
            return ToolResult(name=name, error=str(exc))
        except Exception as exc:  # a tool's own failure is information, not a crash
            return ToolResult(name=name, error=f"{type(exc).__name__}: {exc}")

    def as_json(self) -> str:
        return json.dumps(self.schemas(), ensure_ascii=False, separators=(",", ":"))
