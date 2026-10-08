"""Events yielded by Agent.run(). One event type per thing worth reacting to."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .protocol import ParsedCall, ParsedTurn

__all__ = [
    "Event", "Delta", "Think", "Plan", "ToolCalled", "ToolReturned",
    "ConfirmNeeded", "Final", "Failed", "RoundStarted",
]


@dataclass
class Event:
    """Base class so callers can `match` or isinstance-check loosely."""


@dataclass
class RoundStarted(Event):
    """A new generation round began (round 0 is the first)."""

    round: int
    prompt_tokens: int | None = None


@dataclass
class Delta(Event):
    """A chunk of user-visible answer text, i.e. inside <response>.

    Scaffolding (<think>, <task_plan>, tool calls) is never delivered here; it
    arrives as Think/Plan/ToolCalled so a UI can show or hide it deliberately.
    """

    text: str


@dataclass
class Think(Event):
    """The model's scratchpad for this round. Always English by convention, and
    not meant for the user."""

    text: str


@dataclass
class Plan(Event):
    """The ordered task-plan verbs, e.g. ('<|RAG|>', '<|explain|>').

    `invalid` holds any verb outside the trained set -- a format violation
    rather than a plan step.
    """

    verbs: tuple[str, ...]
    invalid: tuple[str, ...] = ()


@dataclass
class ConfirmNeeded(Event):
    """A tool marked requires_confirmation is about to run.

    Only emitted when no `confirm` callback was given to the Agent; with a
    callback the agent asks it directly and never surfaces this. Set `.approved`
    then call `agent.resume()`, or treat it as a decline.
    """

    call: ParsedCall
    approved: bool | None = None

    def approve(self) -> None:
        self.approved = True

    def decline(self) -> None:
        self.approved = False


@dataclass
class ToolCalled(Event):
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""


@dataclass
class ToolReturned(Event):
    name: str
    value: Any = None
    error: str | None = None
    declined: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.declined


@dataclass
class Final(Event):
    """The answer. `turn` carries the parsed structure of the last generation,
    and `data` the validated object when a schema was requested."""

    text: str
    turn: ParsedTurn | None = None
    rounds: int = 1
    data: Any = None


@dataclass
class Failed(Event):
    """The loop gave up: round limit hit, unparseable arguments after retries,
    or a backend error. `text` is whatever was produced anyway."""

    reason: str
    text: str = ""
    turn: ParsedTurn | None = None
