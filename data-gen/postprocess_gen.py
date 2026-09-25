"""Model response -> validated training record.

Every record carries BOTH representations, deliberately:
  * `messages` / `prompt_messages` -- role/content dicts. The source of truth.
  * `text` / `prompt_text`         -- exactly what the SabiYarn chat template renders from them.
Keeping both means a later template change is a cheap re-render (`python postprocess_gen.py --rerender`)
instead of regenerating the corpus, and training can consume the flat text without a Jinja dependency.

`instruction` / `input` / `context` / `response` are flattened from the LAST exchange for tooling that
expects the Alpaca shape. They are lossy for multi-turn samples; `messages` is what to train on.

Anything that fails validation is dropped and counted, never silently repaired -- a malformed special-token
sequence that reaches training teaches the model to emit malformed special-token sequences.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Optional

from providers.base import Response
from assemble import AssemblyError, build_messages
from rendering.render import render_messages
from schemas.seed import TAGS, Seed

ROLES = ("system", "user", "assistant", "tool")
_THINK = re.compile(r"<think>(.*?)</think>", re.S)
# The only closers in the tokenizer. Everything else a generator invents (</response>, </sentiment>, </NER>)
# is not in the vocabulary and costs 3-4 junk sub-word tokens.
_VALID_CLOSERS = {"</think>", "</task_plan>", "</tool_call>", "</tool_response>", "</context>", "</s>"}
_ANY_CLOSER = re.compile(r"</[A-Za-z_|][^>\s]*>")


def _strip_invalid_closers(text: str) -> str:
    """Remove closing tags that do not exist in the tokenizer.

    This is the ONE sanctioned repair in this module. Everything else that fails validation is dropped,
    because a silently-fixed structural error teaches the model the error. A redundant closing tag is
    different: it carries no information, its removal cannot change meaning, and generators emit them
    constantly however firmly the prompt forbids it (measured: 61 occurrences of </response> in a 26-sample
    pilot). Dropping those samples would throw away good conversations over pure punctuation.
    """
    return _ANY_CLOSER.sub(lambda m: m.group(0) if m.group(0) in _VALID_CLOSERS else "", text)
_RESP = re.compile(r"<response>(.*)", re.S)

STATS: dict[str, int] = {}


def _drop(reason: str) -> None:
    STATS[reason] = STATS.get(reason, 0) + 1


def _json(text: str) -> Optional[dict]:
    """Parse the model's JSON, tolerating a ```json fence but nothing more adventurous."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except ValueError:  # JSONDecodeError, and the bare ValueError for >4300-digit integer literals
        start, end = t.find("{"), t.rfind("}")
        if 0 <= start < end:
            try:
                v = json.loads(t[start:end + 1])
                return v if isinstance(v, dict) else None
            except ValueError:
                return None
        return None


# A tool-calling turn whose plan the generator forgot: infer it from the tool, exactly as the repair script
# does for published data. Deterministic, and the alternative is discarding an otherwise sound conversation.
_PLAN_BY_TOOL = {
    "search_documents": ["<|RAG|>"], "search_internet": ["<|explain|>"], "fetch_page": ["<|analyze|>"],
    "lookup_health_guidance": ["<|recommend|>"], "find_health_facility": ["<|recommend|>"],
    "calculate": ["<|math|>"], "run_statistics": ["<|math|>"], "convert_units": ["<|math|>"],
    "get_exchange_rate": ["<|math|>"], "get_market_prices": ["<|analyze|>"],
    "search_db": ["<|data_extract|>"], "db_get_record": ["<|data_extract|>"],
    "db_insert_record": ["<|plan|>"], "send_message": ["<|plan|>"], "set_reminder": ["<|plan|>"],
    "translate_text": ["<translate>"], "get_weather_forecast": ["<|explain|>"],
    "get_transport_route": ["<|plan|>"], "lookup_crop_guidance": ["<|recommend|>"],
    "lookup_legal_info": ["<|explain|>"], "get_current_datetime": ["<|chat|>"],
}


def _fill_turn_defaults(turns: Any, md: dict) -> Any:
    """Supply the few fields the generator omits, where the value is determined rather than guessed.

    Measured on a live run: 4 turns with no input_lang, 1 with no task_plan, 1 using <|summarize|> for
    <summarize>. Each is a near-miss with exactly one sensible completion, so filling it recovers a sound
    conversation instead of discarding one. Anything genuinely ambiguous is still dropped by the assembler.
    """
    if not isinstance(turns, list):
        return turns
    expect = md.get("expect_markers") or []
    src_default = expect[0] if expect else md.get("lang")
    tgt_default = expect[1] if len(expect) > 1 else md.get("lang")
    for t in turns:
        if not isinstance(t, dict) or t.get("role") != "assistant":
            continue
        if not t.get("input_lang"):
            t["input_lang"] = src_default
        call = t.get("tool_call") or {}
        if not t.get("task_plan"):
            name = (call or {}).get("name")
            t["task_plan"] = _PLAN_BY_TOOL.get(name, ["<|chat|>"])
        if not call and not t.get("target_lang"):
            t["target_lang"] = tgt_default
    return turns


def _messages_from_turns(raw: Any) -> Optional[list[dict]]:
    """Structured turns -> template-shaped messages, with all markers built by assemble.py.

    This replaced asking the generator for the finished marker string, which it got wrong in 88 of 108
    published records. Anything the assembler cannot build correctly is dropped with a specific reason
    rather than patched.
    """
    try:
        return build_messages(raw)
    except AssemblyError as exc:
        _drop(f"assembly:{str(exc)[:60]}")
        return None
    except Exception as exc:  # noqa: BLE001
        _drop(f"assembly_error:{type(exc).__name__}")
        return None


def _clean_messages(raw: Any) -> Optional[list[dict]]:
    """Normalise to template-shaped dicts, or None if the structure is unusable."""
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict] = []
    for m in raw:
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            return None
        role = m["role"]
        msg: dict[str, Any] = {"role": role, "content": _strip_invalid_closers(m.get("content") or "")}
        if role == "tool":
            if not msg["content"]:
                return None
            msg["name"] = m.get("name") or ""
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = (c or {}).get("function") or {}
                name, args = fn.get("name"), fn.get("arguments")
                if not name:
                    return None
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        return None
                calls.append({"function": {"name": name, "arguments": args or {}}})
            msg["tool_calls"] = calls
            msg.pop("content", None) if not msg["content"] else None
        out.append(msg)
    return out


def _validate_conversation(msgs: list[dict], seed: Seed, *, ends_with: str,
                           md: Optional[dict] = None) -> bool:
    md = md or {}
    conv = seed.conversation
    # User turns are what the generator is asked for and what is checked; assistant and tool turns follow
    # from them (see the seed's min_user_turns/max_user_turns).
    n_user = sum(1 for m in msgs if m["role"] == "user")
    uses_tools = any(m.get("tool_calls") for m in msgs) or any(m["role"] == "tool" for m in msgs)
    hi_turns = int(conv.get("max_user_turns_with_tools", 6) if uses_tools else conv.get("max_user_turns", 5))
    if not (int(conv.get("min_user_turns", 3)) <= n_user <= hi_turns):
        _drop(f"user_turns:{n_user}{'+tools' if uses_tools else ''}")
        return False
    max_total = int(conv.get("max_total_messages_with_tools", 20) if uses_tools
                    else conv.get("max_total_messages", 12))
    if len(msgs) > max_total:
        _drop(f"total_messages:{len(msgs)}{'+tools' if uses_tools else ''}")
        return False
    if msgs[-1]["role"] != ends_with:
        _drop(f"ends_with:{msgs[-1]['role']}")
        return False
    if any(m["role"] == "system" for m in msgs[1:]):
        _drop("system_not_first")
        return False
    # a tool result must follow an assistant tool_call
    for i, m in enumerate(msgs):
        if m["role"] == "tool" and not (i and msgs[i - 1].get("tool_calls")):
            _drop("orphan_tool_result")
            return False
    tool_names = {t.name for t in seed.tools}
    for m in msgs:
        for c in m.get("tool_calls", []):
            if c["function"]["name"] not in tool_names:
                _drop("unknown_tool")
                return False
    # Safety net only: distractors are now injected after generation, so the generator never saw them and
    # cannot have called one. Kept in case a name ever collides, because a sample that calls a distractor
    # teaches the exact opposite of what distractors are for.
    distractors = set(md.get("distractor_tools") or [])
    if distractors:
        for m in msgs:
            for c in m.get("tool_calls", []):
                if c["function"]["name"] in distractors:
                    _drop(f"called_distractor_tool:{c['function']['name']}")
                    return False
    invented = _invented_pipe_tokens(msgs)
    if invented:
        _drop(f"invented_pipe_token:{invented[0]}")
        return False
    if _degenerate_user_message(msgs):
        _drop("degenerate_user_message")
        return False
    return True


def _assistant_parts(content: str) -> dict[str, str]:
    think = _THINK.search(content)
    resp = _RESP.search(content)
    return {"think": (think.group(1).strip() if think else ""),
            "response": (resp.group(1).strip() if resp else "")}


def _flatten(msgs: list[dict]) -> dict[str, str]:
    """instruction/input/context/response from the last exchange, for Alpaca-shaped tooling."""
    last_user = next((m for m in reversed(msgs) if m["role"] == "user"), None)
    last_asst = next((m for m in reversed(msgs) if m["role"] == "assistant" and m.get("content")), None)
    ctx = "\n\n".join(m.get("content", "") for m in msgs if m["role"] == "tool")
    return {"instruction": (last_user or {}).get("content", ""), "input": "",
            "context": ctx, "response": _assistant_parts((last_asst or {}).get("content", ""))["response"]}


# Every <|...|> token that exists in the tokenizer. Anything else is a pseudo-token the generator invented --
# measured at 70 occurrences vs 21 correct markers in low-resource languages, where the model substitutes
# <|efi|> for <|input_lang|><efi>. Those are junk sub-words the target model would learn to emit.
_VALID_PIPE_TOKENS = {
    "<|system|>", "<|user|>", "<|assistant|>", "<|input_lang|>", "<|target_lang|>",
    "<|chat|>", "<|generate|>", "<|edit|>", "<|data_extract|>", "<|math|>", "<|JSON|>", "<|code|>",
    "<|plan|>", "<|analyze|>", "<|recommend|>", "<|explain|>", "<|debug|>", "<|RAG|>",
    "<|is_technical|>", "<|is_legal|>", "<|is_medical|>", "<|is_scientific|>", "<|is_non_technical|>",
    "<|is_financial|>", "<|hate|>",
}
_PIPE_RE = re.compile(r"<\|[^|>]{1,20}\|>")


def _invented_pipe_tokens(msgs: list[dict]) -> list[str]:
    found = []
    for m in msgs:
        for t in _PIPE_RE.findall(m.get("content") or ""):
            if t not in _VALID_PIPE_TOKENS:
                found.append(t)
    return found


def _distinct_ngram(text: str, n: int = 4) -> float:
    w = text.split()
    if len(w) <= n:
        return 1.0
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(g)) / len(g)


def _degenerate_user_message(msgs: list[dict], floor: float = 0.75) -> bool:
    """A looping user message ("mme nka mme nka nte nnyin eyenam eyọm nnyin mme nka...") is the signature of
    a model that does not actually know the language. Checked on USER turns specifically: repetition metrics
    over assistant responses alone scored those samples 0.974 and passed them."""
    for m in msgs:
        if m["role"] != "user":
            continue
        c = m.get("content") or ""
        if len(c.split()) >= 12 and _distinct_ngram(c) < floor:
            return True
    return False


_MARKER_RE = re.compile(r"<\|input_lang\|><([a-z]{2,4})>.*?<\|target_lang\|><([a-z]{2,4})>", re.S)


def _markers_ok(msgs: list[dict], expect: Optional[list[str]]) -> Optional[bool]:
    """Did the assistant set <|input_lang|>/<|target_lang|> to the pair the io_direction called for?

    Recorded rather than enforced: a mislabelled direction is bad data, but measuring compliance first tells
    us whether to filter on it or fix the prompt. compare_models.py reports the rate.
    """
    if not expect:
        return None
    seen = [m for msg in msgs if msg["role"] == "assistant"
            for m in _MARKER_RE.findall(msg.get("content") or "")]
    if not seen:
        return None
    # Satisfied if ANY answering turn uses the required pair. Requiring every turn was wrong: a conversation
    # that reads English material and then takes Pidgin follow-ups legitimately has several directions, and
    # io_direction names the primary pattern, not a constraint on every turn.
    return any(list(pair) == list(expect) for pair in seen)


# Varied so the model does not bind its behaviour to one name and a deployment can rename it without
# retraining. Chosen deterministically from the sample id.
IDENTITIES = [
    "You are a helpful AI assistant.",
    "You are an AI assistant.",
    "You are SabiYarn, an AI assistant for West African languages.",
    "You are an AI assistant. Your name is Aegis.",
    "You are an AI assistant called Sabi.",
    "You are a helpful assistant that speaks West African languages.",
    "You are an AI assistant. Your name is Ọmọlúàbí.",
    "You are a multilingual AI assistant.",
]


def _rebuild_system_message(md: dict, rng_seed: str) -> Optional[str]:
    """Canonical system message: the preamble plus the applicable tools AND the injected distractors.

    Built here rather than taken from the generator for three reasons. It guarantees the distractors are
    present (the generator never saw them, so it cannot have called one). It gives every sample in the corpus
    an identically-formatted catalogue, instead of whatever prose the generator wrapped around its JSON. And
    it keeps the distractor definitions out of the generation request, which is 300-1,200 input tokens saved
    per request for no loss of signal -- the target model learns from the finished catalogue, and cannot tell
    where a definition came from.
    """
    real = md.get("tool_definitions") or []
    fake = md.get("distractor_definitions") or []
    if not real and not fake:
        return None
    catalogue = list(real) + list(fake)
    rng = random.Random(rng_seed)
    # Deterministic shuffle so distractors are not always last -- position must not be a tell.
    rng.shuffle(catalogue)
    identity = md.get("identity") or rng.choice(IDENTITIES)
    # Compact JSON: indent=2 cost ~40% more tokens for no benefit to the model.
    return f"{identity} You have these tools:\n" + json.dumps(catalogue, ensure_ascii=False,
                                                              separators=(",", ":"))


def _tags(raw: Any, fallback: list[str]) -> list[str]:
    got = [t for t in (raw or []) if t in TAGS]
    return sorted(set(got) | set(fallback)) or list(fallback)


def to_record(seed: Seed, resp: Response) -> Optional[dict]:
    """Never raises. A single malformed response must not be able to kill a 500k-request run."""
    try:
        return _to_record(seed, resp)
    except Exception as exc:  # noqa: BLE001
        _drop(f"postprocess_error:{type(exc).__name__}")
        return None


def _to_record(seed: Seed, resp: Response) -> Optional[dict]:
    md = resp.metadata or {}
    data = _json(resp.text)
    if data is None:
        _drop("json_invalid")
        return None
    conf = data.get("confidence")
    try:
        conf = min(1.0, max(0.0, float(conf))) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    base = {"id": resp.custom_id, "lang": md.get("lang", ""), "confidence": conf,
            "io_direction": md.get("io_direction"),
            "model": resp.model, "domain": md.get("domain", ""), "subtopic": md.get("subtopic", "")}

    if seed.kind == "pretrain":
        text = (data.get("text") or "").strip()
        if len(text.split()) < 60:
            _drop("too_short")
            return None
        if data.get("language_self_check") is False:
            _drop("language_self_check_false")
            return None
        return {**base, "genre": md.get("genre", ""), "title": (data.get("title") or "").strip(), "text": text}

    if seed.kind == "sft":
        # `turns` is the current contract (structured fields); `messages` is the legacy pre-assembled shape,
        # still accepted so old batch output can be post-processed.
        turns = data.get("turns") or data.get("messages") or data.get("conversation")
        if isinstance(turns, list) and any(isinstance(t, dict) and ("task_plan" in t or "response" in t
                                                                   or "tool_call" in t) for t in turns):
            msgs = _messages_from_turns(_fill_turn_defaults(turns, md))
        elif data.get("turns") is not None:
            msgs = _messages_from_turns(_fill_turn_defaults(data["turns"], md))
        else:
            msgs = _clean_messages(data.get("messages"))
        if msgs is None:
            _drop("messages_malformed") if data.get("turns") is None else None
            return None
        if not _validate_conversation(msgs, seed, ends_with="assistant", md=md):
            return None
        if not _assistant_parts(msgs[-1].get("content", ""))["response"]:
            _drop("final_turn_has_no_response_token")
            return None
        sysmsg = _rebuild_system_message(md, resp.custom_id)
        if sysmsg:
            if msgs and msgs[0]["role"] == "system":
                msgs[0]["content"] = sysmsg
            else:
                msgs.insert(0, {"role": "system", "content": sysmsg})
        return {**base, "tags": _tags(data.get("tags"), md.get("tags", [])),
                "tasks": data.get("tasks") or md.get("tasks", []),
                "tools": md.get("tools", []), "distractor_tools": md.get("distractor_tools", []),
                "io_markers_ok": _markers_ok(msgs, md.get("expect_markers")),
                "messages": msgs, "text": render_messages(msgs), **_flatten(msgs)}

    # rl
    if data.get("prompt_turns") is not None:
        msgs = _messages_from_turns(_fill_turn_defaults(data["prompt_turns"], md))
    else:
        msgs = _clean_messages(data.get("prompt_messages"))
    if msgs is None:
        _drop("prompt_messages_malformed") if data.get("prompt_turns") is None else None
        return None
    if not _validate_conversation(msgs, seed, ends_with="user", md=md):
        return None
    responses = data.get("responses") or []
    want = int(seed.conversation["responses_per_prompt"])
    responses = [r for r in responses if isinstance(r, dict) and (r.get("content") or "").strip()]
    if len(responses) < 2:
        _drop("too_few_responses")
        return None
    quals = [(r.get("quality") or "").lower() for r in responses]
    if "best" not in quals or "worst" not in quals:
        _drop("no_best_or_worst")
        return None
    texts = [_strip_invalid_closers(r["content"]).strip() for r in responses]
    if len(set(texts)) < len(texts):
        _drop("duplicate_responses")
        return None
    order = sorted(range(len(responses)), key=lambda i: {"best": 0, "partial": 1, "worst": 2}.get(quals[i], 1))
    rec = {**base, "tags": _tags(data.get("tags"), md.get("tags", [])),
           "tasks": data.get("tasks") or md.get("tasks", []), "tools": md.get("tools", []),
           "prompt_messages": msgs, "prompt_text": render_messages(msgs, add_generation_prompt=True),
           "ranking": [quals[i] for i in order],
           "rationale": [responses[i].get("why", "") for i in order],
           **_flatten(msgs)}
    for n in range(want):
        rec[f"response_{n + 1}"] = texts[order[n]] if n < len(order) else None
    return rec


def to_records(seed: Seed, resp: Response) -> list[dict]:
    """Split a packed response into one record per member. Unpacked responses give a list of 0 or 1."""
    md = resp.metadata or {}
    if not md.get("packed"):
        r = to_record(seed, resp)
        return [r] if r else []
    data = _json(resp.text)
    if data is None:
        _drop("json_invalid")
        return []
    items = data.get("samples") or data.get("conversations") or []
    if not isinstance(items, list):
        _drop("pack_not_a_list")
        return []
    members, ids = md.get("members", []), md.get("custom_ids", [])
    if len(items) != len(members):
        # A short pack is salvageable -- take what lined up, count the rest -- but a long one means the model
        # lost track of the ordering and nothing can be trusted to belong to the spec it claims.
        _drop(f"pack_size:{len(items)}of{len(members)}")
        if len(items) > len(members):
            return []
    out = []
    for i, item in enumerate(items):
        if i >= len(members) or not isinstance(item, dict):
            continue
        sub = Response(ids[i], json.dumps(item, ensure_ascii=False), True, None,
                       resp.prompt_tokens // max(len(items), 1),
                       resp.completion_tokens // max(len(items), 1), resp.model, None, members[i])
        rec = to_record(seed, sub)
        if rec:
            out.append(rec)
    return out


def summary() -> str:
    if not STATS:
        return "no drops"
    return "drops: " + ", ".join(f"{k}={v}" for k, v in sorted(STATS.items(), key=lambda kv: -kv[1]))
