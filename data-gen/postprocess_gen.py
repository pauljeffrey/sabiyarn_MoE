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
import os
import random
import re
import traceback
import unicodedata
from pathlib import Path
from typing import Any, Optional

from providers.base import Response
from assemble import LABEL_TOKENS, AssemblyError, build_messages
from disassemble import looks_pre_assembled, turn_from_assistant
from longdocs import _text_from
from longdocs import splice as splice_document
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


# Set DATA_GEN_DEBUG_DIR to capture the raw response behind every drop. The counters tell you how often a
# thing goes wrong; only the raw text tells you why, and without it a reason like `prompt_messages_malformed`
# is unactionable -- it says the contract was broken without saying what the model sent instead.
_DEBUG_DIR = os.environ.get("DATA_GEN_DEBUG_DIR") or ""
_debug_seen: dict[str, int] = {}
_DEBUG_PER_REASON = 5  # enough to see the pattern, few enough to stay readable


def _drop(reason: str, raw: str = "", custom_id: str = "") -> None:
    STATS[reason] = STATS.get(reason, 0) + 1
    if not (_DEBUG_DIR and raw):
        return
    n = _debug_seen.get(reason, 0)
    if n >= _DEBUG_PER_REASON:
        return
    _debug_seen[reason] = n + 1
    try:
        d = Path(_DEBUG_DIR)
        d.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in reason)[:60]
        (d / f"{safe}.{n}.txt").write_text(f"# custom_id: {custom_id}\n# reason: {reason}\n\n{raw}",
                                          encoding="utf-8")
    except OSError:
        pass  # diagnostics must never break a run


def _score_of(r: dict) -> Optional[float]:
    v = r.get("score")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if 0.0 <= f <= 10.0 else None


def _fold(s: str) -> str:
    """Casefold and strip diacritics for a SUBSTRING test only.

    These languages are written with inconsistent accenting, so a discriminator phrase quoted by the
    generator will often differ from the reply by a diacritic or two. Matching loosely here makes the check
    about the words rather than the keystrokes; it is never used to produce output.
    """
    s = unicodedata.normalize("NFD", (s or "").casefold())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


def _capture(resp, tag: str) -> None:
    """Capture a raw response for a drop that was recorded deeper down, where resp is not in scope."""
    if _DEBUG_DIR:
        _drop(f"_raw:{tag}", resp.text or "", resp.custom_id)
        STATS.pop(f"_raw:{tag}", None)  # diagnostics only; must not appear in the yield accounting


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


# Label tokens, longest first so <lang_ID> is not shadowed by a shorter prefix.
_LABEL_BY_TOKEN = {v: k for k, v in LABEL_TOKENS.items()}
_LEADING_LABEL = re.compile(r"^\s*(" + "|".join(re.escape(t) for t in
                                                sorted(_LABEL_BY_TOKEN, key=len, reverse=True)) + r")\s*")
# Tasks whose answer legitimately IS a stream of `word <tag> LABEL` pairs, so <tag> there is content.
_SEQUENCE_TASKS = {"ner", "pos_tagging"}


def _objectify_turns(turns: Any) -> Any:
    """Recover turns the generator emitted as FLAT key/value arrays.

    Measured on the first live RL run: 3 of 24 responses came back as
        ["role", "user", "content", "I'm reading about..."]
    instead of {"role": "user", "content": "..."}. The content is intact and the conversion is exact for an
    even-length list whose alternating keys are strings, so dropping these threw away sound data over a
    JSON shape. An odd-length list, or one with a non-string key, is left alone for the assembler to reject.
    """
    if not isinstance(turns, list):
        return turns
    out = []
    for t in turns:
        if (isinstance(t, list) and len(t) % 2 == 0 and len(t) >= 2
                and all(isinstance(t[i], str) for i in range(0, len(t), 2))):
            out.append({t[i]: t[i + 1] for i in range(0, len(t), 2)})
        else:
            out.append(t)
    return out


def _candidate_text(content: Any, md: dict) -> str:
    """One RL candidate -> the assistant string a reward model will score.

    The generator sends `content` as a DICT OF FIELDS -- think, task_plan, input_lang, response -- because
    that is the contract every other part of the pipeline teaches it. This handler used to assume a
    pre-assembled string and crashed with AttributeError on the dict, which discarded whole samples that
    were otherwise excellent. Both shapes are now accepted; the dict goes through the same assembler as the
    prompt turns, so the markers in a candidate are built exactly like the markers everywhere else.
    """
    if isinstance(content, str):
        s = content.strip()
        # A STRINGIFIED field dict: the generator JSON-encodes the turn instead of nesting it. Measured on a
        # live run, 26 of 33 kept samples held raw JSON as the candidate text -- they passed every gate,
        # because nothing was checking that a candidate looks like an assistant turn, and would have taught
        # the model to emit JSON. Parse it back and assemble it like any other field dict.
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict) and (parsed.get("response") is not None
                                             or parsed.get("tool_call") is not None):
                content = parsed
            else:
                return _strip_invalid_closers(s).strip()
        else:
            return _strip_invalid_closers(s).strip()
    if not isinstance(content, dict):
        return ""
    turn = {**content, "role": "assistant"}
    try:
        built = build_messages(_fill_turn_defaults([turn], md))
    except (AssemblyError, Exception):  # noqa: BLE001
        return ""
    for m in built:
        if m.get("role") != "assistant":
            continue
        body = _strip_invalid_closers(m.get("content") or "").strip()
        # A tool-call turn carries the call in `tool_calls`, NOT in the content -- the chat template is what
        # turns it into text. So the assembled content alone ends at </task_plan> and the call is invisible.
        # Measured: 0 of 108 candidates contained <tool_call>, which made the entire tool-use half of the
        # corpus unscoreable by the verifiers, and made a tool-call candidate look like an empty answer.
        # Append it exactly as the template renders it, which is also exactly what the policy emits.
        for call in m.get("tool_calls") or []:
            fn = (call or {}).get("function") or {}
            name, args = fn.get("name") or "", fn.get("arguments")
            if not name:
                continue
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            body += f"<tool_call>{name}\n{args}</tool_call>"
        return body.strip()
    return ""


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
    sequence = bool(_SEQUENCE_TASKS & set(md.get("tasks") or [md.get("task")]))
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
        # `content` where `response` was asked for. An assistant turn with neither a response nor a tool call
        # is discarded as empty, and this was the single largest drop on a low-resource pilot: the generator
        # uses the `content` key it uses for user turns.
        if not t.get("response") and isinstance(t.get("content"), str) and t["content"].strip():
            if looks_pre_assembled(t["content"]):
                # The OLD finished-string shape. Its cause was the few-shot exemplars, which were rendered
                # verbatim from finished training records and so demonstrated exactly what the contract
                # forbids (see prompts._fewshot_block). That is fixed at the source; this recovers the cases
                # where the generator does it anyway, using the same parser that repaired the published corpus.
                parsed = turn_from_assistant({"content": t["content"],
                                              "tool_calls": ([{"function": call}] if call else [])},
                                             t.get("input_lang") or src_default)
                if parsed:
                    t.clear()
                    t.update(parsed)
                    call = t.get("tool_call") or {}
            elif not call:
                t["response"] = t.pop("content")
        # Invented CLOSING tags (</sentiment>, </intent>, </response>). _strip_invalid_closers is the one
        # sanctioned repair in this module and it was only wired into the LEGACY pre-assembled path, so the
        # current field-based contract never got it -- measured live as stray-tag drops on otherwise sound
        # conversations. A closer that is not in the tokenizer carries no information and its removal cannot
        # change meaning, which is what makes this different from stripping an opening marker.
        # A field also carrying its OWN wrapper (`think`: "...</think>", `response`: "<response>...") is pure
        # duplication -- the assembler emits the wrapper -- so removing it cannot change meaning.
        for key, wrappers in (("response", ("<response>",)), ("think", ("<think>", "</think>"))):
            if isinstance(t.get(key), str):
                v = _strip_invalid_closers(t[key])
                for w in wrappers:
                    v = v.replace(w, "")
                t[key] = v.strip()
            # <think> MUST NOT OUTGROW THE ANSWER. A reasoning block longer than the response it precedes is
        # spending the context window on narration, and at 32,768 tokens that budget is real. Trimmed rather
        # than dropped: the turn is otherwise sound, and the generator reaches for padding here more than
        # anywhere else. 120 words is roughly three full sentences of genuine deliberation.
        th, rp = t.get("think"), t.get("response")
        if isinstance(th, str) and th:
            tw = len(th.split())
            too_long = tw > 120 or (isinstance(rp, str) and rp and tw > max(60, len(rp.split())))
            if too_long:
                keep = " ".join(th.split()[:120])
                t["think"] = keep.rsplit(".", 1)[0] + "." if "." in keep else keep
                STATS["repair:trimmed_long_think"] = STATS.get("repair:trimmed_long_think", 0) + 1
    # A label token written INLINE instead of in the `label_token` field. The token is legal, the
        # position is not, and the assembler emits it in exactly the right place from the field -- so moving
        # it is information-preserving, unlike stripping it. 10 of 31 drops on the same pilot.
        resp = t.get("response")
        if isinstance(resp, str) and not t.get("label_token"):
            m = _LEADING_LABEL.match(resp)
            if m:
                t["label_token"] = _LABEL_BY_TOKEN[m.group(1)]
                t["response"] = resp[m.end():]
        # A `label_token` that is not a label token at all. Measured: the generator put the TASK NAME there
        # ("creative_writing"), which the assembler rightly refuses because it is not in the tokenizer. A
        # label token only means anything for the classification tasks, so for anything else the field is
        # noise and discarding it recovers an otherwise sound sample -- whereas keeping it loses the sample
        # and would, if it ever got through, teach the model to emit a junk token.
        # Accepted either as a key ("summary") or as the token itself ("<summary>"), matching the assembler.
        lt = t.get("label_token")
        if lt is not None and LABEL_TOKENS.get(lt, lt) not in LABEL_TOKENS.values():
            t.pop("label_token")
            STATS["repair:dropped_invalid_label_token"] = STATS.get("repair:dropped_invalid_label_token", 0) + 1
        if sequence:
            t["sequence_labels"] = True
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


_CONSTRAINT_KINDS = {"max_words", "exactly_sentences", "list_items", "forbidden_word", "required_word"}


def _clean_verifier(raw: Any, msgs: list[dict]) -> dict:
    """Keep only verifier fields that are well formed AND consistent with the prompt.

    A verifier is ground truth a program will trust without reading the answer, so a wrong one is actively
    harmful: it marks the good reply wrong and trains the policy towards the bad one. Everything here is
    therefore checked rather than taken on faith, and anything doubtful is dropped -- a missing field costs
    one check's weight, a wrong field inverts it.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    prompt = " ".join(m.get("content") or "" for m in msgs).lower()

    ans = raw.get("expected_answer")
    if isinstance(ans, (int, float)) or (isinstance(ans, str) and ans.strip()):
        s = str(ans).strip()
        if len(s) <= 40:
            out["expected_answer"] = s

    c = raw.get("constraint")
    if isinstance(c, dict) and c.get("kind") in _CONSTRAINT_KINDS:
        kind, val = c["kind"], c.get("value")
        if kind in ("max_words", "exactly_sentences", "list_items"):
            try:
                n = int(val)
                if 1 <= n <= 500:
                    out["constraint"] = {"kind": kind, "value": n}
            except (TypeError, ValueError):
                pass
        elif isinstance(val, str) and val.strip():
            word = val.strip()
            # A forbidden/required word the USER never mentioned is one the generator invented. The prompt is
            # the only evidence the constraint was actually imposed, so require it to appear there.
            if len(word) <= 40 and word.lower() in prompt:
                out["constraint"] = {"kind": kind, "value": word}

    els = raw.get("brief_elements")
    if isinstance(els, list):
        keep = [str(e).strip() for e in els if isinstance(e, (str, int, float)) and str(e).strip()]
        keep = [e for e in keep if len(e) <= 60][:8]
        if keep:
            out["brief_elements"] = keep

    ac = raw.get("answer_chunks")
    if isinstance(ac, (list, tuple)):
        ids = [str(x).strip().lower() for x in ac]
        ids = [i for i in ids if re.fullmatch(r"c\d{1,3}", i)]
        if ids:
            out["answer_chunks"] = ids[:6]

    disc = raw.get("discriminator")
    if isinstance(disc, dict):
        keep = {}
        for side in ("good", "bad"):
            v = disc.get(side)
            if isinstance(v, str) and 2 <= len(v.strip()) <= 120:
                keep[side] = v.strip()
        if keep:
            out["discriminator"] = keep

    mw = raw.get("max_words")
    try:
        n = int(mw)
        if 10 <= n <= 2_000:
            out["max_words"] = n
    except (TypeError, ValueError):
        pass
    return out


def _validate_conversation(msgs: list[dict], seed: Seed, *, ends_with: "str | tuple[str, ...]",
                           md: Optional[dict] = None) -> bool:
    md = md or {}
    conv = seed.conversation
    # User turns are what the generator is asked for and what is checked; assistant and tool turns follow
    # from them (see the seed's min_user_turns/max_user_turns).
    n_user = sum(1 for m in msgs if m["role"] == "user")
    uses_tools = any(m.get("tool_calls") for m in msgs) or any(m["role"] == "tool" for m in msgs)
    hi_turns = int(conv.get("max_user_turns_with_tools", 6) if uses_tools else conv.get("max_user_turns", 5))
    lo_turns = int(conv.get("min_user_turns", 3))
    # Prefer the bounds the REQUEST actually used. They differ from the seed-wide floor for low-resource
    # languages (capped lower) and for long-document samples (one or two turns beside a 2,000-word document),
    # and checking the seed-wide floor instead would reject every one of those as `user_turns:1`.
    if md.get("max_user_turns"):
        lo_turns, hi_turns = int(md.get("min_user_turns", 1)), int(md["max_user_turns"])
    if not (lo_turns <= n_user <= hi_turns):
        _drop(f"user_turns:{n_user}{'+tools' if uses_tools else ''}")
        return False
    max_total = int(conv.get("max_total_messages_with_tools", 20) if uses_tools
                    else conv.get("max_total_messages", 12))
    if len(msgs) > max_total:
        _drop(f"total_messages:{len(msgs)}{'+tools' if uses_tools else ''}")
        return False
    want = (ends_with,) if isinstance(ends_with, str) else ends_with
    if msgs[-1]["role"] not in want:
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
    if _degenerate_user_message(msgs, document=md.get("document")):
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


# Repetition floors. A looping passage is the signature of a generator writing in a language it half-knows: it
# runs out of vocabulary and cycles. Two measures, because they fail differently --
#   4-gram  catches a short clause repeated over and over (the Fon case: 0.14 against a healthy 0.96);
#   8-gram  catches whole sentences repeated verbatim, which is almost never legitimate prose even in an FAQ.
# The floors sit between the measured failures and the measured legitimate cases: on a 12-document pilot the
# worst acceptable document was a Twi how-to at 4-gram 0.705 / 8-gram 0.791, and the failures were at
# 0.14-0.19 / 0.17-0.22.
DEGENERATE_4GRAM_FLOOR = 0.60
DEGENERATE_8GRAM_FLOOR = 0.50
# TYPE-TOKEN RATIO: unique words over total words. The fourth gate, and the one that caught what the other
# three let through -- a live run's first printed sample was an Ewe document recycling about thirty words for
# 138, with "nu le hafié me" and "woanye dzɔ" reappearing in slightly different company each time:
#
#     d4 = 0.644 (above the 0.60 floor)   d8 = 0.901   no verbatim sentence repeat   ->  PASSED
#     type-token ratio = 0.188
#
# The n-gram measures miss it precisely because the SURROUNDINGS vary; the vocabulary does not. Calibrated
# against 55 documents already judged good, whose ratios run min 0.265, 5th percentile 0.283, median 0.379. A
# floor of 0.25 sits below every one of them and above the failure, and holds across the 60-420 word range these
# documents occupy (a fixed floor would be wrong over a much wider range, since longer text naturally repeats
# more function words).
DEGENERATE_TTR_FLOOR = 0.25
_TTR_MIN_WORDS = 80


# A whole sentence repeated word for word. This is the sharpest of the three, and the n-gram ratios miss it:
# a 121-word Ewe document with one 14-word sentence repeated THREE times scored 4-gram 0.63 and 8-gram 0.70 --
# above both floors, because the surrounding text is varied enough to dilute the ratio in a short document.
# Measured on the same batch, the four sound documents had 0 repeats across 16-18 sentences each, so this
# separates cleanly. 8 words is the floor for a "sentence" so that headings and stock phrases ("Thank you.",
# section labels) are not counted.
_SENTENCE = re.compile(r"[^.!?\n]+")
_MIN_SENTENCE_WORDS = 8


def repeated_sentence(text: str) -> Optional[str]:
    seen: dict[str, int] = {}
    for raw in _SENTENCE.findall(text or ""):
        s = " ".join(raw.split()).lower()
        if len(s.split()) < _MIN_SENTENCE_WORDS:
            continue
        seen[s] = seen.get(s, 0) + 1
        if seen[s] > 1:
            return s[:60]
    return None


def type_token_ratio(text: str) -> float:
    words = [w.lower().strip(".,;:!?'\u2019\"()") for w in (text or "").split()]
    words = [w for w in words if w]
    return len(set(words)) / max(len(words), 1)


def degenerate_reason(text: str, *, min_words: int = 40) -> Optional[str]:
    """Why this passage looks looped, or None if it is fine."""
    words = (text or "").split()
    if len(words) < min_words:
        return None
    if len(words) >= _TTR_MIN_WORDS:
        ttr = type_token_ratio(text)
        if ttr < DEGENERATE_TTR_FLOOR:
            return f"ttr={ttr:.2f}"
    d4 = _distinct_ngram(text, 4)
    if d4 < DEGENERATE_4GRAM_FLOOR:
        return f"4gram={d4:.2f}"
    d8 = _distinct_ngram(text, 8)
    if d8 < DEGENERATE_8GRAM_FLOOR:
        return f"8gram={d8:.2f}"
    dup = repeated_sentence(text)
    if dup:
        return f"repeated_sentence:{dup}"
    return None


def _degenerate_user_message(msgs: list[dict], floor: float = 0.75,
                             document: Optional[dict] = None) -> bool:
    """A looping user message ("mme nka mme nka nte nnyin eyenam eyọm nnyin mme nka...") is the signature of
    a model that does not actually know the language. Checked on USER turns specifically: repetition metrics
    over assistant responses alone scored those samples 0.974 and passed them.

    A SPLICED DOCUMENT is excluded. It is not the generator's spontaneous prose -- it is stage-1 output that
    legitimately repeats headings, action lines and figures across thousands of words, and a document that
    really does loop is caught in stage 1, where it is the document's problem rather than grounds for
    discarding a sound conversation built on it.
    """
    body = (document or {}).get("text") or ""
    for m in msgs:
        if m["role"] != "user":
            continue
        c = m.get("content") or ""
        if body:
            c = c.replace(body, " ")
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



def _pair_record(base: dict, md: dict, seed: Seed, msgs: list[dict], entry: Any,
                 verifier: dict, want: int) -> Optional[dict]:
    """One extra ranked pair, at an EARLIER turn, as its own training record.

    The prefix is truncated to the user turn this pair answers, so the record is a complete
    (prefix, chosen, rejected) triple on its own -- which is how multi-turn DPO is normally done. The
    generator was told to write each pair against the prefix as it stands at that point with the BEST earlier
    reply assumed, so the truncation matches what it was asked for.

    Returns None rather than raising: an unusable extra pair must not cost the main record.
    """
    if not isinstance(entry, dict):
        return None
    try:
        after = int(entry.get("after_user_turn"))
    except (TypeError, ValueError):
        return None
    responses = entry.get("responses") or []
    if not isinstance(responses, list) or len(responses) < 2:
        return None

    # Truncate the prefix to just after the `after`-th user turn.
    seen, cut = 0, None
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            seen += 1
            if seen == after:
                cut = i + 1
                break
    if cut is None or cut < 1:
        return None
    prefix = msgs[:cut]
    # The same shape rules as the final pair: fields assembled, both sides present and different.
    cands = []
    for r in responses:
        if not isinstance(r, dict):
            continue
        content = r.get("content")
        if content is None and (r.get("response") is not None or r.get("tool_call") is not None):
            content = {k: v for k, v in r.items() if k not in ("quality", "why", "role", "score")}
        txt = _candidate_text(content, md)
        if txt and "<|input_lang|>" in txt:
            cands.append({**r, "_text": txt})
    if len(cands) < 2:
        return None
    quals = [(c.get("quality") or "").strip().lower() for c in cands]
    if "best" not in quals or "worst" not in quals:
        return None
    texts = [c["_text"] for c in cands]
    if len(set(texts)) < len(texts):
        return None
    order = sorted(range(len(cands)), key=lambda i: {"best": 0, "partial": 1, "worst": 2}.get(quals[i], 1))
    sc = [_score_of(cands[i]) for i in order]
    if sc[0] is not None and sc[-1] is not None and (sc[0] - sc[-1]) < 2.0:
        return None

    rec = {**base,
           "id": f"{base.get('id', '')}__t{after}",
           "tags": base.get("tags") or [], "tasks": base.get("tasks") or [],
           "tools": md.get("tools", []), "verifier": verifier,
           "prompt_messages": prefix,
           "prompt_text": render_messages(prefix, add_generation_prompt=True),
           "ranking": [quals[i] for i in order],
           "scores": sc,
           "rationale": [cands[i].get("why", "") for i in order],
           "paired_turn": after,
           **_flatten(prefix)}
    for n in range(want):
        rec[f"response_{n + 1}"] = texts[order[n]] if n < len(order) else None
    return rec


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
    # TODAY'S DATE. Without it the model cannot tell that a tool result from "12 March" is stale, which makes
    # stale_data_awareness unlearnable -- it would have to guess the gap instead of working it out. It also
    # grounds every "today", "this week" and "last month" in the corpus, and it is what a real deployment
    # puts in the system prompt anyway. Costs about 12 tokens.
    today = md.get("today")
    date_line = f" Today is {today}." if today else ""
    # Compact JSON: indent=2 cost ~40% more tokens for no benefit to the model.
    return (f"{identity}{date_line} You have these tools:\n"
            + json.dumps(catalogue, ensure_ascii=False, separators=(",", ":")))


def _tags(raw: Any, fallback: list[str]) -> list[str]:
    got = [t for t in (raw or []) if t in TAGS]
    return sorted(set(got) | set(fallback)) or list(fallback)


def to_record(seed: Seed, resp: Response) -> Optional[dict]:
    """Never raises. A single malformed response must not be able to kill a 500k-request run."""
    try:
        return _to_record(seed, resp)
    except Exception as exc:  # noqa: BLE001
        # The class alone is unactionable -- AttributeError could be anywhere in 600 lines. Capture the
        # traceback with the raw response so the bug is locatable from one run.
        _drop(f"postprocess_error:{type(exc).__name__}", traceback.format_exc() + "\n\n--- raw ---\n"
              + (resp.text or "")[:8000], resp.custom_id)
        return None


def _to_record(seed: Seed, resp: Response) -> Optional[dict]:
    md = resp.metadata or {}
    data = _json(resp.text)
    if data is None:
        _drop("json_invalid", resp.text or "", resp.custom_id)
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
        # `text` arriving as a list of paragraphs or an object of section->prose is the same document in a
        # different container, and .strip() on it raised AttributeError -- which the catch-all turned into an
        # opaque `postprocess_error` and threw the document away. Reuse the document reader, which already
        # flattens every container longdocs sees.
        text = _text_from(data).strip()
        if len(text.split()) < 60:
            _drop("too_short")
            return None
        if data.get("language_self_check") is False:
            _drop("language_self_check_false")
            return None
        bad = degenerate_reason(text)
        if bad:
            # THE most important gate on this path, and it did not exist. Measured on a 12-document pilot: both
            # Fon documents were the same clause repeated for 400 words, distinct-4gram 0.143 and 0.191 -- and
            # they self-reported `confidence` 0.9 and 0.7, so the generator's own estimate is worthless for a
            # language it does not really know. 30,000 documents like that do not merely waste the run: a model
            # pretrained on looped text learns to loop.
            _drop(f"degenerate_text:{bad}")
            return None
        return {**base, "genre": md.get("genre", ""), "title": (data.get("title") or "").strip(),
                "distinct_4gram": round(_distinct_ngram(text), 3), "text": text}

    if seed.kind == "sft":
        # `turns` is the current contract (structured fields); `messages` is the legacy pre-assembled shape,
        # still accepted so old batch output can be post-processed.
        turns = data.get("turns") or data.get("messages") or data.get("conversation")
        # Two-stage document samples: the generator wrote __DOCUMENT__ in the user turn and the real document
        # travels in metadata, so put it back before anything is validated or rendered.
        if md.get("document"):
            turns, err = splice_document(turns, md["document"])
            if err:
                _drop(f"document:{err}")
                return None
        if isinstance(turns, list) and any(isinstance(t, dict) and ("task_plan" in t or "response" in t
                                                                   or "tool_call" in t) for t in turns):
            msgs = _messages_from_turns(_fill_turn_defaults(_objectify_turns(turns), md))
        elif data.get("turns") is not None:
            msgs = _messages_from_turns(_fill_turn_defaults(_objectify_turns(data["turns"]), md))
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
        doc = md.get("document") or {}
        return {**base, "tags": _tags(data.get("tags"), md.get("tags", [])),
                "tasks": data.get("tasks") or md.get("tasks", []),
                "tools": md.get("tools", []), "distractor_tools": md.get("distractor_tools", []),
                "io_markers_ok": _markers_ok(msgs, md.get("expect_markers")),
                # auditable, so a length or language regression in stage 1 is visible in the corpus itself
                **({"doc_words": doc.get("words"), "doc_lang": doc.get("doc_lang"),
                    "doc_form": doc.get("form")} if doc else {}),
                "messages": msgs, "text": render_messages(msgs), **_flatten(msgs)}

    # rl
    if data.get("prompt_turns") is not None:
        msgs = _messages_from_turns(_fill_turn_defaults(_objectify_turns(data["prompt_turns"]), md))
    else:
        msgs = _clean_messages(data.get("prompt_messages"))
    msgs_pre = msgs
    if msgs:
        # The generator often writes the final assistant reply INTO the prefix as well as into `responses`.
        # The prefix must stop where the candidates begin, and the candidates are the better source: they
        # come as a ranked pair. So a trailing assistant turn is redundant and dropping it recovers the
        # sample -- 20 of 39 drops on a live run were exactly this.
        # Only a trailing ANSWER is redundant. A trailing TOOL CALL is not: it is the retrieval step, and
        # trimming it leaves a RAG prefix as a bare user turn with the tool result gone -- which silently
        # turned document-grounded samples into something else and made every grounding check unscoreable.
        while (len(msgs) > 1 and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls")):
            msgs = msgs[:-1]
            STATS["repair:trimmed_trailing_answer"] = STATS.get("repair:trimmed_trailing_answer", 0) + 1
        if not msgs:
            msgs = msgs_pre  # nothing but assistant turns: let the validator reject it with a real reason
    if msgs is None:
        # Both branches must record. Previously the prompt_turns branch returned None with no counter
        # incremented, so those drops vanished from the accounting entirely and the yield looked better
        # than it was by however many of them there were.
        _drop("prompt_messages_malformed" if data.get("prompt_turns") is None else "prompt_turns_unusable",
              resp.text or "", resp.custom_id)
        return None
    # A RAG or tool prefix legitimately ends with the TOOL RESULT: the candidates are the answer read off
    # that result, which is the whole lesson of those tasks. Requiring "user" rejected every one of them.
    if not _validate_conversation(msgs, seed, ends_with=("user", "tool"), md=md):
        _capture(resp, "validate_conversation")
        return None
    responses = data.get("responses") or []
    want = int(seed.conversation["responses_per_prompt"])
    # Assemble first, then filter on what came out: a candidate whose fields will not assemble is as
    # useless as an empty one, and deciding that here keeps the two checks from disagreeing.
    # The candidate is expected as {"content": <fields>, "quality": "best"|"worst", "why": ...}. The
    # generator sometimes emits the assistant TURN directly as the list item instead, with the fields at the
    # top level and no wrapper. The fields are all there, so reading the item as its own content recovers it.
    def _content_of(r: dict) -> Any:
        if r.get("content") is not None:
            return r["content"]
        if r.get("response") is not None or r.get("tool_call") is not None:
            return {k: v for k, v in r.items() if k not in ("quality", "why", "role")}
        return None

    responses = [{**r, "_text": _candidate_text(_content_of(r), md)}
                 for r in responses if isinstance(r, dict)]
    responses = [r for r in responses if r["_text"]]
    # A candidate must LOOK like an assembled assistant turn. Without this, a stringified JSON blob counted
    # as a valid candidate and reached the corpus -- 26 of 33 on one run -- because every other check only
    # asked whether the text was non-empty and different from its sibling.
    #
    # The marker to test is <|input_lang|>, which opens EVERY assembled assistant turn. Testing
    # <|target_lang|> instead rejected legitimate tool-call candidates, which end at <task_plan> and carry no
    # target language or <response> at all -- they are a call, not a reply.
    bad = [r for r in responses if "<|input_lang|>" not in r["_text"]]
    if bad:
        _drop("candidate_not_an_assembled_turn", bad[0]["_text"][:2000], resp.custom_id)
        return None
    if len(responses) < 2:
        _drop("too_few_responses", resp.text or "", resp.custom_id)
        return None
    # The labels the contract asks for are best/worst, but a generator reaches for its own vocabulary often
    # enough to matter (7 of 44 drops on one run). These are synonyms, not judgements, so mapping them is
    # information-preserving -- unlike guessing a label that was never given.
    _QUAL = {"chosen": "best", "good": "best", "better": "best", "preferred": "best", "correct": "best",
             "rejected": "worst", "bad": "worst", "worse": "worst", "incorrect": "worst", "wrong": "worst",
             "partial": "partial", "best": "best", "worst": "worst"}
    quals = [_QUAL.get((r.get("quality") or "").strip().lower(), (r.get("quality") or "").strip().lower())
             for r in responses]
    if "best" not in quals or "worst" not in quals:
        _drop("no_best_or_worst")
        return None
    texts = [r["_text"] for r in responses]
    if len(set(texts)) < len(texts):
        _drop("duplicate_responses")
        return None
    order = sorted(range(len(responses)), key=lambda i: {"best": 0, "partial": 1, "worst": 2}.get(quals[i], 1))

    # VERIFY THE DISCRIMINATOR. The generator was asked to name the one concrete phrase that makes the worse
    # reply worse; this checks the claim against the replies it actually wrote. It is the cheapest quality
    # gate available -- a substring test, no tokens -- and it catches the failure that matters most: two
    # replies that differ in wording but not in substance, which teach the policy nothing and which no
    # programmatic reward can separate afterwards.
    #
    # Without a judge in the loop this is also what makes the factual tasks scoreable at all: the phrase is
    # the ground truth, so "is the right value present" and "is the invented one absent" become lookups.
    verifier = rec_verifier = _clean_verifier(data.get("verifier"), msgs)
    disc = (rec_verifier or {}).get("discriminator") or {}
    if disc:
        best_t = texts[order[0]]
        worst_t = texts[order[-1]]
        bad_sides = []
        g, b = disc.get("good"), disc.get("bad")
        if g and not (_fold(g) in _fold(best_t) and _fold(g) not in _fold(worst_t)):
            bad_sides.append("good")
        if b and not (_fold(b) in _fold(worst_t) and _fold(b) not in _fold(best_t)):
            bad_sides.append("bad")
        if bad_sides:
            _drop(f"discriminator_does_not_hold:{'+'.join(bad_sides)}", resp.text or "", resp.custom_id)
            return None
    # A pair whose two sides scored nearly the same is a pair that teaches almost nothing, and the generator
    # has already told us so. Dropping it here costs no tokens and keeps the margin honest.
    sc = [_score_of(responses[i]) for i in order]
    if sc[0] is not None and sc[-1] is not None and (sc[0] - sc[-1]) < 2.0:
        _drop(f"score_margin_too_small:{sc[0]:.0f}v{sc[-1]:.0f}")
        return None

    rec = {**base, "tags": _tags(data.get("tags"), md.get("tags", [])),
           "tasks": data.get("tasks") or md.get("tasks", []), "tools": md.get("tools", []),
           # The ground truth the programmatic scorers need (rl/verifiers.py). Kept even when empty so the
           # column exists for every row, and sanitised: a verifier the generator invented wrongly would
           # score the GOOD answer as bad, which is worse than having none.
           "verifier": verifier,
           "prompt_messages": msgs, "prompt_text": render_messages(msgs, add_generation_prompt=True),
           "ranking": [quals[i] for i in order],
           # The generator's own 0-10 scores, in the same order as the responses. DPO needs only the
           # ordering, which `ranking` carries; the scores give a MARGIN, which is what lets a pair whose
           # two sides are nearly as good as each other be filtered out later rather than trained on.
           "scores": [_score_of(responses[i]) for i in order],
           "rationale": [responses[i].get("why", "") for i in order],
           **_flatten(msgs)}
    for n in range(want):
        rec[f"response_{n + 1}"] = texts[order[n]] if n < len(order) else None

    # EXTRA PAIRS at earlier turns. Each becomes its own (prefix, chosen, rejected) record, which is where
    # the extra data comes from: the prefix was written once, so a second ranked turn is close to free.
    extra = []
    for entry in (data.get("turn_responses") or []):
        er = _pair_record(base, md, seed, msgs, entry, verifier, want)
        if er is not None:
            extra.append(er)
        else:
            STATS["extra_pair_unusable"] = STATS.get("extra_pair_unusable", 0) + 1
    if extra:
        # Deduplicate against the final pair: a generator that repeats the last turn here adds nothing.
        final_key = rec.get("response_1")
        extra = [e for e in extra if e.get("response_1") != final_key]
        if extra:
            rec["_extra_pairs"] = extra
            STATS["repair:extra_pairs_kept"] = STATS.get("repair:extra_pairs_kept", 0) + len(extra)
    return rec


def to_records(seed: Seed, resp: Response) -> list[dict]:
    """Split a packed response into one record per member. Unpacked responses give a list of 0 or 1."""
    md = resp.metadata or {}
    if not md.get("packed"):
        r = to_record(seed, resp)
        if not r:
            return []
        return [r] + r.pop("_extra_pairs", [])
    data = _json(resp.text)
    if data is None:
        _drop("json_invalid")
        return []
    # `documents` is the pretrain pack's key; `samples`/`conversations` are sft/rl's.
    items = data.get("samples") or data.get("conversations") or data.get("documents") or []
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
    """Drops and repairs reported separately. A repair KEPT a sample, so listing it among the drops overstates
    the failure rate and hides which reasons are actually costing data."""
    if not STATS:
        return "no drops"
    drops = {k: v for k, v in STATS.items() if not k.startswith("repair:")}
    fixed = {k[len("repair:"):]: v for k, v in STATS.items() if k.startswith("repair:")}
    parts = []
    if drops:
        parts.append("drops: " + ", ".join(f"{k}={v}" for k, v in sorted(drops.items(), key=lambda kv: -kv[1])))
    if fixed:
        parts.append("repaired: " + ", ".join(f"{k}={v}" for k, v in sorted(fixed.items(), key=lambda kv: -kv[1])))
    return "  ".join(parts) or "no drops"
