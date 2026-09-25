#!/usr/bin/env python3
"""Compare generator models on the data they actually produced.

    python compare_models.py                          # every sft__* namespace found
    python compare_models.py --kind sft --langs yor,hau

Generate the SAME plan rows with each model under its own `--run-tag`, then run this. Because plan rows are
deterministic, every model answered an identical set of prompts, so the numbers are directly comparable.

What is measured, and why these and not "does it look good":

  format      invented closing tags, missing <response>, malformed tool calls -- structural correctness the
              target model will imitate verbatim, so errors here are trained in.
  thinking    share of <think> blocks that are actually English, and their length. English thinking is a hard
              requirement; a model that reasons in the target language has failed the spec.
  tools       call rate, unknown-tool rate, and whether required arguments are present. Also whether the
              DISTRACTOR tools got called, which is the direct measure of tool-selection discipline.
  language    orthography: does Yoruba have its diacritics, Igbo its dotted vowels, Hausa its hooked letters?
              A model that writes unmarked ASCII Yoruba is producing unusable data however fluent it reads.
              Plus English leakage, which is how low-resource generation usually fails.
  repetition  distinct-4gram ratio over assistant responses. Catches the degenerate looping that makes a
              corpus worthless without being obvious in a spot check.
  economy     cost per KEPT sample -- the only cost number that matters, since rejects are still billed.

None of this replaces a speaker reading samples. It is here to stop a model that fails measurably from ever
reaching a speaker's desk.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate import OUT_ROOT  # noqa: E402
from schemas.seed import Seed  # noqa: E402

VALID_CLOSERS = {"</think>", "</task_plan>", "</tool_call>", "</tool_response>", "</context>", "</s>"}
ANY_CLOSER = re.compile(r"</[A-Za-z_|][^>\s]*>")
THINK = re.compile(r"<think>(.*?)</think>", re.S)
RESPONSE = re.compile(r"<response>(.*)", re.S)

# Orthographic marks that MUST appear in real text in these languages.
DIACRITICS = {
    "yor": set("ọẹṣàáèéìíòóùúńǹ"),
    "ibo": set("ịọụṅ"),
    "hau": set("ɓɗƙƴ"),
    "twi": set("ɛɔ"), "aka": set("ɛɔ"),
    "ewe": set("ɖƒŋɣʋ"), "fon": set("ɖɛɔ"),
    "ful": set("ɓɗƴŋ"), "fuv": set("ɓɗƴŋ"),
    "efi": set("ọụ"), "urh": set("ẹọ"),
    "pcm": set(),   # English-lexified: no diacritics expected
}
ENGLISH_STOPWORDS = {"the", "and", "is", "are", "of", "to", "in", "that", "this", "for", "with", "you",
                     "it", "on", "as", "be", "have", "has", "will", "can", "not", "but", "they", "we"}


def _load(namespace: str, langs: Optional[list[str]]) -> list[dict]:
    root = OUT_ROOT / namespace
    if not root.exists():
        return []
    out = []
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if langs and lang_dir.name not in langs:
            continue
        for shard in sorted(lang_dir.glob("shard-*.jsonl")):
            for line in shard.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    return out


def _distinct_ngram(text: str, n: int = 4) -> float:
    w = text.split()
    if len(w) <= n:
        return 1.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams)


def _is_english(text: str) -> bool:
    """Crude but effective for <think> blocks: mostly ASCII and English function words present."""
    if not text.strip():
        return False
    ascii_share = sum(ord(c) < 128 for c in text) / len(text)
    words = set(re.findall(r"[a-z]+", text.lower()))
    return ascii_share > 0.95 and len(words & ENGLISH_STOPWORDS) >= 2


def analyse(recs: list[dict], seed: Seed) -> dict[str, Any]:
    tool_names = {t.name for t in seed.tools}
    required_args = {t.name: set(t.parameters.get("required", [])) for t in seed.tools}
    m: dict[str, Any] = {
        "n": len(recs), "models": Counter(r.get("model", "?") for r in recs),
        "langs": Counter(r["lang"] for r in recs),
        "bad_closers": 0, "missing_response": 0,
        "think_total": 0, "think_english": 0, "think_chars": [],
        "with_tools": 0, "calls": 0, "unknown_tool": 0, "missing_args": 0, "distractor_called": 0,
        "user_turns": Counter(), "total_msgs": Counter(),
        "conf": [], "resp_chars": [], "distinct4": [],
        "diacritics_ok": defaultdict(lambda: [0, 0]), "eng_leak": defaultdict(list),
        "tags": Counter(),
        # completeness: a "tool result" of "ok" or a two-word assistant reply is technically valid and
        # useless as training data, so measure substance, not just presence.
        "user_chars": [], "tool_chars": [], "tool_english": [0, 0],
        "io_dirs": Counter(), "io_markers_ok": [0, 0], "think_after_tool": 0, "think_before_tool": 0,
        "stub_tool_results": 0, "stub_responses": 0,
    }
    for r in recs:
        msgs = r.get("messages") or r.get("prompt_messages") or []
        m["user_turns"][sum(1 for x in msgs if x["role"] == "user")] += 1
        m["total_msgs"][len(msgs)] += 1
        if r.get("confidence") is not None:
            m["conf"].append(r["confidence"])
        for t in r.get("tags", []):
            m["tags"][t] += 1
        distractors = set(r.get("distractor_tools") or [])
        if r.get("io_direction"):
            m["io_dirs"][r["io_direction"]] += 1
        if r.get("io_markers_ok") is not None:
            m["io_markers_ok"][0] += bool(r["io_markers_ok"])
            m["io_markers_ok"][1] += 1
        used_tool = False
        prev_was_tool_result = False
        for msg in msgs:
            content = msg.get("content") or ""
            if msg["role"] == "user":
                m["user_chars"].append(len(content))
            if msg["role"] == "tool":
                m["tool_chars"].append(len(content))
                if len(content.strip()) < 25:
                    m["stub_tool_results"] += 1          # "ok" / "success" teaches nothing
                words = re.findall(r"[a-zA-Z]+", content.lower())
                m["tool_english"][1] += 1
                m["tool_english"][0] += bool(words) and len(set(words) & ENGLISH_STOPWORDS) >= 1
            if msg["role"] == "assistant" and THINK.search(content):
                if prev_was_tool_result:
                    m["think_after_tool"] += 1
                elif msg.get("tool_calls"):
                    m["think_before_tool"] += 1
            prev_was_tool_result = msg["role"] == "tool"
            m["bad_closers"] += sum(1 for c in ANY_CLOSER.findall(content) if c not in VALID_CLOSERS)
            for th in THINK.findall(content):
                m["think_total"] += 1
                m["think_chars"].append(len(th))
                if _is_english(th):
                    m["think_english"] += 1
            for call in msg.get("tool_calls") or []:
                used_tool = True
                m["calls"] += 1
                fn = call.get("function", {})
                name = fn.get("name", "")
                if name not in tool_names:
                    m["unknown_tool"] += 1
                elif not required_args.get(name, set()) <= set((fn.get("arguments") or {}).keys()):
                    m["missing_args"] += 1
                if name in distractors:
                    m["distractor_called"] += 1
            if msg["role"] == "assistant" and not msg.get("tool_calls"):
                body = RESPONSE.search(content)
                if body:
                    txt = body.group(1)
                    m["resp_chars"].append(len(txt))
                    if len(txt.strip()) < 40:
                        m["stub_responses"] += 1
                    m["distinct4"].append(_distinct_ngram(txt))
                    want = DIACRITICS.get(r["lang"], set())
                    if want:
                        ok, tot = m["diacritics_ok"][r["lang"]]
                        m["diacritics_ok"][r["lang"]] = [ok + bool(set(txt.lower()) & want), tot + 1]
                    if r["lang"] != "pcm":
                        words = re.findall(r"[a-zA-Z]+", txt.lower())
                        if len(words) >= 12:
                            m["eng_leak"][r["lang"]].append(
                                sum(w in ENGLISH_STOPWORDS for w in words) / len(words))
        m["with_tools"] += used_tool
        last = next((x for x in reversed(msgs) if x["role"] == "assistant" and not x.get("tool_calls")), None)
        if last and not RESPONSE.search(last.get("content") or ""):
            m["missing_response"] += 1
    return m


def _mean(xs, default=0.0):
    return sum(xs) / len(xs) if xs else default


def report(results: dict[str, dict[str, Any]]) -> None:
    names = list(results)
    w = max(len(n) for n in names) + 2

    def row(label: str, fn, fmt="{}"):
        cells = "".join(f"{fmt.format(fn(results[n])):>16}" for n in names)
        print(f"  {label:<30}{cells}")

    print(f"\n{'':32}" + "".join(f"{n:>16}" for n in names))
    print("  " + "-" * (30 + 16 * len(names)))
    print("  KEPT SAMPLES")
    row("records analysed", lambda m: m["n"])
    row("languages", lambda m: len(m["langs"]))
    print("  FORMAT (lower is better)")
    row("invented closing tags", lambda m: m["bad_closers"])
    row("final turn missing <response>", lambda m: m["missing_response"])
    print("  THINKING (English is required)")
    row("<think> blocks", lambda m: m["think_total"])
    row("... in English", lambda m: f"{m['think_english']}/{m['think_total']}"
        f" ({m['think_english']/max(m['think_total'],1):.0%})")
    row("mean <think> chars", lambda m: f"{_mean(m['think_chars']):.0f}")
    print("  TOOLS")
    row("samples using tools", lambda m: f"{m['with_tools']}/{m['n']}"
        f" ({m['with_tools']/max(m['n'],1):.0%})")
    row("tool calls", lambda m: m["calls"])
    row("unknown tool name", lambda m: m["unknown_tool"])
    row("missing required args", lambda m: m["missing_args"])
    row("DISTRACTOR called (bad)", lambda m: m["distractor_called"])
    print("  SHAPE")
    row("user turns (mode)", lambda m: m["user_turns"].most_common(1)[0][0] if m["user_turns"] else 0)
    row("total messages (mean)", lambda m: f"{_mean([k*v for k,v in m['total_msgs'].items()])/max(_mean(list(m['total_msgs'].values())),1e-9):.1f}"
        if m["total_msgs"] else "0")
    print("  LANGUAGE")
    for lang in sorted({l for m in results.values() for l in m["diacritics_ok"]}):
        row(f"diacritics present: {lang}", lambda m, L=lang: (
            f"{m['diacritics_ok'][L][0]}/{m['diacritics_ok'][L][1]}"
            f" ({m['diacritics_ok'][L][0]/max(m['diacritics_ok'][L][1],1):.0%})"
            if L in m["diacritics_ok"] else "-"))
    row("English leak (mean, non-pcm)", lambda m: f"{_mean([x for v in m['eng_leak'].values() for x in v]):.1%}")
    print("  LANGUAGE DIRECTION (target: even across 5)")
    row("directions seen", lambda m: len(m["io_dirs"]))
    row("most/least common", lambda m: (f"{m['io_dirs'].most_common(1)[0][1]}/"
                                        f"{min(m['io_dirs'].values())}" if m["io_dirs"] else "-"))
    row("markers match direction", lambda m: (f"{m['io_markers_ok'][0]}/{m['io_markers_ok'][1]}"
                                              f" ({m['io_markers_ok'][0]/max(m['io_markers_ok'][1],1):.0%})"
                                              if m["io_markers_ok"][1] else "-"))
    print("  COMPLETENESS (substance, not presence)")
    row("mean user message chars", lambda m: f"{_mean(m['user_chars']):.0f}")
    row("mean tool result chars", lambda m: f"{_mean(m['tool_chars']):.0f}")
    row("stub tool results (<25 ch)", lambda m: m["stub_tool_results"])
    row("stub responses (<40 ch)", lambda m: m["stub_responses"])
    row("tool results in English", lambda m: (f"{m['tool_english'][0]}/{m['tool_english'][1]}"
                                              f" ({m['tool_english'][0]/max(m['tool_english'][1],1):.0%})"
                                              if m["tool_english"][1] else "-"))
    row("think BEFORE tool call", lambda m: m["think_before_tool"])
    row("think AFTER tool result", lambda m: m["think_after_tool"])
    print("  QUALITY")
    row("distinct-4gram (higher=better)", lambda m: f"{_mean(m['distinct4'], 1.0):.3f}")
    row("mean response chars", lambda m: f"{_mean(m['resp_chars']):.0f}")
    row("mean self-confidence", lambda m: f"{_mean(m['conf']):.2f}" if m["conf"] else "-")
    row("distinct tags covered", lambda m: len(m["tags"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="sft", choices=["pretrain", "sft", "rl"])
    ap.add_argument("--langs", default=None)
    ap.add_argument("--namespaces", default=None,
                    help="comma list, e.g. sft__gemma4,sft__llama33. Default: every <kind>__* on disk")
    a = ap.parse_args()
    langs = [s.strip() for s in a.langs.split(",")] if a.langs else None
    seed = Seed.load(a.kind)

    if a.namespaces:
        spaces = [s.strip() for s in a.namespaces.split(",")]
    else:
        spaces = sorted(p.name for p in OUT_ROOT.iterdir()
                        if p.is_dir() and (p.name == a.kind or p.name.startswith(a.kind + "__")))
    results = {}
    for ns in spaces:
        recs = _load(ns, langs)
        if recs:
            results[ns.replace(a.kind + "__", "")] = analyse(recs, seed)
        else:
            print(f"(skipping {ns}: no records)")
    if not results:
        raise SystemExit("no data found. Generate with --run-tag first.")
    report(results)
    print("\nNone of this replaces a speaker reading samples; it exists to stop a measurably bad model from")
    print("reaching that point. Read 10 samples from whichever model wins before committing to it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
