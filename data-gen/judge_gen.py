#!/usr/bin/env python3
"""Second batch pass: judge the RL candidates and reduce them to response_1 (chosen) / response_2 (rejected).

    python judge_gen.py --dry-run --limit 3
    python judge_gen.py --provider together --batch            # half price, 24h window
    python judge_gen.py --provider together --fetch <batch_id> --push
    python judge_gen.py --provider openrouter --limit 500 --push

Why a separate pass rather than trusting the generator's own best/partial/worst labels: the model that wrote
three candidates is a poor judge of its own work, and its labels are an intention ("I will now write a bad
one"), not an assessment. A second, independent read is what turns them into a preference signal.

Three things this does to keep the judgement honest:

  * ANONYMISED AND SHUFFLED. Candidates are relabelled A/B/C in a per-sample random order, and the
    generator's own quality labels are not shown. LLM judges have a strong, well-documented position bias;
    if the generator's "best" were always A, the judge would agree with itself and teach nothing.
  * A DIFFERENT MODEL. Default judge is a different model from the generator (self-preference bias is real).
    `--model` overrides; using the same model is allowed but logged as a warning.
  * EXPLICIT CRITERIA, RANKED. Honesty about the knowledge boundary outranks fluency, which outranks style.
    A fluent confabulation must lose to a blunt "I don't know", and an over-refusal must lose to a correct
    answer that was actually available. Without that ordering a judge optimises for polish.

The judge may also REWRITE the winner (`--rewrite`), which usually produces a better chosen response than
any candidate. That makes the pair partly distillation rather than pure preference learning: the chosen side
then carries the judge model's style as well as its correctness. Off by default for that reason -- turn it on
deliberately, and keep some non-rewritten pairs in the mix so the model is not only learning style.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate import OUT_ROOT, ShardWriter, done_ids  # noqa: E402
from providers import get_provider                     # noqa: E402
from providers.base import Request, Response           # noqa: E402
from schemas.seed import Seed                          # noqa: E402

JUDGED_KIND = "rl_judged"
LABELS = "ABCDE"

CRITERIA = """\
Rank the candidate replies by these criteria, IN THIS ORDER. A lower criterion never overturns a higher one.

1. HONESTY AT THE KNOWLEDGE BOUNDARY (decisive).
   - A reply that invents a fact, or answers from a tool result that does not actually contain the answer,
     LOSES to a reply that plainly says it does not know. Fluency does not rescue a confabulation.
   - But the reverse also holds: if the answer WAS available -- in the supplied document, in the tool result,
     or from ordinary world knowledge -- then refusing or hedging LOSES to answering. Over-refusal is a
     failure, not safety.
2. GROUNDING. Claims traceable to the tool result or document beat unsupported assertions. Citing where
   something came from beats asserting it.
3. CORRECT TOOL BEHAVIOUR. The right tool, sane arguments, and NOT calling the irrelevant tools in the
   catalogue. A reply that answers without calling a tool it clearly needed is worse than one that calls it.
4. TASK COMPLIANCE. Actually does what was asked, in the language that was asked, in the required format.
5. LANGUAGE QUALITY. Fluent, natural, correct orthography and diacritics for the target language.
6. BREVITY AND TONE. No sycophancy, no padding, no restating the question. Shorter wins when equal.

Reasoning in `why` must be in ENGLISH and name the criterion that decided it."""


def _iter_rl_records(langs: Optional[list[str]] = None) -> Iterator[dict]:
    root = OUT_ROOT / "rl"
    if not root.exists():
        raise SystemExit(f"no RL data at {root}. Run: python generate.py --kind rl ... first")
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if langs and lang_dir.name not in langs:
            continue
        for shard in sorted(lang_dir.glob("shard-*.jsonl")):
            with shard.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue


def _candidates(rec: dict) -> list[str]:
    out = []
    for i in range(1, 6):
        v = rec.get(f"response_{i}")
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


def build_judge_request(rec: dict, seed: Seed) -> Optional[Request]:
    cands = _candidates(rec)
    if len(cands) < 2:
        return None
    rng = random.Random(rec["id"])              # deterministic shuffle: same order on every re-run
    order = list(range(len(cands)))
    rng.shuffle(order)
    shown = "\n\n".join(f"--- CANDIDATE {LABELS[i]} ---\n{cands[j]}" for i, j in enumerate(order))
    tools = rec.get("tools") or []
    distract = rec.get("distractor_tools") or []

    system = (
        f"{seed.target_model.get('philosophy', '')}\n\n"
        "You are judging candidate replies written for this model's preference training. You did not write "
        "them. Judge only the FINAL assistant reply; the conversation before it is fixed and shared.\n\n"
        f"{CRITERIA}\n\nReturn strict JSON only."
    )
    user = f"""CONVERSATION SO FAR (ends with the user turn the candidates answer):
{rec.get('prompt_text') or json.dumps(rec.get('prompt_messages', []), ensure_ascii=False)}

Language of this conversation: {rec.get('lang')}
Tools that were in scope: {', '.join(tools) if tools else '(none)'}
Of those, deliberately irrelevant: {', '.join(distract) if distract else '(none)'}

{shown}

Return JSON:
{{"winner": "<letter>", "loser": "<letter>", "deciding_criterion": <1-6>,
  "why": "<one or two sentences in English naming what decided it>",
  "confidence": <float 0-1 that this ranking is correct>,
  "both_bad": <true if EVERY candidate fails criterion 1 or 2 -- then the pair is discarded>}}"""

    return Request(custom_id=f"judge__{rec['id']}", max_tokens=700, temperature=0.0,
                   messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                   response_format={"type": "json_object"},
                   metadata={"record": rec, "order": order})


def apply_verdict(resp: Response, *, min_confidence: float = 0.0) -> Optional[dict]:
    """Judge response -> a 2-candidate preference record, or None if it should be dropped."""
    rec = (resp.metadata or {}).get("record")
    order = (resp.metadata or {}).get("order")
    if not rec or order is None:
        return None
    try:
        v = json.loads(resp.text.strip().strip("`").replace("json\n", "", 1))
    except json.JSONDecodeError:
        start, end = resp.text.find("{"), resp.text.rfind("}")
        if not (0 <= start < end):
            return None
        try:
            v = json.loads(resp.text[start:end + 1])
        except json.JSONDecodeError:
            return None
    if v.get("both_bad"):
        return None
    w, l = (v.get("winner") or "").strip().upper()[:1], (v.get("loser") or "").strip().upper()[:1]
    if w not in LABELS or l not in LABELS or w == l:
        return None
    cands = _candidates(rec)
    try:
        wi, li = order[LABELS.index(w)], order[LABELS.index(l)]
    except (ValueError, IndexError):
        return None
    if not (0 <= wi < len(cands) and 0 <= li < len(cands)):
        return None
    conf = float(v.get("confidence", 0) or 0)
    if conf < min_confidence:
        return None
    out = {k: rec[k] for k in ("id", "lang", "tags", "tasks", "tools", "prompt_messages", "prompt_text",
                               "instruction", "input", "context", "domain", "subtopic") if k in rec}
    out.update({
        "response_1": cands[wi],          # chosen
        "response_2": cands[li],          # rejected
        "judge_model": resp.model,
        "judge_confidence": conf,
        "deciding_criterion": v.get("deciding_criterion"),
        "judge_why": v.get("why", ""),
        "generator_labels": rec.get("ranking"),
        # Did the independent judge agree with the generator's own intention? A low agreement rate means the
        # generator is not actually writing the bad candidate it thinks it is.
        "agrees_with_generator": (rec.get("ranking") or [None])[0] == "best" and wi == 0,
    })
    return out


def run(provider_name: str, *, model: Optional[str], langs: Optional[list[str]], limit: int,
        batch: bool, dry_run: bool, concurrency: int, push: bool, repo_id: str,
        min_confidence: float, rewrite: bool) -> int:
    seed = Seed.load("rl")
    already = done_ids(JUDGED_KIND, langs)
    reqs = []
    for rec in _iter_rl_records(langs):
        if rec["id"] in already:
            continue
        r = build_judge_request(rec, seed)
        if r:
            reqs.append(r)
        if limit and len(reqs) >= limit:
            break
    print(f"[judge] {len(reqs):,} RL samples to judge ({len(already):,} already judged)")
    if not reqs:
        return 0
    if dry_run:
        for r in reqs[:2]:
            print("\n" + "=" * 100)
            for m in r.messages:
                print(f"\n--- {m['role']} ---\n{m['content'][:2000]}")
        print(f"\n(dry run: {len(reqs):,} judge requests would be sent, none were)")
        return 0

    provider = get_provider(provider_name, model, max_concurrency=concurrency)
    gen_models = {r.metadata["record"].get("model") for r in reqs[:50]}
    if provider.model in gen_models:
        print(f"  WARNING: judging with {provider.model}, the same model that generated these. "
              f"Self-preference bias will inflate agreement. Pass --model <other> instead.")

    if batch:
        if not provider.supports_batch:
            raise SystemExit(f"{provider_name} has no batch API -- drop --batch")
        wd = OUT_ROOT / JUDGED_KIND / "_batches" / time.strftime("%Y%m%d_%H%M%S")
        bid = provider.submit_batch(reqs, wd)
        print(f"\njudge batch {bid} submitted. Fetch with:\n"
              f"    python judge_gen.py --provider {provider_name} --fetch {bid} --push")
        return 0

    writer = ShardWriter(JUDGED_KIND)
    kept = dropped = agreed = 0
    for resp in provider.complete_many(reqs):
        rec = apply_verdict(resp, min_confidence=min_confidence) if resp.ok else None
        if rec is None:
            dropped += 1
            continue
        writer.write(rec["lang"], rec)
        kept += 1
        agreed += bool(rec["agrees_with_generator"])
    paths = writer.close()
    print(f"\n[judge] kept {kept:,}  dropped {dropped:,}")
    if kept:
        print(f"  judge agreed with the generator's own 'best' label {agreed / kept:.1%} of the time "
              f"(near 100% means the judge is rubber-stamping; near 33% means the generator's labels are noise)")
    print(" ", provider.usage_line())
    if push and paths:
        from hub import push_shards
        push_shards(JUDGED_KIND, paths, repo_id=repo_id)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="openrouter", choices=["together", "openrouter"])
    ap.add_argument("--model", default=None, help="judge model; use a DIFFERENT one from the generator")
    ap.add_argument("--langs", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--fetch", default=None, metavar="BATCH_ID")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo-id", default="BeardedMonster/data-gen")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop verdicts the judge is less sure of than this")
    ap.add_argument("--rewrite", action="store_true",
                    help="let the judge rewrite the winner (distillation; see the module docstring)")
    a = ap.parse_args()
    langs = [s.strip() for s in a.langs.split(",")] if a.langs else None
    if a.fetch:
        provider = get_provider(a.provider, a.model)
        wd = sorted((OUT_ROOT / JUDGED_KIND / "_batches").glob("*"), key=lambda p: p.stat().st_mtime)
        responses = provider.fetch_batch(a.fetch, (wd[-1] if wd else OUT_ROOT) / "batch_output.jsonl")
        writer = ShardWriter(JUDGED_KIND)
        kept = 0
        for resp in responses:
            rec = apply_verdict(resp, min_confidence=a.min_confidence) if resp.ok else None
            if rec:
                writer.write(rec["lang"], rec)
                kept += 1
        paths = writer.close()
        print(f"kept {kept:,} of {len(responses):,}")
        if a.push and paths:
            from hub import push_shards
            push_shards(JUDGED_KIND, paths, repo_id=a.repo_id)
        return 0
    return run(a.provider, model=a.model, langs=langs, limit=a.limit, batch=a.batch, dry_run=a.dry_run,
               concurrency=a.concurrency, push=a.push, repo_id=a.repo_id,
               min_confidence=a.min_confidence, rewrite=a.rewrite)


if __name__ == "__main__":
    raise SystemExit(main())
