"""Prove a language works BEFORE spending a rental on it.

The problem this exists for is specific and was measured, not imagined. Asked for pretraining prose in the six
low-resource languages, gemma-4-31b-it produced usable documents at wildly different rates:

    ewe 83%   ful 50%   fuv 42%   fon 17%   urh 17%   efi 8%       (n=12 each)

and the failures are LOOPS -- one clause repeated for hundreds of words -- which the generator scored up to
0.9 on its own `confidence` field. At 8%, efi's 30,000-document target needs ~375,000 requests. Discovering
that forty hours into a paid box, with nobody watching the log, is the failure mode this module prevents.

So: generate a handful per language, measure, and act per language rather than per run. A language that fails
is DROPPED and named; the rest proceed. Aborting everything because Efik is hard would be the wrong call, and
so would silently burning the budget on it.

The report is written next to the shards so a later run can read what was decided and why.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

# Per-language clean rate below which a language is not worth generating. Set from the measurement above: it
# keeps ewe (83%) and ful (50%), and excludes fon/urh/efi (8-17%) whose cost per kept sample is 6-12x the
# others. fuv at 42% sits just above it deliberately -- marginal, and the operator can see the number.
MIN_CLEAN_RATE = 0.35
# Samples per language. Small enough to be cheap, large enough that 0/n and n/n mean something: at n=8 a
# language at the 35% floor has a ~3% chance of scoring 0, so a zero is a real signal rather than luck.
DEFAULT_PER_LANG = 8


def clean_rate_report(results: dict[str, tuple[int, int]]) -> str:
    lines = []
    for lang, (ok, total) in sorted(results.items(), key=lambda kv: -(kv[1][0] / max(kv[1][1], 1))):
        rate = ok / max(total, 1)
        verdict = "keep" if rate >= MIN_CLEAN_RATE else "DROP"
        lines.append(f"    {lang:5s} {ok:2d}/{total:<2d} {rate:5.0%}  {verdict}")
    return "\n".join(lines)


def run_preflight(kind: str, langs: list[str], *, generate: Callable[[list[str], int], dict[str, tuple[int, int]]],
                  per_lang: int = DEFAULT_PER_LANG, out_dir: Optional[Path] = None,
                  min_rate: float = MIN_CLEAN_RATE) -> list[str]:
    """Returns the languages worth generating. `generate(langs, per_lang)` -> {lang: (clean, attempted)}.

    Raises SystemExit if nothing passes, because continuing would spend the whole rental on output that the
    post-processor is going to reject anyway.
    """
    print(f"\n=== preflight: {per_lang} samples per language across {len(langs)} language(s)", flush=True)
    t0 = time.time()
    results = generate(langs, per_lang)
    print(f"  measured in {(time.time() - t0) / 60:.1f}m")
    print(clean_rate_report(results))

    keep = [l for l in langs if results.get(l, (0, 0))[0] / max(results.get(l, (0, 1))[1], 1) >= min_rate]
    dropped = [l for l in langs if l not in keep]
    if dropped:
        print(f"  DROPPING {', '.join(dropped)}: below {min_rate:.0%} clean. Generating these would spend most "
              f"of the rental on documents the post-processor rejects.\n"
              f"  Override with --min-clean-rate 0 if you want them anyway, or generate them with a different "
              f"model.", flush=True)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "preflight.json").write_text(json.dumps(
            {"kind": kind, "per_lang": per_lang, "min_rate": min_rate,
             "results": {k: list(v) for k, v in results.items()},
             "kept": keep, "dropped": dropped, "at": time.strftime("%Y-%m-%dT%H:%M:%S")},
            indent=2), encoding="utf-8")
    if not keep:
        raise SystemExit(
            f"preflight: NO language reached {min_rate:.0%} clean for {kind}. Nothing would survive "
            f"post-processing, so the run is not worth starting.\n"
            f"Check the model name, then try --context 16384, a different model, or --min-clean-rate to "
            f"proceed anyway.")
    print(f"  proceeding with: {', '.join(keep)}", flush=True)
    return keep
