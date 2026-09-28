#!/usr/bin/env python3
"""Measure tokens per word per language with the TARGET tokenizer, from text this pipeline actually produced.

    python scripts/measure_tokens_per_word.py
    python scripts/measure_tokens_per_word.py --extra /tmp/some/other/out

Every token budget in budgets.py and assemble.py is converted to words through this table, so an error here
propagates into every document length and every response cap. The first version was calibrated against a
generic tokenizer and was wrong by up to 2x -- SabiYarn-32k is trained ON these languages and is far more
efficient on them than a general-purpose vocabulary. The symptom was documents arriving at half their
requested size.

Re-run this whenever there is materially more text, especially for efi, urh, ful and fuv, whose samples are
still under 1,500 words each.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

MIN_WORDS_PER_SAMPLE = 25          # shorter texts are labels and greetings; they skew the ratio
MIN_WORDS_PER_LANG = 400           # below this, report but do not trust


def _tokenizer():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    from schemas.seed import Seed
    name = Seed.load("sft").target_model.get("tokenizer", "BeardedMonster/SabiYarn-32k")
    return name, Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", nargs="*", default=[], help="additional out/ roots to include")
    a = ap.parse_args()
    name, tok = _tokenizer()
    print(f"tokenizer: {name}")

    roots = [str(HERE / "data" / "out")] + a.extra
    per: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    files = [f for root in roots for f in glob.glob(f"{root}/**/*.jsonl", recursive=True)]
    for f in files:
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            lang = r.get("lang")
            if not lang:
                continue
            # Only text whose language is actually `lang`: an SFT response on an english_english turn is
            # English however the record is filed, and counting it would pull the ratio towards English.
            if "text" in r:
                texts = [r["text"]]
            elif r.get("io_direction") in ("native_native", "english_to_native", "crosslingual"):
                texts = [r.get("response") or ""]
            else:
                texts = []
            for t in texts:
                if len(t.split()) >= MIN_WORDS_PER_SAMPLE:
                    per[lang][0] += len(t.split())
                    per[lang][1] += len(tok.encode(t).ids)

    print(f"scanned {len(files):,} shard(s)\n")
    print(f"{'lang':5s} {'words':>9s} {'tokens/word':>12s}   note")
    table = {}
    for lang in sorted(per, key=lambda l: per[l][0], reverse=True):
        w, t = per[lang]
        ratio = t / max(w, 1)
        thin = "  THIN -- do not trust" if w < MIN_WORDS_PER_LANG else ""
        if w >= MIN_WORDS_PER_LANG:
            table[lang] = round(ratio, 2)
        print(f"{lang:5s} {w:>9,} {ratio:>12.2f}{thin}")
    print("\npaste into assemble._TOKENS_PER_WORD:")
    print("   ", json.dumps(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
