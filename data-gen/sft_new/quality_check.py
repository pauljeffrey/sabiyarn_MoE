"""Measure the translation quality of a candidate source with AfriCOMET-QE instead of judging it by eye.

    HF_HOME=SCRATCH python -m sft_new.quality_check --n 100 [--only NAME ...] --comet-python ../.venv-comet/bin/python

Each case takes `n` aligned (source, translation) pairs from a dataset, scores them with masakhane/africomet-qe-stl-1.1 (reference-free, run in the
separate Python 3.11 environment through rl/comet_worker.py) and reports mean, median, P10, P90 and the share below 0.46. That share is the
repo's calibrated failure line: a fluent-but-wrong sentence scores about 0.44-0.46 and half-omitted output about 0.42; human reference
translations score P10/P50/P90 = .51/.65/.76 (English-centric pairs) and .46/.59/.71 (interlanguage pairs). A CONTROL case (a dataset that is
known to be good) is scored the same way, so a source is judged against it and not against an absolute number. The sentence-level controls are
also repeated at paragraph length, because QE scores drift down on long segments (the aya sets are paragraphs).

Only the four languages AfriCOMET-QE was validated on here (English, Hausa, Igbo, Yoruba) can be scored. A dataset that is not a translation
(question answering, summarisation) cannot be: for those the script only counts machine-translation artefacts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from sft_new import hubio  # noqa: E402
from sft_new.textutil import clean  # noqa: E402

LOW = 0.46


def scorer(python: str):
    proc = subprocess.Popen([python, "-m", "rl.comet_worker", "--model", "masakhane/africomet-qe-stl-1.1"], cwd=str(HERE.parent), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    assert json.loads(proc.stdout.readline()).get("ready")

    def score(pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(pairs), 16):
            chunk = pairs[i:i + 16]
            proc.stdin.write(json.dumps({"data": [{"src": a[:1800], "mt": b[:1800]} for a, b in chunk], "batch_size": 8}) + "\n")
            proc.stdin.flush()
            r = json.loads(proc.stdout.readline())
            if "error" in r:
                raise RuntimeError(r["error"])
            out += r["scores"]
        return out
    return score, proc


# name -> (repo, source column, translation column, label)
CASES = {
    "control: english_to_igbo eng>ibo": ("Aletheia-ng/english_to_igbo", "English", "Igbo"),
    "control: WebScraped_Igbo_To_Eng eng>ibo": ("Aletheia-ng/WebScraped_Igbo_To_Eng", "english", "igbo"),
    "control, paragraph-length only: english_to_igbo eng>ibo (>=350 chars)": ("Aletheia-ng/english_to_igbo", "English", "Igbo", 350),
    "aya-translate eng>yor": ("Aletheia-ng/aya-translate", "english", "yoruba"),
    "aya-translate eng>ibo": ("Aletheia-ng/aya-translate", "english", "ibo"),
    "aya-translate eng>hau": ("Aletheia-ng/aya-translate", "english", "hausa"),
    "aya_interlanguage yor>ibo": ("Aletheia-ng/aya_interlanguage_translate", "yoruba", "ibo"),
    "aya_interlanguage hau>yor": ("Aletheia-ng/aya_interlanguage_translate", "hausa", "yoruba"),
    "aya_interlanguage eng>hau": ("Aletheia-ng/aya_interlanguage_translate", "english", "hausa"),
    "aya_interlanguage eng>yor": ("Aletheia-ng/aya_interlanguage_translate", "english", "yoruba"),
    "aya_interlanguage ibo>yor": ("Aletheia-ng/aya_interlanguage_translate", "ibo", "yoruba"),
    "aya_interlanguage hau>ibo": ("Aletheia-ng/aya_interlanguage_translate", "hausa", "ibo"),
    "aya_interlanguage yor>hau": ("Aletheia-ng/aya_interlanguage_translate", "yoruba", "hausa"),
}


def sample_pairs(repo: str, a: str, b: str, n: int, seed: int = 0, pool: int = 4000, min_len: int = 40):
    rows = [(clean(r.get(a)), clean(r.get(b))) for r in itertools.islice(hubio.iter_repo(repo, columns=[a, b]), pool)]
    rows = [(x, y) for x, y in rows if min_len <= len(x) <= 1500 and min_len <= len(y) <= 1500]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def summarize(xs: list[float]) -> dict:
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "mean": round(statistics.mean(xs), 3), "median": round(q(0.5), 3), "p10": round(q(0.1), 3), "p90": round(q(0.9), 3),
            f"share<{LOW}": round(sum(1 for x in xs if x < LOW) / len(xs), 3)}


def artefacts(n: int = 3000) -> dict:
    """aya_collection answers are machine-translated Flan numbered lists: count the tell-tale pieces."""
    rows = list(itertools.islice(hubio.iter_repo("Aletheia-ng/aya_collection_language_split", columns=["targets", "task_type"]), n))
    qa = [clean(r["targets"]) for r in rows if r["task_type"] == "question-answering"]
    marks = ("Na farko.", "Na biyu.", "Ta yaya za mu iya yin hakan?", "Abin da ya faru.", "A'a ba haka ba")
    return {"qa_rows": len(qa), "share_with_translated_list_markers": round(sum(1 for t in qa if any(m in t for m in marks)) / max(1, len(qa)), 3),
            "median_words": statistics.median(len(t.split()) for t in qa) if qa else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--comet-python", default=str(HERE.parent / ".venv-comet" / "bin" / "python"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    score, proc = scorer(a.comet_python)
    res = {}
    try:
        for name, spec in CASES.items():
            repo, ca, cb = spec[:3]
            if a.only and not any(o in name for o in a.only):
                continue
            pairs = sample_pairs(repo, ca, cb, a.n, min_len=spec[3] if len(spec) > 3 else 40)
            res[name] = summarize(score(pairs))
            print(name, res[name], flush=True)
        if not a.only or any("collection" in o for o in a.only):
            res["aya_collection artefacts"] = artefacts()
            print("aya_collection artefacts", res["aya_collection artefacts"], flush=True)
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
