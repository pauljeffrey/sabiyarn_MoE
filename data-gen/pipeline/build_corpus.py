#!/usr/bin/env python3
"""Build OpenAI Batch API input files for the pretrain / sft / dpo kinds.

Free and local: writes `<data>/batch_input/<kind>.jsonl` (the API payload)
and `<kind>.manifest.jsonl` (side-car with the sampled attribute tuple for
every custom_id, needed by postprocessing and coverage audits). If the run
exceeds the Batch API limits it is split into `<kind>__part0.jsonl`,
`<kind>__part1.jsonl`, ... exactly like the original `build_batch.py`
(same 50,000-request limit, plus a file-size limit because 50,000 of these
long prompts can exceed the 200 MB file cap).

Usage:
    python pipeline/build_corpus.py --kind pretrain --config configs/pretrain.yaml
    python pipeline/build_corpus.py --kind sft --per-language 20      # small pilot
    python pipeline/build_corpus.py --kind all --dry-run               # estimate only
    python run.py build --kind dpo --config configs/dpo.yaml
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.corpus_config import CorpusConfig, load_config
from config.settings import CORPUS_KINDS, MAX_BATCH_FILE_BYTES, DataPaths
from generators.base import BatchRequestSpec, write_batch_and_manifest
from pipeline.build_batch import MAX_REQUESTS_PER_BATCH_FILE

GENERATOR_MODULES = {"pretrain": "generators.pretrain", "sft": "generators.sft_tasks", "dpo": "generators.dpo"}


def write_split_batches(
    name: str,
    specs: Iterable[BatchRequestSpec],
    batch_dir: Path,
    *,
    max_requests_per_file: int = MAX_REQUESTS_PER_BATCH_FILE,
    max_file_bytes: int = MAX_BATCH_FILE_BYTES,
) -> list[Path]:
    """Stream `specs` into `<name>.jsonl` (+ manifest), or `<name>__partN.jsonl`
    when the request-count or file-size limit would be exceeded.

    Shared by the corpus builders and the judge stage. Old files with the same
    name are removed first so a stale `__partN` can never be submitted by accident.
    """
    for pattern in (f"{name}.jsonl", f"{name}.manifest.jsonl", f"{name}__part*.jsonl"):
        for p in batch_dir.glob(pattern):
            p.unlink()

    chunk: list[BatchRequestSpec] = []
    chunk_bytes = 0
    parts: list[str] = []
    seen: set[str] = set()

    def flush() -> None:
        nonlocal chunk, chunk_bytes
        if not chunk:
            return
        part = f"{name}__part{len(parts)}"
        write_batch_and_manifest(part, chunk, out_dir=batch_dir)
        parts.append(part)
        chunk, chunk_bytes = [], 0

    for spec in specs:
        if spec.custom_id in seen:
            raise ValueError(f"Duplicate custom_id: {spec.custom_id}")
        seen.add(spec.custom_id)
        size = len(json.dumps(spec.to_batch_line(), ensure_ascii=False).encode("utf-8")) + 1
        if chunk and (len(chunk) >= max_requests_per_file or chunk_bytes + size > max_file_bytes):
            flush()
        chunk.append(spec)
        chunk_bytes += size
    flush()

    if len(parts) == 1:  # single file: use the plain name, like build_batch.py does
        (batch_dir / f"{parts[0]}.jsonl").rename(batch_dir / f"{name}.jsonl")
        (batch_dir / f"{parts[0]}.manifest.jsonl").rename(batch_dir / f"{name}.manifest.jsonl")
        parts = [name]

    written = [batch_dir / f"{p}.jsonl" for p in parts]
    for path in written:
        with path.open(encoding="utf-8") as f:
            n = sum(1 for _ in f)
        print(f"[{name}] wrote {n} requests -> {path.name} (+ {path.stem}.manifest.jsonl)")
    if not written:
        print(f"[{name}] no requests to write (all sample counts are 0?)")
    return written


def build_kind(
    cfg: CorpusConfig,
    data_dir: Optional[Path] = None,
    *,
    languages: Optional[list[str]] = None,
    max_requests_per_file: int = MAX_REQUESTS_PER_BATCH_FILE,
    max_file_bytes: int = MAX_BATCH_FILE_BYTES,
) -> list[Path]:
    """Generate every request for `cfg.kind` and write the batch + manifest files."""
    gen = importlib.import_module(GENERATOR_MODULES[cfg.kind])
    return write_split_batches(
        cfg.kind, gen.iter_requests(cfg, languages), DataPaths(data_dir).batch_input,
        max_requests_per_file=max_requests_per_file, max_file_bytes=max_file_bytes,
    )


def audit_manifest(kind: str, data_dir: Optional[Path] = None) -> dict[str, dict]:
    """Coverage of what was *requested*, read back from the manifest side-car.

    Because every manifest line stores the sampled attribute tuple, coverage
    can be audited (and compared with the coverage of the kept records in the
    postprocess report) without any API output.
    """
    from pipeline.postprocess_corpus import load_manifest
    from sampling.sampler import coverage_report

    by_lang: dict[str, list[dict]] = {}
    for entry in load_manifest(kind, DataPaths(data_dir).batch_input).values():
        by_lang.setdefault(entry["language"], []).append(entry["context"]["attributes"])
    out: dict[str, dict] = {}
    for lang, attrs in sorted(by_lang.items()):
        rep = coverage_report(attrs, include_counts=False)["attributes"]
        secondary = [v["entropy_norm"] for k, v in rep.items() if k not in ("pair", "domain")]
        out[lang] = {
            "requests": len(attrs),
            "pairs_used": rep["pair"]["distinct"],
            "pair_max_min": (rep["pair"]["min"], rep["pair"]["max"]),
            "pair_entropy": rep["pair"]["entropy_norm"],
            "domain_entropy": rep["domain"]["entropy_norm"],
            "min_secondary_entropy": min(secondary) if secondary else None,
        }
    return out


def print_audit(kind: str, data_dir: Optional[Path] = None) -> None:
    audit = audit_manifest(kind, data_dir)
    if not audit:
        return
    print(f"[{kind}] requested-coverage audit (from the manifest; normalised entropy 1.0 = perfectly even):")
    print(f"  {'lang':<6}{'requests':>9}{'pairs_used':>11}{'min/max per pair':>18}{'pair_H':>8}{'domain_H':>10}{'min attr_H':>12}")
    for lang, a in audit.items():
        lo, hi = a["pair_max_min"]
        print(f"  {lang:<6}{a['requests']:>9}{a['pairs_used']:>11}{f'{lo}/{hi}':>18}{a['pair_entropy']:>8.3f}{a['domain_entropy']:>10.3f}{a['min_secondary_entropy']:>12.3f}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True, help=f"one of {CORPUS_KINDS} or 'all'")
    parser.add_argument("--config", default=None, help="config yaml (single --kind only); default configs/<kind>.yaml")
    parser.add_argument("--per-language", type=int, default=None, help="override samples per language (pilots/tests)")
    parser.add_argument("--languages", default=None, help="comma-separated subset of language codes")
    parser.add_argument("--model", default=None, help="override the generation model")
    parser.add_argument("--data-dir", default=None, help="output root (default: DATA_GEN_OUTPUT_DIR or ./data)")
    parser.add_argument("--max-requests-per-file", type=int, default=MAX_REQUESTS_PER_BATCH_FILE,
                        help="lower this to fit your Batch API enqueued-token limit (smaller files can be submitted one at a time)")
    parser.add_argument("--dry-run", action="store_true", help="print the volume/cost estimate and write nothing")
    args = parser.parse_args(argv)

    kinds = CORPUS_KINDS if args.kind == "all" else [args.kind]
    if any(k not in CORPUS_KINDS for k in kinds):
        parser.error(f"--kind must be one of {CORPUS_KINDS} or 'all'")
    if args.config and len(kinds) != 1:
        parser.error("--config requires a single --kind")
    langs = [c.strip() for c in args.languages.split(",") if c.strip()] if args.languages else None

    for kind in kinds:
        cfg = load_config(args.config or kind)
        if args.model:
            cfg.model = args.model
        cfg = cfg.with_overrides(per_language=args.per_language, languages=langs)
        if args.dry_run:
            from pipeline.estimate import estimate_kind, format_estimate

            print(format_estimate(estimate_kind(cfg)))
            print()
            continue
        data_dir = Path(args.data_dir) if args.data_dir else None
        build_kind(cfg, data_dir, max_requests_per_file=args.max_requests_per_file)
        print_audit(kind, data_dir)
    if not args.dry_run:
        print("Done. Inspect the batch_input/*.jsonl files (and run `estimate`) before running submit.")


if __name__ == "__main__":
    main()
