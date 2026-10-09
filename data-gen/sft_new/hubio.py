"""Read rows from a Hub dataset repo without ever keeping it on disk.

Parquet files are opened through the Hub filesystem and read row group by row group (HTTP range requests), so a 250 MB shard costs
only the bytes of the groups actually consumed and nothing is cached. jsonl/csv files are streamed the same way. Nothing here
writes to the Hugging Face cache, so there is nothing to clean up afterwards (`clear_cache` removes the little that a stray
`hf_hub_download` may leave).
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
from pathlib import Path
from typing import Iterator, Optional

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem


def token() -> Optional[str]:
    from africomet_translation.build import hf_token
    return hf_token()


_FS = None


def fs() -> HfFileSystem:
    global _FS
    if _FS is None:
        _FS = HfFileSystem(token=token())
    return _FS


def list_files(repo: str, patterns: tuple[str, ...] = (".parquet", ".jsonl", ".json", ".csv")) -> list[tuple[str, int]]:
    info = HfApi(token=token()).dataset_info(repo, files_metadata=True)
    return sorted((s.rfilename, s.size or 0) for s in info.siblings if s.rfilename.endswith(patterns) and not s.rfilename.startswith("."))


def iter_file(repo: str, path: str, columns: Optional[list[str]] = None, batch_size: int = 512) -> Iterator[dict]:
    """Rows of one file as dicts, streamed."""
    full = f"datasets/{repo}/{path}"
    if path.endswith(".parquet"):
        with fs().open(full, "rb") as fh:
            pf = pq.ParquetFile(fh)
            cols = [c for c in (columns or pf.schema_arrow.names) if c in pf.schema_arrow.names]
            for b in pf.iter_batches(batch_size=batch_size, columns=cols):
                names = b.schema.names
                arrays = [b.column(i).to_pylist() for i in range(len(names))]
                for vals in zip(*arrays):
                    yield {k: (None if isinstance(v, float) and v != v else v) for k, v in zip(names, vals)}      # a float NaN is a null cell
    elif path.endswith(".jsonl"):
        with fs().open(full, "rb") as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
    elif path.endswith(".json"):
        with fs().open(full, "rb") as fh:
            data = json.load(io.TextIOWrapper(fh, encoding="utf-8"))
        yield from (data if isinstance(data, list) else data.get("data", []))
    elif path.endswith(".csv"):
        with fs().open(full, "rb") as fh:
            yield from csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
    else:
        raise ValueError(path)


def iter_repo(repo: str, files: Optional[list[str]] = None, columns: Optional[list[str]] = None, limit: Optional[int] = None,
              match: Optional[str] = None) -> Iterator[dict]:
    """Rows across the repo's files (or the named ones), with `__file` set to the path. `match` is a substring filter on paths."""
    paths = files or [p for p, _ in list_files(repo)]
    if match:
        paths = [p for p in paths if match in p]
    n = 0
    for p in paths:
        for r in iter_file(repo, p, columns):
            r["__file"] = p
            yield r
            n += 1
            if limit and n >= limit:
                return


def clear_cache() -> None:
    """Delete everything a run may have left in the Hugging Face cache (datasets only; model caches such as GlotLID are kept)."""
    if not os.environ.get("HF_HOME"):                 # only ever touches a cache that this run was pointed at explicitly
        return
    root = Path(os.environ["HF_HOME"])
    for d in (root / "hub").glob("datasets--*"):
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(root / "datasets", ignore_errors=True)
    shutil.rmtree(root / "xet", ignore_errors=True)
