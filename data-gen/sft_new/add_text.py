"""Restructure every assistant turn (label tokens and <tag> into <task_plan>, see sft_new/grammar.py) and (re)write the derived
columns `text` (chat-template string) and `output_structure` (the expected prefix up to <response>). Rerunnable: the grammar
step is idempotent and `--force` recomputes files that already carry `text`.

Original docstring: add the `text` column (the chat-template string that gets tokenized) to every parquet file of the SFT repo, in place.

    nice -n 15 python -m sft_new.add_text [--only NAME ...]

`text` = canonical SabiYarn chat template applied to `messages` (<s><|user|>...</s><|assistant|>...</s>). A `language` column is
added to the single-language task sets (it equals source_lang). Files that already have `text` are skipped, so the run resumes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]
REPO = "BeardedMonster/sft-new"
MONO = ("structured-", "artificial-", "instruct-")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--force", action="store_true", help="recompute files that already have a text column")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(HERE))
    from africomet_translation.build import hf_token
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from sft_new.grammar import normalize_messages_json
    from sft_new.render import render_json
    token = hf_token()
    api = HfApi(token=token)
    files = sorted(f.path for f in api.list_repo_tree(a.repo, repo_type="dataset", recursive=True) if f.path.endswith(".parquet"))
    by_dir: dict[str, list[str]] = {}
    for p in files:
        by_dir.setdefault(p.split("/")[1], []).append(p)
    for name, ps in by_dir.items():
        if a.only and name not in a.only:
            continue
        tmp = Path(tempfile.mkdtemp())
        ops, done = [], 0
        for p in ps:
            local = hf_hub_download(a.repo, p, repo_type="dataset", token=token, cache_dir=str(tmp / "c"))
            tb = pq.read_table(local)
            if "text" in tb.schema.names and not a.force:
                shutil.rmtree(tmp / "c", ignore_errors=True)
                continue
            tasks = tb["task"].to_pylist() if "task" in tb.schema.names else [None] * tb.num_rows
            norm = [normalize_messages_json(m, t) for m, t in zip(tb["messages"].to_pylist(), tasks)]
            for col, vals in (("messages", [n[0] for n in norm]), ("text", [render_json(n[0]) for n in norm]),
                              ("output_structure", [n[1] for n in norm])):
                arr = pa.array(vals, pa.string())
                if col in tb.schema.names:
                    tb = tb.set_column(tb.schema.get_field_index(col), col, arr)
                else:
                    tb = tb.append_column(col, arr)
            if name.startswith(MONO) and "language" not in tb.schema.names and "source_lang" in tb.schema.names:
                tb = tb.append_column("language", tb["source_lang"])
            outp = tmp / "o" / Path(p).name
            outp.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(tb, outp, compression="zstd")
            ops.append(CommitOperationAdd(p, str(outp)))
            done += 1
            shutil.rmtree(tmp / "c", ignore_errors=True)
            if len(ops) >= 6:                                                        # keep local disk and commits small
                api.create_commit(a.repo, ops, repo_type="dataset", commit_message=f"{name}: add `text` column")
                shutil.rmtree(tmp / "o", ignore_errors=True)
                ops = []
        if ops:
            api.create_commit(a.repo, ops, repo_type="dataset", commit_message=f"{name}: add `text` column")
        shutil.rmtree(tmp, ignore_errors=True)
        print(name, "files updated:", done, "of", len(ps), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
