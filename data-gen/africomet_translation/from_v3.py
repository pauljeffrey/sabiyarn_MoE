"""Turn the published v3 dataset back into source-window shards (the stage-A format), so the old languages do not have to be re-extracted.

v3 rows are (source window, random target): every distinct (source_language, input) is one window. Windows whose source licence is
non-commercial are dropped (v3 predates that rule), Ibibio / Urhobo are dropped (not in AfriCOMET), and exact duplicates are removed.

    python -m africomet_translation.from_v3 --out OUT
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = "BeardedMonster/africomet-translation"
KEEP = {"pcm", "yor", "hau", "ibo", "fon", "ewe", "twi", "efi", "eng"}


def is_non_commercial(lic: str) -> bool:
    u = (lic or "").upper().replace("_", "-")
    return "-NC" in u or "NONCOMMERCIAL" in u.replace("-", "").replace(" ", "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from africomet_translation.build import hf_token
    from africomet_translation.extract import ShardWriter
    from africomet_translation.text_utils import sent_hash, split_sentences
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    token = hf_token()
    fs = HfFileSystem(token=token)
    files = sorted(f.path for f in HfApi(token=token).list_repo_tree(REPO, repo_type="dataset", recursive=True) if f.path.endswith(".parquet"))
    writers: dict[str, ShardWriter] = {}
    seen: set[int] = set()
    stats = {"in": 0, "nc": 0, "dup": 0, "dropped_lang": 0, "kept": {}}
    cols = ["source_language", "input", "n_sentences", "src_tokens", "source_dataset", "source_license", "source_kind"]
    for p in files:
        with fs.open(f"datasets/{REPO}/{p}", "rb") as fh:
            for b in pq.ParquetFile(fh).iter_batches(batch_size=50_000, columns=cols):
                d = b.to_pydict()
                for i in range(len(d["input"])):
                    stats["in"] += 1
                    lang = d["source_language"][i]
                    if lang not in KEEP:
                        stats["dropped_lang"] += 1
                        continue
                    if is_non_commercial(d["source_license"][i]):
                        stats["nc"] += 1
                        continue
                    text = d["input"][i]
                    h = int.from_bytes(hashlib.blake2b(f"{lang}\x1f{text}".encode(), digest_size=8).digest(), "little")
                    if h in seen:
                        stats["dup"] += 1
                        continue
                    seen.add(h)
                    sents = split_sentences(text) or [text]
                    w = writers.setdefault(lang, ShardWriter(Path(a.out) / "windows" / lang, "v3-reuse"))
                    w.add(text=text, n_sentences=int(d["n_sentences"][i] or len(sents)), tokens=int(d["src_tokens"][i] or 0),
                           hashes=[sent_hash(s) for s in sents], source=d["source_dataset"][i], license=d["source_license"][i],
                           kind=d["source_kind"][i], ref="", ref_lang="")
                    stats["kept"][lang] = stats["kept"].get(lang, 0) + 1
        print(p, file=sys.stderr, flush=True)
    for w in writers.values():
        w.close()
    print(stats, file=sys.stderr)


if __name__ == "__main__":
    main()
