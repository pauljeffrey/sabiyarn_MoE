"""RL prompts as a memory-mapped bin: build it once, then random-access any prompt by index (torch-free except `collate_left_pad`).

Files (prefix P, all written next to each other):
  P.bin              uint16 token ids of every prompt (the chat-template `text` column, e.g.  <s><|user|>INSTRUCTION+SOURCE</s><|assistant|>),
                     concatenated in globally shuffled order. layout="csr" (default): no padding. layout="padded": fixed rows of L = max prompt
                     length, LEFT-padded with the pad id.
  P.index.npy        csr only: uint64 offsets, row i = bin[index[i]:index[i+1]]            (padded layout: row i = bin[i*L:(i+1)*L])
  P.meta.npy         structured array, one record per row: src (uint8 language id), tgt (uint8 language id)
  P.sources.bin      utf-8 bytes of the SOURCE text of every row (AfriCOMET-QE scores a translation from (source text, translation))
  P.sources.index.npy uint64 offsets into P.sources.bin
  P.manifest.json    tokenizer, layout, pad/eos id, n_rows, eval_start, max/mean/p99 prompt tokens, language table with AfriCOMET codes

Why CSR is the default instead of one fixed width for every row: a fixed width equals the longest prompt (the worst case sets the size of
all 10M rows), while the length of a typical prompt is a third of that, so the file is ~3x larger and every batch is padded to the global
maximum instead of to its own longest prompt. Random access by index costs the same in both layouts (an offset lookup versus a multiply),
and `collate_left_pad` pads each batch to its own longest row. The padded layout is still available (`layout="padded"`) if you want the
fixed stride.

The last `eval_rows` rows of the shuffled order are the held-out eval prompts (`manifest["eval_start"]`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

AFRICOMET_CODE = {"eng": "en", "hau": "ha", "ibo": "ig", "yor": "yo", "pcm": "pcm", "fon": "fon", "ewe": "ee", "twi": "tw", "efi": "efi",
                  "afr": "af", "zul": "zu", "sna": "sn", "nya": "ny", "xho": "xh", "sot": "st", "kin": "rw", "tsn": "tn", "orm": "om",
                  "wol": "wo", "som": "so", "swh": "sw"}
META_DTYPE = np.dtype([("src", "u1"), ("tgt", "u1"), ("ins", "u1"), ("pos", "u1")])     # ins = instruction language id (255 = none), pos = 1 when the instruction follows the text
NO_LANG = 255


def _expected_structure(src: str, tgt: str, ins: Optional[str] = None, position: str = "prepend") -> str:
    from rl.structure import expected_translation_structure
    return expected_translation_structure(src, tgt, ins, position)


# --------------------------------------------------------------------------------------------- build
def build(shards: Iterable[Path], prefix: Path, tokenize: Callable[[list[str]], list[list[int]]], pad_id: int, eos_id: int,
          tokenizer_name: str, layout: str = "csr", eval_rows: int = 5000, buckets: int = 64, seed: int = 0, work: Optional[Path] = None,
          max_tokens_cap: Optional[int] = None) -> dict:
    """Thorough shuffle (random bucket per row, then random order inside every bucket, buckets in random order), tokenize, write the bin.
    `shards` are parquet files with columns text, input, source_lang, targ_lang and (for the multi-language <|input_lang|> field)
    instruction_language, instruction_position; a shard without the last two is treated as "instruction language unknown". Rows longer than `max_tokens_cap` are dropped."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    assert layout in ("csr", "padded")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    work = Path(work or prefix.parent / (prefix.name + "_buckets"))
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cols = ["text", "input", "source_lang", "targ_lang"]
    extra = ["instruction_language", "instruction_position"]
    writers: dict[int, list] = {b: [] for b in range(buckets)}
    n_in = 0
    langs: dict[str, int] = {}
    for p in shards:
        pf = pq.ParquetFile(p)
        have = [c for c in extra if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=200_000, columns=cols + have):
            t = pa.Table.from_batches([batch])
            for c in extra:
                if c not in have:
                    t = t.append_column(c, pa.array([""] * t.num_rows, pa.string()))
            ids = rng.integers(0, buckets, size=t.num_rows)
            for b in np.unique(ids):
                part = t.filter(pa.array(ids == b))
                f = work / f"b{int(b):03d}-{len(writers[int(b)]):05d}.parquet"
                pq.write_table(part, f, compression="zstd")
                writers[int(b)].append(f)
            n_in += t.num_rows
    order = rng.permutation(buckets)
    bin_f = open(f"{prefix}.bin", "wb")
    src_f = open(f"{prefix}.sources.bin", "wb")
    offsets, src_offsets, meta = [0], [0], []
    dropped = 0
    for b in order:
        files = writers[int(b)]
        if not files:
            continue
        t = pa.concat_tables([pq.read_table(f) for f in files])
        perm = rng.permutation(t.num_rows)
        t = t.take(pa.array(perm))
        text, inp = t.column("text").to_pylist(), t.column("input").to_pylist()
        sl, tl = t.column("source_lang").to_pylist(), t.column("targ_lang").to_pylist()
        il, ip = t.column("instruction_language").to_pylist(), t.column("instruction_position").to_pylist()
        for i in range(0, len(text), 5000):
            toks = tokenize(text[i:i + 5000])
            for k, ids in enumerate(toks):
                if max_tokens_cap and len(ids) > max_tokens_cap:
                    dropped += 1
                    continue
                s, g = sl[i + k], tl[i + k]
                for lang in (s, g, il[i + k] or None):
                    if lang:
                        langs.setdefault(lang, len(langs))
                arr = np.asarray(ids, dtype=np.uint16)
                bin_f.write(arr.tobytes())              # the padded layout is rewritten below once the maximum length is known
                offsets.append(offsets[-1] + arr.size)
                sb = inp[i + k].encode("utf-8")
                src_f.write(sb)
                src_offsets.append(src_offsets[-1] + len(sb))
                meta.append((langs[s], langs[g], langs[il[i + k]] if il[i + k] else NO_LANG, 1 if ip[i + k] == "append" else 0))
    bin_f.close()
    src_f.close()
    lens = np.diff(np.asarray(offsets, dtype=np.uint64)).astype(np.int64)
    n = int(lens.size)
    L = int(lens.max())
    if layout == "padded":                                   # rewrite as fixed rows, left-padded
        flat = np.memmap(f"{prefix}.bin", dtype=np.uint16, mode="r")
        out = np.memmap(f"{prefix}.padded.tmp", dtype=np.uint16, mode="w+", shape=(n * L,))
        out[:] = pad_id
        for i in range(n):
            out[i * L + (L - lens[i]): (i + 1) * L] = flat[offsets[i]: offsets[i + 1]]
        out.flush()
        del flat, out
        Path(f"{prefix}.padded.tmp").replace(f"{prefix}.bin")
    else:
        np.save(f"{prefix}.index.npy", np.asarray(offsets, dtype=np.uint64))
    np.save(f"{prefix}.sources.index.npy", np.asarray(src_offsets, dtype=np.uint64))
    np.save(f"{prefix}.meta.npy", np.asarray(meta, dtype=META_DTYPE))
    inv = {v: k for k, v in langs.items()}
    manifest = {"tokenizer": tokenizer_name, "layout": layout, "dtype": "uint16", "pad_id": pad_id, "eos_id": eos_id, "pad_side": "left",
                "n_rows": n, "eval_rows": min(eval_rows, n // 10), "eval_start": n - min(eval_rows, n // 10), "rows_in": n_in,
                "dropped_over_cap": dropped, "max_prompt_tokens": L, "mean_prompt_tokens": float(lens.mean()),
                "p99_prompt_tokens": float(np.percentile(lens, 99)),
                "languages": [{"id": i, "tag": inv[i], "africomet_code": AFRICOMET_CODE.get(inv[i], inv[i])} for i in range(len(inv))]}
    Path(f"{prefix}.manifest.json").write_text(json.dumps(manifest, indent=1))
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    return manifest


# --------------------------------------------------------------------------------------------- read
class RLPromptBin:
    """Random-access view of a built bin. `ds[i]` -> {prompt_ids, source, source_lang, targ_lang, expected_structure, index}."""

    is_bin = True

    def __init__(self, prefix: str | Path):
        self.prefix = str(prefix)
        self.manifest = json.loads(Path(f"{prefix}.manifest.json").read_text())
        self.layout = self.manifest["layout"]
        self.bin = np.memmap(f"{prefix}.bin", dtype=np.uint16, mode="r")
        self.meta = np.load(f"{prefix}.meta.npy", mmap_mode="r")
        self.src_index = np.load(f"{prefix}.sources.index.npy", mmap_mode="r")
        self.src_bin = np.memmap(f"{prefix}.sources.bin", dtype=np.uint8, mode="r")
        self.index = np.load(f"{prefix}.index.npy", mmap_mode="r") if self.layout == "csr" else None
        self.tags = {l["id"]: l["tag"] for l in self.manifest["languages"]}
        self.pad_id = self.manifest["pad_id"]
        self.max_prompt_tokens = self.manifest["max_prompt_tokens"]
        self.n = self.manifest["n_rows"]

    def __len__(self) -> int:
        return self.n

    def prompt_ids(self, i: int) -> list[int]:
        if self.layout == "csr":
            return self.bin[int(self.index[i]): int(self.index[i + 1])].tolist()
        L = self.max_prompt_tokens
        row = self.bin[i * L:(i + 1) * L]
        return row[row != self.pad_id].tolist() if (row == self.pad_id).any() else row.tolist()

    def __getitem__(self, i: int) -> dict:
        i = int(i)
        if not 0 <= i < self.n:
            raise IndexError(i)
        m = self.meta[i]
        s, t = self.tags[int(m["src"])], self.tags[int(m["tgt"])]
        ins = self.tags.get(int(m["ins"])) if int(m["ins"]) != NO_LANG else None
        pos = "append" if int(m["pos"]) else "prepend"
        a, b = int(self.src_index[i]), int(self.src_index[i + 1])
        return {"index": i, "prompt_ids": self.prompt_ids(i), "source": bytes(self.src_bin[a:b]).decode("utf-8"),
                "source_lang": s, "targ_lang": t, "expected_structure": _expected_structure(s, t, ins, pos),
                "instruction_lang": ins, "instruction_position": pos, "africomet_src": AFRICOMET_CODE.get(s, s), "africomet_tgt": AFRICOMET_CODE.get(t, t), "reference": ""}

    def train_view(self) -> "BinSplit":
        return BinSplit(self, 0, self.manifest["eval_start"])

    def eval_view(self, n: Optional[int] = None) -> "BinSplit":
        start = self.manifest["eval_start"]
        return BinSplit(self, start, self.n if n is None else min(self.n, start + n))


class BinSplit(Sequence):
    """A contiguous slice of the bin as a lazy sequence (so a 10M-row set never becomes a Python list)."""

    is_bin = True

    def __init__(self, ds: RLPromptBin, start: int, stop: int):
        self.ds, self.start, self.stop = ds, start, stop

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        return self.ds[self.start + i]


def collate_left_pad(batch: list[dict], pad_id: int, as_torch: bool = True):
    """Pad every prompt of the batch to the batch's longest, on the LEFT. -> dict(input_ids, attention_mask, items)."""
    L = max(len(b["prompt_ids"]) for b in batch)
    ids = np.full((len(batch), L), pad_id, dtype=np.int64)
    mask = np.zeros((len(batch), L), dtype=np.int64)
    for r, b in enumerate(batch):
        p = b["prompt_ids"]
        ids[r, L - len(p):] = p
        mask[r, L - len(p):] = 1
    if as_torch:
        import torch
        return {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask), "items": batch}
    return {"input_ids": ids, "attention_mask": mask, "items": batch}
