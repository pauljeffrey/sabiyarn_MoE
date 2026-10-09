"""Grow the token embedding and the learned position embedding of a trained SabiYarn checkpoint.

SFT needs two things the pretrained checkpoint does not have:

* more token rows: the tokenizer grew to 52117 ids (language tags and <POS> sit at 52050-52116) but the checkpoint
  has 52050 rows, so every id >= 52050 is out of range. Rounded up to a multiple of 64 (tensor-core friendly): 52160.
* more positions: SFT runs at 64k context. wpe is a learned absolute table, so it needs 65536 rows; the 32768
  already learned are kept untouched and the new rows start from `init_new_positions` (the regressor in
  `position_regressor.py` can overwrite them with a better guess before training).

Everything here edits a live model in place and keeps `config`, the tied `lm_head` and the dtype consistent, so it
is safe to call right after `from_pretrained` and before the optimizer is built.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

SFT_VOCAB_MIN = 52135        # highest tokenizer id is "query" = 52134 (<POS> 52116, <tools> 52117, </tools> 52118, 16 quoted JSON keys 52119-52134; scripts/add_tool_tokens.py); 52160 rows hold all of them
# The tokenizer as it was before the tool-calling vocabulary was added; the 'pieces' of a new token are read from it (see build_piece_map).
PIECE_INIT_REVISION = "0c711cb2f7af669b2782c90c1d0d4d097df21422"
SFT_BLOCK_SIZE = 65536
ROW_MULTIPLE = 64


def round_up(n: int, multiple: int = ROW_MULTIPLE) -> int:
    return ((int(n) + multiple - 1) // multiple) * multiple


def sft_vocab_size(highest_token_id: int = SFT_VOCAB_MIN - 1, multiple: int = ROW_MULTIPLE) -> int:
    """Smallest multiple of `multiple` that holds every tokenizer id (52134 -> 52160)."""
    return round_up(highest_token_id + 1, multiple)


def build_piece_map(old_tok, new_tok, first_new_id: int, last_id: int) -> dict[int, list[int]]:
    """For every new token id (first_new_id..last_id): the ids of the pieces the OLD tokenizer splits its text into, when they all exist in the
    checkpoint (< first_new_id). `"arguments"` -> [", ar, g, uments, "]. Measured on the real tokenizers: this covers the 16 quoted JSON keys and
    <tools>/</tools>; the language tags and <POS> are NOT split (a fast tokenizer keeps an added special token whole), so they have no entry and
    keep the mean + noise start, which is the right default for a token that carries no lexical content."""
    out: dict[int, list[int]] = {}
    for i in range(first_new_id, last_id + 1):
        text = new_tok.convert_ids_to_tokens(i)
        if not text:
            continue
        ids = old_tok(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        if ids and all(j < first_new_id for j in ids):
            out[i] = list(ids)
    return out


def piece_map_from_hub(tokenizer_name: str, first_new_id: int, old_revision: str = PIECE_INIT_REVISION) -> dict[int, list[int]]:
    from transformers import AutoTokenizer
    new_tok = AutoTokenizer.from_pretrained(tokenizer_name)
    old_tok = AutoTokenizer.from_pretrained(tokenizer_name, revision=old_revision)
    return build_piece_map(old_tok, new_tok, first_new_id, len(new_tok) - 1)


@torch.no_grad()
def resize_token_embeddings(model, new_vocab_size: int, seed: int = 0, piece_map: Optional[dict] = None) -> int:
    """Grow wte (and the tied lm_head) to `new_vocab_size` rows. Returns the number of rows added (0 = untouched).

    Existing rows are copied. A new token listed in `piece_map` starts as the MEAN OF THE ROWS OF ITS PIECES (so "arguments" begins near
    ar+g+uments and the model can use it from the first step); any other new row is the mean of the old rows plus small noise (the standard "mean
    initialization": a new token starts as an average token, not as an outlier that wrecks the first logits).
    """
    wte: nn.Embedding = model.transformer.wte
    old, dim = wte.weight.shape
    if new_vocab_size <= old:
        return 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = wte.weight.data
    mean = w.float().mean(0, keepdim=True)
    std = w.float().std(0, keepdim=True).clamp_min(1e-6)
    extra = mean + 0.1 * std * torch.randn(new_vocab_size - old, dim, generator=g)
    for i, pieces in (piece_map or {}).items():
        if old <= i < new_vocab_size and pieces and all(0 <= j < old for j in pieces):
            extra[i - old] = w[torch.tensor(pieces, device=w.device)].float().mean(0).cpu()
    extra = extra.to(device=w.device, dtype=w.dtype)
    new_w = torch.cat([w, extra], dim=0)
    new = nn.Embedding(new_vocab_size, dim, device=w.device, dtype=w.dtype)
    new.weight.data.copy_(new_w)
    model.transformer.wte = new

    tied = bool(getattr(model.config, "tie_word_embeddings", True))
    head = nn.Linear(dim, new_vocab_size, bias=False, device=w.device, dtype=w.dtype)
    if tied:
        head.weight = new.weight
    else:
        old_head = model.lm_head.weight.data
        hm = old_head.float().mean(0, keepdim=True)
        hs = old_head.float().std(0, keepdim=True).clamp_min(1e-6)
        hx = (hm + 0.1 * hs * torch.randn(new_vocab_size - old, dim, generator=g)).to(device=w.device, dtype=w.dtype)
        head.weight.data.copy_(torch.cat([old_head, hx], dim=0))
    model.lm_head = head
    model.config.vocab_size = new_vocab_size
    return new_vocab_size - old


@torch.no_grad()
def init_new_positions(old: torch.Tensor, n_new: int, mode: str = "mean_noise", seed: int = 0) -> torch.Tensor:
    """Starting values for `n_new` positions past the learned ones.

    mean_noise  the mean learned position + a little noise (neutral, keeps activations in range)
    tile        repeat the learned table (position p+L looks like p; the model sees a repeat of its own training range)
    last        copy the final learned rows (continuity at the seam; no structure beyond it)
    """
    old_n = old.shape[0]
    f = old.float()
    if mode == "tile":
        idx = torch.arange(n_new) % old_n
        return f[idx].to(old.dtype)
    if mode == "last":
        k = min(old_n, 256)
        idx = old_n - k + (torch.arange(n_new) % k)
        return f[idx].to(old.dtype)
    g = torch.Generator(device="cpu").manual_seed(seed)
    std = f.std(0, keepdim=True).clamp_min(1e-6)
    return (f.mean(0, keepdim=True) + 0.1 * std * torch.randn(n_new, old.shape[1], generator=g)).to(old.dtype)


@torch.no_grad()
def resize_position_embeddings(model, new_block_size: int, init: str = "mean_noise", seed: int = 0) -> int:
    """Grow wpe to `new_block_size` rows, keeping every learned row. Returns the number of rows added."""
    wpe: nn.Embedding = model.transformer.wpe
    old_n, dim = wpe.weight.shape
    if new_block_size <= old_n:
        return 0
    w = wpe.weight.data
    extra = init_new_positions(w.cpu(), new_block_size - old_n, mode=init, seed=seed).to(device=w.device, dtype=w.dtype)
    new = nn.Embedding(new_block_size, dim, device=w.device, dtype=w.dtype)
    new.weight.data.copy_(torch.cat([w, extra], dim=0))
    model.transformer.wpe = new
    model.config.block_size = new_block_size
    model.config.max_position_embeddings = new_block_size
    for block in model.transformer.h:               # a causal-mask buffer, when a block keeps one, is sized by block_size
        b = getattr(block.attn, "bias", None)
        if isinstance(b, torch.Tensor) and b.dim() == 4 and b.shape[-1] == old_n:
            block.attn.bias = torch.tril(torch.ones(new_block_size, new_block_size, dtype=b.dtype, device=b.device)).view(
                1, 1, new_block_size, new_block_size)
    return new_block_size - old_n


def resize_for_sft(model, vocab_size: Optional[int] = None, block_size: int = SFT_BLOCK_SIZE,
                   position_init: str = "mean_noise", seed: int = 0, piece_map: Optional[dict] = None) -> dict:
    """The one call the SFT trainer makes: vocab -> multiple of 64 (52160), wpe -> 64k. Idempotent."""
    target_vocab = int(vocab_size) if vocab_size else sft_vocab_size()
    if target_vocab % ROW_MULTIPLE:
        raise ValueError(f"vocab_size {target_vocab} is not a multiple of {ROW_MULTIPLE}")
    before = (model.transformer.wte.weight.shape[0], model.transformer.wpe.weight.shape[0])
    added_tok = resize_token_embeddings(model, target_vocab, seed=seed, piece_map=piece_map)
    added_pos = resize_position_embeddings(model, int(block_size), init=position_init, seed=seed)
    return {"vocab_before": before[0], "vocab_after": model.transformer.wte.weight.shape[0], "tokens_added": added_tok, "rows_from_pieces": len(piece_map or {}) if added_tok else 0,
            "block_before": before[1], "block_after": model.transformer.wpe.weight.shape[0], "positions_added": added_pos}
