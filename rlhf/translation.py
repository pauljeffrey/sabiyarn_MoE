"""Translation task glue: turn the tds-sft translation dataset (instruction, input, response) into chat records
for rl/ (SFT warm-start, then AfriCOMET RL), and translate with a trained checkpoint.

A record is  {prompt_messages, source, reference, response, ...}:
  user message = "<instruction>\n\n<source text>"      (system prompt: the same one eval_suite's chat style uses)
so a model trained here is prompted exactly like `python -m eval_suite.run --style chat` prompts it.
"""

from __future__ import annotations

from typing import Iterable, Optional

from rl.config import RLConfig
from rl.data import load_records as _load_generic

SYSTEM = "You are a helpful multilingual assistant for West African language speakers."


def translation_messages(instruction: str, source: str) -> list[dict]:
    user = f"{(instruction or '').strip()}\n\n{(source or '').strip()}".strip()
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def to_record(row: dict) -> Optional[dict]:
    src, ref = (row.get("input") or "").strip(), (row.get("response") or "").strip()
    if not src or not ref:
        return None
    return {"prompt_messages": translation_messages(row.get("instruction", ""), src), "source": src,
            "reference": ref, "response": ref, "language": row.get("language")}


def load_translation_records(cfg: RLConfig) -> list[dict]:
    """Hub dataset (cfg.dataset_id) or jsonl (cfg.data_path) with instruction/input/response columns."""
    rows: Iterable[dict] = _load_generic(cfg)
    return [r for r in (to_record(x) for x in rows) if r is not None]


def translate(instruction: str, source: str, model_path: str, cfg: RLConfig, max_new_tokens: int = 128) -> str:
    import torch

    from rl.common import load_model, load_tokenizer
    from rl.data import render_prompt

    tok, model = load_tokenizer(cfg), load_model(cfg, model_path, trainable=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ids = torch.tensor([render_prompt(tok, translation_messages(instruction, source))], device=device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                             do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                             eos_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.size(1):], skip_special_tokens=True).strip()
