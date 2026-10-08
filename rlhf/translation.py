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
    """tds-sft style rows (instruction, input, response) and the AfriCOMET RL dataset (+ source_lang, targ_lang, output_structure).

    With `output_structure` (or source_lang/targ_lang) the record carries the scaffold the policy must emit before the translation,
    `<|input_lang|><src><task_plan><translate></task_plan><|target_lang|><tgt><response>`, and its prompt has NO system turn so it
    renders exactly like the `text` column the SFT data was tokenized from. `reference` is empty for rows without an aligned
    translation (those are scored reference-free). `response` (used by the SFT warm-start) is scaffold + reference.
    """
    src, ref = (row.get("input") or "").strip(), (row.get("response") or "").strip()
    if not src:
        return None
    struct = (row.get("output_structure") or "").strip()
    if not struct and row.get("source_lang") and row.get("targ_lang"):
        from rl.structure import expected_translation_structure
        struct = expected_translation_structure(row["source_lang"], row["targ_lang"], row.get("instruction_language"),
                                                row.get("instruction_position") or "prepend")
    if not struct:
        if not ref:
            return None
        return {"prompt_messages": translation_messages(row.get("instruction", ""), src), "source": src,
                "reference": ref, "response": ref, "language": row.get("language")}
    user = (row.get("prompt") or "").strip() or f"{(row.get('instruction') or '').strip()}\n\n{src}".strip()     # `prompt` keeps the instruction before/after the text as sampled
    return {"prompt_messages": [{"role": "user", "content": user}], "source": src, "reference": ref,
            "response": (struct + ref) if ref else "", "expected_structure": struct,
            "source_lang": row.get("source_lang"), "targ_lang": row.get("targ_lang"), "language": row.get("language")}


def load_translation_records(cfg: RLConfig) -> list[dict]:
    """Hub dataset (cfg.dataset_id) or jsonl (cfg.data_path) with instruction/input/response columns."""
    rows: Iterable[dict] = _load_generic(cfg)
    recs = [r for r in (to_record(x) for x in rows) if r is not None]
    if getattr(cfg, "reward_shaping", "none") in ("calibrated", "adaptive") and getattr(cfg, "reward_calibration_path", None):
        from rl.shaping import ContentShaper

        bad = ContentShaper.from_file(cfg.reward_calibration_path).unusable
        kept = [r for r in recs if f"{r.get('source_lang')}>{r.get('targ_lang')}" not in bad]
        if len(kept) != len(recs):
            import structlog
            structlog.get_logger().info("translation_pairs_dropped", unusable_pairs=sorted(bad), dropped=len(recs) - len(kept))
        recs = kept
    return recs


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
