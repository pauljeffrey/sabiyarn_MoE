"""Model wrapper: loads a SabiYarn checkpoint and exposes batched generation and continuation scoring.

Design notes
  * Positions are absolute and learned (wpe), so LEFT-padding a batch would shift them. Generation
    therefore batches only prompts of identical token length (no padding at all); scoring uses
    right-padding, which is safe for a causal model.
  * Contexts are prefixed with two eos tokens in `tag` style, mirroring how data/prepare.py lays
    documents into the training bins (`ids + [eos, eos]`), so the first real token sees the same
    context it saw in pretraining.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import torch

from sabiyarn.chat import use_sabiyarn_chat_template

STOP_STRINGS = ("</s>", "<translate>", "<classify>", "<NER>", "<|user|>", "<|assistant|>")


class ModelRunner:
    def __init__(self, model_path: str, tokenizer_name: str = "BeardedMonster/SabiYarn-32k",
                 device: Optional[str] = None, model_code: str = "local", batch_size: int = 32):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, token=token)
        use_sabiyarn_chat_template(self.tok)  # the Hub tokenizer's copy can be stale; --style chat must match SFT data
        if model_code == "local":
            from sabiyarn.model.modeling import GPTJXMoEForCausalLM

            model = GPTJXMoEForCausalLM.from_pretrained(model_path, torch_dtype=dtype, token=token)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype, token=token)
        self.model = model.to(self.device).eval()
        self.eos = self.tok.eos_token_id
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos
        self.max_ctx = min(int(getattr(self.model.config, "block_size", 4096)), 4096)
        self.batch_size = batch_size

    # ------------------------------------------------------------------ encoding
    def encode(self, text: str, style: str) -> list[int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return ([self.eos, self.eos] + ids) if style == "tag" else ids

    def encode_continuation(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, contexts: list[list[int]], max_new_tokens: int = 128, num_beams: int = 1,
                 repetition_penalty: float = 1.0) -> list[str]:
        room = self.max_ctx - max_new_tokens
        contexts = [c[-room:] for c in contexts]
        by_len: dict[int, list[int]] = defaultdict(list)
        for i, c in enumerate(contexts):
            by_len[len(c)].append(i)

        outputs: list[Optional[str]] = [None] * len(contexts)
        for length, idxs in by_len.items():
            for s in range(0, len(idxs), self.batch_size):
                chunk = idxs[s : s + self.batch_size]
                ids = torch.tensor([contexts[i] for i in chunk], device=self.device)
                out = self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                    do_sample=False, num_beams=num_beams, repetition_penalty=repetition_penalty,
                    pad_token_id=self.pad, eos_token_id=self.eos,
                )
                for i, row in zip(chunk, out[:, length:]):
                    outputs[i] = self._clean(self.tok.decode(row, skip_special_tokens=False))
        return outputs  # type: ignore[return-value]

    @staticmethod
    def _clean(text: str) -> str:
        cut = min((text.find(s) for s in STOP_STRINGS if s in text), default=len(text))
        return text[:cut].replace("<pad>", "").strip()

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def score(self, context: list[int], continuations: list[list[int]]) -> list[float]:
        """Sum of log-probabilities of each continuation given the context (one forward pass)."""
        max_cont = max(len(c) for c in continuations)
        context = context[-(self.max_ctx - max_cont):]
        seqs = [context + c for c in continuations]
        width = max(len(s) for s in seqs)
        batch = torch.tensor([s + [self.pad] * (width - len(s)) for s in seqs], device=self.device)
        logprobs = torch.log_softmax(self.model(input_ids=batch).logits.float(), dim=-1)
        scores = []
        for row, cont in enumerate(continuations):
            pos = torch.arange(len(context) - 1, len(context) - 1 + len(cont), device=self.device)
            tgt = torch.tensor(cont, device=self.device)
            scores.append(logprobs[row, pos, tgt].sum().item())
        return scores
