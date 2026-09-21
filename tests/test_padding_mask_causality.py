"""A 2D attention_mask is a padding mask and must not turn attention bidirectional.

Regression: `_expand_attn_mask` used to broadcast the (B, keys) mask over all queries with no causal
component, so passing `attention_mask=torch.ones_like(ids)` (what `generate()` and every tokenizer
call does) let each position see later tokens.
"""
import torch

from sabiyarn.model.configuration import GPTJXMoEConfig
from sabiyarn.model.modeling import GPTJXMoEForCausalLM


def _model(use_moe=False, kv=False):
    torch.manual_seed(0)
    cfg = GPTJXMoEConfig(
        block_size=32, vocab_size=64, n_layer=2, n_heads=2, n_embd=8,
        use_moe=use_moe, use_kv_cache=kv, num_experts=[2, 2] if use_moe else None,
        num_experts_per_tok=2, moe_dim=16,
    ) if use_moe else GPTJXMoEConfig(
        block_size=32, vocab_size=64, n_layer=2, n_heads=2, n_embd=8, use_moe=False, use_kv_cache=kv,
    )
    return GPTJXMoEForCausalLM(cfg).eval()


@torch.no_grad()
def test_all_ones_mask_equals_no_mask():
    m = _model()
    x = torch.randint(0, 64, (2, 12))
    a = m(x).logits
    b = m(x, attention_mask=torch.ones_like(x)).logits
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_future_tokens_do_not_change_earlier_logits_with_mask():
    m = _model()
    x = torch.randint(0, 64, (1, 12))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 64
    mask = torch.ones_like(x)
    a = m(x, attention_mask=mask).logits[:, :-1]
    b = m(y, attention_mask=mask).logits[:, :-1]
    assert (a - b).abs().max().item() == 0.0


@torch.no_grad()
def test_left_padding_matches_unpadded_with_position_ids():
    m = _model()
    x = torch.randint(1, 64, (1, 8))
    ref = m(x).logits
    pad = 3
    xp = torch.cat([torch.zeros(1, pad, dtype=torch.long), x], dim=1)
    mask = torch.cat([torch.zeros(1, pad), torch.ones(1, 8)], dim=1).long()
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = m(xp, attention_mask=mask, position_ids=pos).logits[:, pad:]
    assert torch.isfinite(m(xp, attention_mask=mask, position_ids=pos).logits).all()
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_kv_cache_decode_matches_full_forward_with_mask():
    m = _model(kv=True)
    x = torch.randint(0, 64, (1, 9))
    full = m(x, attention_mask=torch.ones_like(x), use_cache=False).logits[:, -1]
    pre = m(x[:, :-1], attention_mask=torch.ones(1, 8, dtype=torch.long), use_cache=True)
    step = m(x[:, -1:], attention_mask=torch.ones(1, 9, dtype=torch.long), past_key_values=pre.past_key_values,
             use_cache=True).logits[:, -1]
    torch.testing.assert_close(step, full, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_left_padded_batched_generate_matches_single_sequence_generate():
    """Batched sampling for RL: left-padded rows must generate exactly what each prompt does alone."""
    for kv in (False, True):
        m = _model(kv=kv)
        m.generation_config.pad_token_id = 0
        prompts = [torch.randint(1, 64, (n,)) for n in (3, 6, 5)]
        singles = [m.generate(p[None], max_new_tokens=6, do_sample=False, use_cache=kv)[0, len(p):] for p in prompts]
        width = max(len(p) for p in prompts)
        ids = torch.stack([torch.cat([torch.zeros(width - len(p), dtype=torch.long), p]) for p in prompts])
        mask = torch.stack([torch.cat([torch.zeros(width - len(p)), torch.ones(len(p))]).long() for p in prompts])
        batched = m.generate(ids, attention_mask=mask, max_new_tokens=6, do_sample=False, use_cache=kv)[:, width:]
        for row, single in zip(batched, singles):
            assert row.tolist() == single.tolist(), f"use_cache={kv}"
