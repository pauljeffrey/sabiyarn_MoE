import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabiyarn.model.configuration import GPTJXMoEConfig  # noqa: E402
from sabiyarn.model.modeling import GPTJXMoEForCausalLM, MoE  # noqa: E402
from training.training_attention_mask import build_document_block_mask, build_document_causal_mask  # noqa: E402

EOS = 2


def tiny_config(**kw):
    base = dict(
        block_size=128, vocab_size=100, n_layer=2, n_heads=2, n_embd=32, use_moe=True,
        num_experts=4, num_experts_per_tok=2, moe_dim=64, expert_per_layer={"0": 2, "1": 4},
    )
    base.update(kw)
    return GPTJXMoEConfig(**base)


@pytest.mark.parametrize("n_experts", [2, 3, 4, 8])
def test_sparse_experts_match_dense_forward_and_gradients(n_experts):
    torch.manual_seed(0)
    moe = MoE(num_experts_per_tok=2, num_experts=n_experts, emb_dim=16, moe_dim=32, sparse_dispatch=False).eval()
    x = torch.randn(3, 11, 16)
    x_sparse = x.clone().requires_grad_(True)
    x_dense = x.clone().requires_grad_(True)

    moe.sparse_dispatch = False
    y_dense = moe(x_dense)
    y_dense.square().sum().backward()
    grads_dense = {n: p.grad.clone() for n, p in moe.named_parameters()}
    moe.zero_grad()

    moe.sparse_dispatch = True
    y_sparse = moe(x_sparse)
    y_sparse.square().sum().backward()

    torch.testing.assert_close(y_sparse, y_dense, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(x_sparse.grad, x_dense.grad, rtol=1e-4, atol=1e-6)
    for n, p in moe.named_parameters():
        torch.testing.assert_close(p.grad, grads_dense[n], rtol=1e-4, atol=1e-6)


def test_sparse_handles_experts_that_receive_no_tokens_and_a_single_token():
    torch.manual_seed(0)
    moe = MoE(num_experts_per_tok=1, num_experts=4, emb_dim=16, moe_dim=32, sparse_dispatch=True).eval()
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0].fill_(1.0)  # every token routes to expert 0; experts 1-3 get nothing
    x = torch.randn(1, 1, 16)
    dense = MoE(num_experts_per_tok=1, num_experts=4, emb_dim=16, moe_dim=32, sparse_dispatch=False).eval()
    dense.load_state_dict(moe.state_dict())
    torch.testing.assert_close(moe(x), dense(x), rtol=1e-5, atol=1e-6)


def _loss_and_grads(model, x, y, attention_mask=None):
    model.zero_grad()
    out = model(input_ids=x, attention_mask=attention_mask, targets=y)
    out.loss.backward()
    return out.loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_full_model_sparse_equals_dense():
    torch.manual_seed(0)
    model = GPTJXMoEForCausalLM(tiny_config()).eval()
    x = torch.randint(3, 100, (2, 32))
    y = torch.randint(3, 100, (2, 32))
    for m in model.modules():
        if isinstance(m, MoE):
            m.sparse_dispatch = False
    loss_d, grads_d = _loss_and_grads(model, x, y)
    for m in model.modules():
        if isinstance(m, MoE):
            m.sparse_dispatch = True
    loss_s, grads_s = _loss_and_grads(model, x, y)
    torch.testing.assert_close(loss_s, loss_d, rtol=1e-5, atol=1e-6)
    for n in grads_d:
        torch.testing.assert_close(grads_s[n], grads_d[n], rtol=1e-3, atol=1e-6)


def _flex_inputs():
    x = torch.randint(3, 100, (2, 128))
    x[0, 40] = EOS  # documents end at fixed positions, different per row
    x[0, 90] = EOS
    x[1, 10] = EOS
    y = torch.randint(3, 100, (2, 128))
    return x, y


def test_flex_block_mask_matches_dense_document_mask_forward():
    """Same logits and loss as the dense boolean document mask, with document boundaries in the
    middle of the sequence. Forward only: FlexAttention has no CPU backward."""
    torch.manual_seed(0)
    model = GPTJXMoEForCausalLM(tiny_config()).eval()
    x, y = _flex_inputs()
    with torch.no_grad():
        dense = model(input_ids=x, attention_mask=build_document_causal_mask(x, EOS), targets=y)
        flex = model(input_ids=x, attention_mask=build_document_block_mask(x, EOS), targets=y)
        causal_only = model(input_ids=x, targets=y)
    torch.testing.assert_close(flex.logits, dense.logits, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(flex.loss, dense.loss, rtol=1e-4, atol=1e-5)
    # the mask is doing real work: without document blocking the loss differs
    assert abs(causal_only.loss.item() - dense.loss.item()) > 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward needs CUDA")
def test_flex_block_mask_gradients_match_dense_on_gpu():
    torch.manual_seed(0)
    model = GPTJXMoEForCausalLM(tiny_config()).cuda().eval()
    x, y = (t.cuda() for t in _flex_inputs())
    loss_d, grads_d = _loss_and_grads(model, x, y, build_document_causal_mask(x, EOS))
    loss_f, grads_f = _loss_and_grads(model, x, y, build_document_block_mask(x, EOS))
    torch.testing.assert_close(loss_f, loss_d, rtol=1e-3, atol=1e-4)
    for n in grads_d:
        torch.testing.assert_close(grads_f[n], grads_d[n], rtol=1e-2, atol=1e-4)


def test_local_model_class_round_trips_through_save_and_from_pretrained(tmp_path):
    torch.manual_seed(0)
    model = GPTJXMoEForCausalLM(tiny_config()).eval()
    model.save_pretrained(tmp_path)
    reloaded = GPTJXMoEForCausalLM.from_pretrained(tmp_path).eval()
    x = torch.randint(3, 100, (1, 16))
    torch.testing.assert_close(reloaded(input_ids=x).logits, model(input_ids=x).logits)
    assert reloaded.lm_head.weight is reloaded.transformer.wte.weight  # tie survived
    assert reloaded.config.moe_sparse_dispatch is True
