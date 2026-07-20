"""
Confirm build_document_causal_mask plugs into GPTJXMoEForCausalLM.forward
without shape errors, and that it actually changes the model's output
relative to plain causal masking (i.e. it's not silently ignored).
"""
import torch

from sabiyarn.model.configuration import GPTJXMoEConfig
from sabiyarn.model.modeling import GPTJXMoEForCausalLM
from training.training_attention_mask import build_document_causal_mask

EOS = 5


def _tiny_model():
    torch.manual_seed(0)
    config = GPTJXMoEConfig(
        block_size=32, vocab_size=64, n_layer=2, n_heads=2, n_embd=8,
        use_moe=False, use_kv_cache=False,
    )
    model = GPTJXMoEForCausalLM(config)
    model.eval()
    return model


def test_forward_accepts_document_causal_mask_without_error():
    model = _tiny_model()
    input_ids = torch.randint(0, 64, (2, 10))
    input_ids[:, 4] = EOS
    mask = build_document_causal_mask(input_ids, EOS)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask)
    assert out.logits.shape == (2, 10, 64)


def test_document_mask_changes_logits_vs_plain_causal():
    model = _tiny_model()
    input_ids = torch.randint(1, 64, (1, 10))
    input_ids[:, 4] = EOS  # split into two documents

    with torch.no_grad():
        out_plain = model(input_ids=input_ids)
        out_doc = model(input_ids=input_ids, attention_mask=build_document_causal_mask(input_ids, EOS))

    # tokens after the EOS boundary can no longer see the first document,
    # so their logits should differ from plain (unrestricted) causal attention
    assert not torch.allclose(out_plain.logits[:, 5:], out_doc.logits[:, 5:])
    # tokens within the first document are unaffected (nothing to block yet)
    assert torch.allclose(out_plain.logits[:, :4], out_doc.logits[:, :4], atol=1e-5)
