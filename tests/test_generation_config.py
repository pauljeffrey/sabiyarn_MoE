"""The trainer's generation configs must actually run on this model.

These import the real config dicts off Trainer, so they keep testing whatever is actually configured. The
beam-search case is kept even though no shipped config uses num_beams > 1 any more (it degenerates on these
weights -- see _GENERATION_CONFIG's comment): beam search reorders the KV cache through the model's own
`_reorder_cache` every step, which nothing else exercises, so the test guards that path against bit-rot in
case anyone turns beams back on.
"""
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_stub = types.ModuleType("training.constant_tokens")  # the real one downloads a tokenizer at import
for _n, _v in dict(MASK=-100, assistant_token=1, end_of_text_token=2, system_token=3, user_token=4).items():
    setattr(_stub, _n, _v)
sys.modules["training.constant_tokens"] = _stub

from sabiyarn.model.configuration import GPTJXMoEConfig  # noqa: E402
from sabiyarn.model.modeling import GPTJXMoEForCausalLM  # noqa: E402
from training.new_train import Trainer  # noqa: E402

EOS = 2
VOCAB = 64


def _model():
    torch.manual_seed(0)
    cfg = GPTJXMoEConfig(
        block_size=64, vocab_size=VOCAB, n_layer=2, n_heads=2, n_embd=8, use_moe=True, use_kv_cache=True,
        num_experts=4, num_experts_per_tok=2, moe_dim=16, expert_per_layer={"0": 2, "1": 4},
    )
    return GPTJXMoEForCausalLM(cfg).eval()


def _trainer(model, capture: list | None = None):
    """A Trainer with just enough wired up to call _generate_with_config."""
    t = object.__new__(Trainer)
    t.model = model
    t.fsdp_plugin = None
    t.tokenizer = types.SimpleNamespace(eos_token_id=EOS, pad_token_id=None)
    t.accelerator = types.SimpleNamespace(unwrap_model=lambda m: m)
    if capture is not None:
        real = model.generate
        model.generate = lambda *a, **kw: (capture.append(kw), real(*a, **kw))[1]
    return t


@pytest.mark.parametrize("name", ["_GENERATION_CONFIG", "_STARTUP_SAMPLE_CONFIG", "_STARTUP_GREEDY_CONFIG"])
@pytest.mark.parametrize("batch", [1, 5])  # 1 = the startup comparison, 5 = _sample_prompt mid-training
def test_trainer_generation_configs_run(name, batch):
    cfg = dict(getattr(Trainer, name), max_new_tokens=8)  # shortened; the decoding knobs are what matter
    model = _model()
    out = _trainer(model)._generate_with_config(torch.randint(3, VOCAB, (batch, 5)), cfg)
    assert out.size(0) == batch and out.size(1) > 5
    assert torch.isfinite(out.float()).all()


def test_eos_token_id_comes_from_the_tokenizer():
    """config.json on the Hub has no eos_token_id, so without this generation never stops early."""
    captured: list[dict] = []
    model = _model()
    _trainer(model, captured)._generate_with_config(torch.randint(3, VOCAB, (1, 5)),
                                                    dict(Trainer._GENERATION_CONFIG, max_new_tokens=4))
    assert captured[0]["eos_token_id"] == EOS
    assert captured[0]["pad_token_id"] == EOS  # pad_token_id is None on this tokenizer -> falls back to eos


def test_beam_search_still_works_if_re_enabled():
    """No shipped config sets num_beams > 1, but _reorder_cache has no other coverage."""
    model = _model()
    out = _trainer(model)._generate_with_config(
        torch.randint(3, VOCAB, (2, 5)),
        dict(max_new_tokens=8, num_beams=3, do_sample=False, early_stopping=True, length_penalty=1.0))
    assert out.size(0) == 2 and torch.isfinite(out.float()).all()


def test_an_explicit_eos_in_the_config_is_not_overridden():
    captured: list[dict] = []
    model = _model()
    _trainer(model, captured)._generate_with_config(torch.randint(3, VOCAB, (1, 5)),
                                                    dict(Trainer._GENERATION_CONFIG, max_new_tokens=4, eos_token_id=9))
    assert captured[0]["eos_token_id"] == 9


def test_generation_actually_stops_at_eos():
    """Rig the lm_head so </s> is always the argmax: greedy decoding must stop instead of padding to length."""
    model = _model()
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[EOS] = 1e4  # tied to wte, so this also perturbs embeddings -- fine, we only read argmax
    out = _trainer(model)._generate_with_config(
        torch.randint(3, VOCAB, (1, 5)), dict(max_new_tokens=20, do_sample=False, num_beams=1))
    assert out.size(1) < 5 + 20, "generation ran to max_new_tokens despite emitting eos every step"
    assert out[0, -1].item() == EOS
