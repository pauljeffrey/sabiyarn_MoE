"""rl/: config, data rendering, DPO maths and training, group-baseline RL, rewards -- on a tiny random model."""
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from rl.config import load_rl_config
from rl.rewards import ChrfReward, CometWorkerReward

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tiny_env(tmp_path_factory):
    """(tokenizer dir, model dir, dpo.jsonl path, sft.jsonl path) for a ~50k-parameter MoE."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    from sabiyarn.hub import register_local_model_code
    from sabiyarn.model.configuration import GPTJXMoEConfig
    from sabiyarn.model.modeling import GPTJXMoEForCausalLM

    root = tmp_path_factory.mktemp("rl")
    words = ["ba", "ka", "mo", "ni", "sa", "yo", "ru", "te", "wa", "do"]
    corpus = [" ".join(words[(i + j) % 10] for j in range(8)) for i in range(50)] + ["hello there how are you today"] * 5
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    specials = ["<pad>", "<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>"]
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=120, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    tok_dir = root / "tok"
    tok.save_pretrained(tok_dir)

    torch.manual_seed(0)
    cfg = GPTJXMoEConfig(block_size=256, vocab_size=len(tok), n_layer=2, n_heads=2, n_embd=16, use_moe=True,
                         num_experts=4, num_experts_per_tok=2, moe_dim=32, expert_per_layer={"0": 2, "1": 4},
                         use_kv_cache=True)
    model = GPTJXMoEForCausalLM(cfg)
    register_local_model_code()
    model_dir = root / "model"
    model.save_pretrained(model_dir)

    def convo(i):
        return [{"role": "system", "content": "be brief"}, {"role": "user", "content": f"{words[i % 10]} {words[(i + 3) % 10]} ka"}]

    dpo = [{"id": i, "language": "yor" if i % 2 else "hau", "prompt_messages": convo(i),
            "chosen": " ".join(words[(i + j) % 10] for j in range(4)), "rejected": "do do do do do do do do"} for i in range(80)]
    sft = [{"id": i, "language": "yor", "input": words[i % 10], "prompt_messages": convo(i), "response": words[(i + 1) % 10] + " ka"}
           for i in range(60)]
    dpo_path, sft_path = root / "dpo.jsonl", root / "sft.jsonl"
    dpo_path.write_text("\n".join(json.dumps(r) for r in dpo), encoding="utf-8")
    sft_path.write_text("\n".join(json.dumps(r) for r in sft), encoding="utf-8")
    return tok_dir, model_dir, dpo_path, sft_path


def make_cfg(tiny_env, tmp_path, **kw):
    tok_dir, model_dir, dpo_path, _ = tiny_env
    base = dict(algo="dpo", model_path=str(model_dir), tokenizer=str(tok_dir), data_path=str(dpo_path),
                out_dir=str(tmp_path / "out"), mixed_precision="no", mlflow=False, max_prompt_len=64, max_seq_len=96,
                batch_size=4, grad_accum_steps=2, eval_samples=8, log_every=1, eval_every=1000, save_every=0,
                learning_rate=1e-3, warmup_ratio=0.0, max_new_tokens=6)
    base.update(kw)
    return load_rl_config(str(ROOT / "rl" / "config.yaml"), **base)


# ------------------------------------------------------------------------------------------- config


def test_shipped_configs_parse():
    a = load_rl_config(str(ROOT / "rl" / "config.yaml"))
    b = load_rl_config(str(ROOT / "rlhf" / "configs" / "default.yaml"))
    assert a.algo == "dpo" and b.algo == "rl"
    assert b.model_path == "Aletheia-ng/SabiYarn_MoE-280M" and b.tokenizer == "BeardedMonster/SabiYarn-32k"


def test_unknown_key_is_an_error_and_env_overrides(monkeypatch, tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  learning_rat: 1e-6\n")
    with pytest.raises(ValueError, match="learning_rat"):
        load_rl_config(str(p))
    p.write_text("train:\n  learning_rate: 1e-6\n  languages_typo_free: 1\n".replace("  languages_typo_free: 1\n", ""))
    monkeypatch.setenv("RL_LEARNING_RATE", "3e-7")
    monkeypatch.setenv("RL_LANGUAGES", "yor, hau")
    monkeypatch.setenv("RL_MLFLOW", "false")
    cfg = load_rl_config(str(p))
    assert cfg.learning_rate == 3e-7 and cfg.languages == ["yor", "hau"] and cfg.mlflow is False


# ------------------------------------------------------------------------------------------- data


def test_encoding_masks_the_prompt_and_drops_overlong(tiny_env, tmp_path):
    from rl.common import load_tokenizer
    from rl.data import encode_dpo, load_records, pad_batch

    cfg = make_cfg(tiny_env, tmp_path)
    tok = load_tokenizer(cfg)
    pairs = encode_dpo(tok, load_records(cfg), cfg)
    assert len(pairs) == 80
    c, r = pairs[0]
    assert c.prompt_ids == r.prompt_ids and c.answer_ids[-1] == tok.eos_token_id
    text = tok.decode(c.prompt_ids)
    assert text.startswith("<s><|system|>be brief</s><|user|>") and text.endswith("<|assistant|>")
    b = pad_batch([c, r], tok.pad_token_id)
    assert b["loss_mask"][0].sum().item() == len(c.answer_ids) and b["loss_mask"][0, : len(c.prompt_ids)].sum().item() == 0
    tight = make_cfg(tiny_env, tmp_path, max_prompt_len=8, max_seq_len=12)
    assert encode_dpo(tok, load_records(tight), tight) == []


# ------------------------------------------------------------------------------------------- DPO


def test_dpo_loss_math():
    from rl.dpo import dpo_loss

    z = torch.zeros(3)
    loss, rc, rr = dpo_loss(z, z, z, z, beta=0.1)
    assert torch.allclose(loss, torch.full((3,), math.log(2)))
    # policy prefers chosen more than the reference does -> lower loss, positive margin
    l2, rc, rr = dpo_loss(torch.tensor([1.0]), torch.tensor([-1.0]), torch.zeros(1), torch.zeros(1), beta=0.5)
    assert l2.item() < math.log(2) and (rc - rr).item() == pytest.approx(0.5 * 2.0)
    # label smoothing at 0.5 makes the loss independent of the margin
    ls, _, _ = dpo_loss(torch.tensor([5.0]), torch.tensor([-5.0]), torch.zeros(1), torch.zeros(1), beta=1.0, label_smoothing=0.5)
    assert ls.item() == pytest.approx(0.5 * (math.log1p(math.exp(-10)) + math.log1p(math.exp(10))))
    ipo, _, _ = dpo_loss(torch.tensor([1.0]), torch.tensor([0.0]), torch.zeros(1), torch.zeros(1), beta=0.5, loss_type="ipo")
    assert ipo.item() == pytest.approx((1.0 - 1.0) ** 2)
    hinge, _, _ = dpo_loss(z, z, z, z, beta=0.1, loss_type="hinge")
    assert torch.allclose(hinge, torch.ones(3))


def test_dpo_first_loss_is_ln2_learns_and_ships_loadable_code(tiny_env, tmp_path):
    from rl.common import Run
    from rl.dpo import train_dpo

    cfg = make_cfg(tiny_env, tmp_path, epochs=3, eval_every=5)
    seen = []
    run = Run(cfg, cpu=True)
    orig = run.log
    run.log = lambda m, step, echo=False: (seen.append((step, dict(m))), orig(m, step, echo=echo))[-1]
    final = train_dpo(cfg, run)

    first = next(m for _, m in seen if "train/loss" in m)
    assert first["train/init_logratio_absmax"] < 1e-4  # policy == reference at step 0 (eval mode, same dtype)
    assert first["train/loss"] == pytest.approx(math.log(2), abs=0.02)
    evals = [m for _, m in seen if "eval/loss" in m]
    assert evals[-1]["eval/reward_accuracy"] > 0.9 and evals[-1]["eval/loss"] < math.log(2)
    out = Path(final)
    assert (out / "model.safetensors").exists() and (out / "modeling.py").exists() and (out / "configuration.py").exists()
    assert "auto_map" in json.loads((out / "config.json").read_text())
    assert (out / "chat_template.jinja").exists() or "chat_template" in (out / "tokenizer_config.json").read_text()
    assert json.loads((out / "rl_state.json").read_text())["algo"] == "dpo"


def test_dpo_length_normalised_and_sft_term_run(tiny_env, tmp_path):
    from rl.common import Run
    from rl.dpo import train_dpo

    cfg = make_cfg(tiny_env, tmp_path, max_steps=2, length_normalize=True, sft_alpha=0.1, dpo_loss="ipo", beta=0.5)
    assert Path(train_dpo(cfg, Run(cfg, cpu=True))).is_dir()


# ------------------------------------------------------------------------------------------- SFT


def test_sft_reduces_loss(tiny_env, tmp_path):
    from rl.common import Run
    from rl.sft import train_sft

    tok_dir, model_dir, _, sft_path = tiny_env
    cfg = make_cfg(tiny_env, tmp_path, algo="sft", data_path=str(sft_path), epochs=4, eval_every=3, learning_rate=3e-3)
    seen = []
    run = Run(cfg, cpu=True)
    orig = run.log
    run.log = lambda m, step, echo=False: (seen.append(m), orig(m, step, echo=echo))[-1]
    train_sft(cfg, run)
    losses = [m["train/loss"] for m in seen if "train/loss" in m]
    assert losses[-1] < losses[0]


# ------------------------------------------------------------------------------------------- RL


def test_group_advantages():
    from rl.policy_gradient import group_advantages

    adv, flat = group_advantages([1.0, 3.0, 5.0, 2.0, 2.0, 2.0], group_size=3, std_norm=False)
    assert adv.tolist() == pytest.approx([-2.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    assert flat == pytest.approx(0.5)  # the second group has zero variance
    adv, _ = group_advantages([1.0, 3.0, 5.0], group_size=3, std_norm=True)
    assert adv.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-3)


def test_batched_sampling_groups_rows_by_prompt_and_cuts_at_eos(tiny_env, tmp_path):
    from rl.common import load_model, load_tokenizer
    from rl.data import render_prompt
    from rl.policy_gradient import sample_completions

    cfg = make_cfg(tiny_env, tmp_path, algo="rl", group_size=3)
    tok, model = load_tokenizer(cfg), load_model(cfg)
    prompts = [render_prompt(tok, [{"role": "user", "content": t}]) for t in ("ba ka", "mo ni sa yo ru te")]
    comps = sample_completions(model, tok, prompts, cfg)
    assert len(comps) == 6 and all(0 < len(c) <= cfg.max_new_tokens for c in comps)
    assert all(tok.eos_token_id not in c[:-1] for c in comps)
    greedy = sample_completions(model, tok, prompts, cfg, greedy=True)
    single = [sample_completions(model, tok, [p], cfg, greedy=True)[0] for p in prompts]
    assert greedy == single  # left-padded batching == one prompt at a time


def test_rl_runs_end_to_end_and_moves_the_policy(tiny_env, tmp_path):
    from rl.common import Run, load_model
    from rl.policy_gradient import train_rl

    _, model_dir, _, sft_path = tiny_env
    rows = [json.loads(l) for l in sft_path.read_text().splitlines()]
    for r in rows:
        r["reference"] = r["response"]
    cfg = make_cfg(tiny_env, tmp_path, algo="rl", group_size=4, batch_size=2, grad_accum_steps=1, max_steps=3,
                   rl_micro_batch=4, learning_rate=1e-3, kl_coef=0.05)
    before = load_model(cfg).state_dict()
    final = train_rl(cfg, reward_fn=ChrfReward(), run=Run(cfg, cpu=True, grad_accum=1), records=rows)
    cfg2 = make_cfg(tiny_env, tmp_path, model_path=final)
    after = load_model(cfg2).state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert (Path(final) / "modeling.py").exists()


# ------------------------------------------------------------------------------------------- rewards


def test_chrf_reward_orders_by_similarity():
    r = ChrfReward()
    s = r([{"reference": "the cat sat on the mat"}] * 2, ["the cat sat on the mat", "completely different words"])
    assert s[0] == pytest.approx(1.0) and s[1] < 0.3


def test_comet_worker_protocol_survives_library_noise(tmp_path):
    """The worker is a separate process; prints from libraries must not corrupt the JSON protocol."""
    reward = CometWorkerReward(sys.executable, "unused", batch_size=4, use_gpu=False, fake=True)
    try:
        items = [{"source": "s", "reference": "abcdef"}, {"source": "s", "reference": "abcdef"}]
        a = reward(items, ["abcdef", "xyz"])
        b = reward(items, ["abcdef", "abc"])  # second request on the same worker
        assert a[0] == pytest.approx(1.0) and a[1] < a[0] and b[1] > a[1]
    finally:
        reward.close()


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_rl_update_raises_logprob_of_positive_advantage_and_lowers_negative(tiny_env, tmp_path, sign):
    from rl.common import Run, build_optimizer, load_model, load_tokenizer, pad_id_of, token_logps
    from rl.data import Encoded, pad_batch, render_prompt
    from rl.policy_gradient import rl_update

    cfg = make_cfg(tiny_env, tmp_path, algo="rl", learning_rate=1e-2, kl_coef=0.0, rl_micro_batch=2, group_size=2)
    tok = load_tokenizer(cfg)
    run = Run(cfg, cpu=True, grad_accum=1)
    run.pad_id = pad_id_of(tok)
    policy, ref = load_model(cfg), load_model(cfg, trainable=False)
    opt = build_optimizer(policy, cfg)
    prompt = render_prompt(tok, [{"role": "user", "content": "ba ka"}])
    seqs = [Encoded(prompt, tok("mo ni", add_special_tokens=False)["input_ids"] + [tok.eos_token_id])] * 2

    def logp():
        b = pad_batch(seqs[:1], run.pad_id)
        return (token_logps(policy, b["input_ids"], b["attention_mask"]) * b["loss_mask"][:, 1:]).sum().item()

    before = logp()
    rl_update(run, policy, ref, opt, seqs, torch.tensor([sign, sign]), cfg, lr=1e-2)
    assert (logp() - before) * sign > 0
