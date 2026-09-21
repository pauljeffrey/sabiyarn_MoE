"""Configuration for the post-training stack (rl/: SFT warm-start, DPO, reward-based RL).

One YAML file, sections are just for readability: every key is a field of `RLConfig` and unknown keys are an
error (a typo'd hyper-parameter silently falling back to its default is how runs get wasted). Any field can also
be overridden from the environment as `RL_<FIELD>` (e.g. `RL_LEARNING_RATE=5e-7`, `RL_MODEL_PATH=out/ckpt`),
which is how `modal run` / vast.ai launches change a value without editing the file. Secrets (HF / S3 / MLflow)
are only ever read from the environment (.env), never from the YAML.

Presets: rl/config.yaml (DPO on data-gen's dpo.jsonl), rlhf/configs/default.yaml (AfriCOMET translation RL).
"""

from __future__ import annotations

import dataclasses
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "rl" / "config.yaml"

ALGOS = ("sft", "dpo", "rl")
DPO_LOSSES = ("sigmoid", "ipo", "hinge")


@dataclass
class RLConfig:
    algo: str = "dpo"  # sft | dpo | rl

    # ---- model ---------------------------------------------------------------------------------------
    model_path: str = "Aletheia-ng/SabiYarn_MoE-280M"  # checkpoint dir or Hub id: the policy's starting point
    tokenizer: str = "BeardedMonster/SabiYarn-32k"
    model_code: str = "local"  # local: sabiyarn/model/*.py from this repo | hub: trust_remote_code files next to the weights
    param_dtype: str = "float32"  # fp32 master weights + bf16 autocast (see training.param_dtype for why)
    moe_dispatch: str = "sparse"  # sparse | dense
    reference_path: Optional[str] = None  # frozen reference for DPO / KL; null = the starting policy

    # ---- data ----------------------------------------------------------------------------------------
    data_path: str = "data-gen/data/processed/dpo.jsonl"  # jsonl (data-gen output) ...
    dataset_id: Optional[str] = None  # ... or a Hub dataset (overrides data_path)
    dataset_split: str = "train"
    max_samples: int = 0  # 0 = all
    eval_samples: int = 500  # held out from the same file (deterministic, by seed)
    max_prompt_len: int = 512
    max_seq_len: int = 1024  # prompt + completion tokens
    languages: list = field(default_factory=list)  # keep only these language codes (data-gen records carry `language`)
    seed: int = 42

    # ---- optimisation --------------------------------------------------------------------------------
    batch_size: int = 4  # per device (DPO: pairs; sft: examples; rl: prompts)
    grad_accum_steps: int = 4
    epochs: float = 1.0
    max_steps: int = 0  # 0 = epochs decide
    learning_rate: float = 5e-7  # DPO wants ~1e-7..1e-6, SFT ~1e-5, RL ~1e-6
    min_lr: float = 0.0
    warmup_ratio: float = 0.1
    scheduler: str = "cosine"  # cosine | linear | wsd
    wsd_decay_frac: float = 0.2
    wsd_decay_shape: str = "sqrt"
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"  # bf16 | no
    eval_every: int = 100
    save_every: int = 500
    log_every: int = 10

    # ---- DPO -----------------------------------------------------------------------------------------
    beta: float = 0.1  # KL strength: higher = stay closer to the reference
    dpo_loss: str = "sigmoid"  # sigmoid | ipo | hinge
    label_smoothing: float = 0.0  # conservative DPO: assumed label-noise rate (sigmoid only)
    length_normalize: bool = False  # average (not sum) token log-probs; recommended for ipo
    sft_alpha: float = 0.0  # + alpha * NLL(chosen): keeps chosen likelihood from collapsing (RPO)

    # ---- reward-based RL (group-baseline policy gradient) ---------------------------------------------
    group_size: int = 4  # completions sampled per prompt; advantage = reward - group mean
    max_new_tokens: int = 128
    temperature: float = 1.0  # sampling AND scoring temperature; 1.0 keeps the policy-gradient estimator unbiased
    top_p: float = 1.0  # < 1 biases the estimator (it is not part of the scored distribution)
    kl_coef: float = 0.05  # per-token k3 KL penalty against the reference
    advantage_std_norm: bool = True  # divide advantages by the group std
    rl_micro_batch: int = 16  # sequences per forward/backward chunk
    reward: str = "chrf"  # chrf | africomet
    reward_id: str = "masakhane/africomet-stl"
    reward_python: Optional[str] = None  # python of an env with `unbabel-comet` (it needs transformers<5); null = in-process
    reward_batch_size: int = 16
    reward_gpu: bool = True

    # ---- output / tracking ----------------------------------------------------------------------------
    out_dir: str = "outputs/dpo"
    hf_push_repo_id: Optional[str] = None  # push the final model here (needs HF_WRITE_TOKEN / HF_TOKEN)
    s3_prefix: Optional[str] = None  # mirror out_dir to S3 under this prefix (uses training's S3_* env vars)
    mlflow: bool = True
    mlflow_tracking_uri: Optional[str] = None  # MLFLOW_TRACKING_URI env wins; null = ./mlruns
    mlflow_experiment: str = "sabiyarn-posttraining"
    run_name: Optional[str] = None
    trust_remote_code: bool = True  # for the tokenizer (BeardedMonster/SabiYarn-32k ships one)

    # derived at run time (not YAML keys)
    max_iters: int = 0
    lr_decay_iters: int = 0
    warmup_iters: int = 0

    def validate(self) -> "RLConfig":
        if self.algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}, got {self.algo!r}")
        if self.dpo_loss not in DPO_LOSSES:
            raise ValueError(f"dpo_loss must be one of {DPO_LOSSES}, got {self.dpo_loss!r}")
        if self.model_code not in ("local", "hub"):
            raise ValueError("model_code must be 'local' or 'hub'")
        if self.moe_dispatch not in ("sparse", "dense"):
            raise ValueError("moe_dispatch must be 'sparse' or 'dense'")
        if self.mixed_precision not in ("bf16", "no"):
            raise ValueError("mixed_precision must be 'bf16' or 'no'")
        if self.param_dtype not in ("float32", "bfloat16"):
            raise ValueError("param_dtype must be 'float32' or 'bfloat16'")
        if self.max_prompt_len >= self.max_seq_len:
            raise ValueError("max_prompt_len must be smaller than max_seq_len")
        if self.group_size < 2 and self.algo == "rl":
            raise ValueError("group_size must be >= 2: the baseline is the group mean")
        for name in ("batch_size", "grad_accum_steps", "rl_micro_batch"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        return self

    def set_schedule(self, total_steps: int) -> None:
        """Fill the derived fields lr_schedule.lr_at() reads."""
        self.max_iters = max(1, int(total_steps))
        self.lr_decay_iters = self.max_iters
        self.warmup_iters = int(self.max_iters * self.warmup_ratio)


_FIELD_TYPES = {f.name: t for f, t in ((f, typing.get_type_hints(RLConfig)[f.name]) for f in dataclasses.fields(RLConfig))}
_DERIVED = {"max_iters", "lr_decay_iters", "warmup_iters"}


def _coerce(name: str, value: Any) -> Any:
    tp = _FIELD_TYPES[name]
    args = typing.get_args(tp)
    if type(None) in args:  # Optional[X]
        if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none", "null")):
            return None
        tp = next(a for a in args if a is not type(None))
    if tp is bool:
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    if tp is int:
        return int(float(value))
    if tp is float:
        return float(value)
    if tp is list:
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return list(value or [])
    return str(value)


def _flatten_yaml(raw: dict) -> dict:
    flat: dict = {}
    for key, value in raw.items():
        if isinstance(value, dict) and key not in _FIELD_TYPES:  # a section header
            for k, v in value.items():
                if k in flat:
                    raise ValueError(f"duplicate key {k!r} in config")
                flat[k] = v
        else:
            flat[key] = value
    return flat


def load_rl_config(path: Optional[str] = None, **overrides: Any) -> RLConfig:
    """YAML (RL_CONFIG_PATH or rl/config.yaml) < keyword overrides < RL_<FIELD> environment variables."""
    path = Path(path or os.getenv("RL_CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.exists() else {}
    values = {**_flatten_yaml(raw), **overrides}
    unknown = sorted(set(values) - set(_FIELD_TYPES) | (set(values) & _DERIVED))
    if unknown:
        raise ValueError(f"unknown / non-configurable keys in {path}: {unknown}")
    for name in _FIELD_TYPES:
        env = os.getenv(f"RL_{name.upper()}")
        if env is not None and name not in _DERIVED:
            values[name] = env
    cfg = RLConfig(**{k: _coerce(k, v) for k, v in values.items()})
    return cfg.validate()
