"""Shared machinery for SFT / DPO / RL: model + tokenizer loading, log-probs, optimiser, run bookkeeping.

Everything here mirrors how training/new_train.py treats the model, so a post-trained checkpoint is
interchangeable with a pretrained one:
  * the model class comes from this repo (`model_code: local`) or from the Hub files next to the weights,
    and saved checkpoints ship modeling.py / configuration.py / auto_map (sabiyarn.hub);
  * fp32 master weights + bf16 autocast (Accelerate `mixed_precision: bf16`), DDP across GPUs;
  * lr comes from training.lr_schedule, metrics go to MLflow through training.tracking;
  * prompts are rendered with the one canonical chat template (sabiyarn/chat_template.jinja).

Two model-specific facts drive the details below:
  * MoE router noise: in train mode the router adds N(0, 0.1) noise to its logits. A policy scored with noise
    and a reference scored without would disagree even at step 0, so policies are run in eval mode (dropout is
    off, gradients still flow) and DPO's loss starts at exactly ln 2.
  * `forward()` builds a KV cache when config.use_kv_cache is set; every scoring call passes use_cache=False.
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import structlog
import torch
import torch.nn.functional as F

from rl.config import RLConfig
from sabiyarn.chat import use_sabiyarn_chat_template
from sabiyarn.hub import register_local_model_code
from training.lr_schedule import lr_at
from training.tracking import MlflowTracker

LOG = structlog.get_logger()

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# --------------------------------------------------------------------------------------------- loading


def load_tokenizer(cfg: RLConfig):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer, trust_remote_code=cfg.trust_remote_code, token=hf_token())
    return use_sabiyarn_chat_template(tok)  # the Hub copy of the template can lag behind the SFT data's


def pad_id_of(tok) -> int:
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def load_model(cfg: RLConfig, path: Optional[str] = None, *, trainable: bool = True):
    path = path or cfg.model_path
    # reference in the SAME dtype as the policy: at step 0 both must produce identical log-probs
    kwargs = {"torch_dtype": _DTYPES[cfg.param_dtype]}
    if cfg.model_code == "local":
        from sabiyarn.model.modeling import GPTJXMoEForCausalLM

        register_local_model_code()
        model = GPTJXMoEForCausalLM.from_pretrained(path, **kwargs)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, token=hf_token(), **kwargs)
    sparse = cfg.moe_dispatch == "sparse"
    for module in model.modules():
        if hasattr(module, "sparse_dispatch"):
            module.sparse_dispatch = sparse
    model.config.moe_sparse_dispatch = sparse
    model.config.use_cache = False
    model.eval()  # see module docstring: no router noise / dropout for policy OR reference
    for p in model.parameters():
        p.requires_grad = trainable
    return model


# --------------------------------------------------------------------------------------------- log-probs


def token_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """(B, T-1) log p(input_ids[:, t+1] | input_ids[:, :t+1]) in fp32. Right-padded input, 2D padding mask.
    `temperature` scores under the tempered distribution the RL sampler draws from."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1].float()
    if temperature != 1.0:
        logits = logits / temperature
    target = input_ids[:, 1:]
    return logits.gather(-1, target.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(dim=-1)


def masked_sum_and_count(logps: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = loss_mask[:, 1:].to(logps.dtype)  # loss_mask is aligned with input_ids; logps with input_ids[:, 1:]
    return (logps * m).sum(-1), m.sum(-1)


# --------------------------------------------------------------------------------------------- optimiser


def build_optimizer(model, cfg: RLConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and "wte" not in name and "wpe" not in name else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=fused or None)


def steps_for(cfg: RLConfig, batches_per_epoch: int) -> int:
    """Optimizer steps for the run: epochs of `batches_per_epoch` micro-batches, `grad_accum_steps` per step."""
    per_epoch = max(1, batches_per_epoch // cfg.grad_accum_steps)
    total = int(math.ceil(per_epoch * cfg.epochs))
    return min(total, cfg.max_steps) if cfg.max_steps > 0 else total


# --------------------------------------------------------------------------------------------- run bookkeeping


class Run:
    """Accelerator + tracking + checkpoint/push for one post-training run (all ranks construct it)."""

    def __init__(self, cfg: RLConfig, *, cpu: bool = False, grad_accum: Optional[int] = None):
        from accelerate import Accelerator
        from accelerate.utils import set_seed

        self.cfg = cfg
        set_seed(cfg.seed)
        self.accelerator = Accelerator(
            mixed_precision=cfg.mixed_precision, gradient_accumulation_steps=grad_accum or cfg.grad_accum_steps, cpu=cpu,
        )
        self.device = self.accelerator.device
        self.master = self.accelerator.is_main_process
        self.tracker = MlflowTracker()
        self.step = 0
        self.pad_id = 0  # set by the algorithm once the tokenizer is loaded
        self._t0 = time.time()

    # -- lifecycle
    def start_tracking(self, extra_params: Optional[dict] = None) -> None:
        if not (self.master and self.cfg.mlflow):
            return
        params = {k: v for k, v in vars(self.cfg).items()}
        params.update(extra_params or {}, world_size=self.accelerator.num_processes)
        self.tracker.start(
            tracking_uri=self.cfg.mlflow_tracking_uri, experiment_name=self.cfg.mlflow_experiment,
            run_name=self.cfg.run_name or f"{self.cfg.algo}-{time.strftime('%m%d-%H%M')}",
            run_id=os.environ.get("MLFLOW_RUN_ID"), params=params, tags={"algo": self.cfg.algo},
        )

    def finish(self, status: str = "FINISHED") -> None:
        if self.master:
            self.tracker.end(status=status)

    # -- per step
    def lr(self, it: int) -> float:
        return lr_at(it, self.cfg)

    def apply_lr(self, optimizer, it: int) -> float:
        lr = self.lr(it)
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr

    def log(self, metrics: dict[str, Optional[float]], step: int, *, echo: bool = False) -> None:
        if not self.master:
            return
        self.tracker.log_metrics(metrics, step=step)
        if echo:
            LOG.info("step", step=step, elapsed_s=round(time.time() - self._t0), **{
                k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items() if v is not None})

    def mean_over_ranks(self, values: dict[str, float]) -> dict[str, float]:
        """Average per-rank scalars so what is logged is the global figure, not rank 0's slice."""
        if self.accelerator.num_processes == 1:
            return values
        keys = sorted(values)
        t = torch.tensor([values[k] for k in keys], device=self.device, dtype=torch.float32)
        t = self.accelerator.reduce(t, reduction="mean")
        return dict(zip(keys, t.tolist()))

    # -- checkpoints
    def save(self, model, tokenizer, tag: str, extra_state: Optional[dict] = None) -> str:
        out = Path(self.cfg.out_dir) / tag
        self.accelerator.wait_for_everyone()
        if self.master:
            out.mkdir(parents=True, exist_ok=True)
            # Every rank holds a full copy under DDP, so rank 0 alone writes it.
            self.accelerator.unwrap_model(model).save_pretrained(out, safe_serialization=True)
            tokenizer.save_pretrained(out)  # includes the canonical chat template
            (out / "rl_state.json").write_text(json.dumps(
                {"algo": self.cfg.algo, "step": self.step, "config": _jsonable(vars(self.cfg)), **(extra_state or {})},
                indent=2), encoding="utf-8")
            LOG.info("checkpoint_saved", path=str(out), step=self.step)
        self.accelerator.wait_for_everyone()
        return str(out)

    def publish(self, ckpt_dir: str) -> None:
        """Optional Hub push + S3 mirror of the final checkpoint (rank 0 only)."""
        if not self.master:
            return
        if self.cfg.hf_push_repo_id:
            from huggingface_hub import HfApi

            token = os.environ.get("HF_WRITE_TOKEN") or hf_token()
            api = HfApi(token=token)
            api.create_repo(self.cfg.hf_push_repo_id, exist_ok=True, private=True)
            api.upload_folder(folder_path=ckpt_dir, repo_id=self.cfg.hf_push_repo_id,
                              commit_message=f"{self.cfg.algo} checkpoint (step {self.step})")
            LOG.info("hf_pushed", repo=self.cfg.hf_push_repo_id)
        if self.cfg.s3_prefix:
            from training.load_config import load_train_config
            from training.s3_utils import upload_folder

            t = load_train_config()
            if not (t.s3_bucket and t.s3_endpoint and t.s3_access_key and t.s3_secret_key):
                LOG.warning("s3_skipped", reason="S3_* environment variables / train_config s3 section incomplete")
                return
            upload_folder(ckpt_dir, f"{self.cfg.s3_prefix.rstrip('/')}/{Path(ckpt_dir).name}", bucket=t.s3_bucket,
                          endpoint=t.s3_endpoint, access_key=t.s3_access_key, secret_key=t.s3_secret_key,
                          prefix=t.s3_prefix, override=True)
            LOG.info("s3_uploaded", prefix=self.cfg.s3_prefix)


def _jsonable(d: dict) -> dict:
    return {k: (v if isinstance(v, (str, int, float, bool, type(None), list)) else str(v)) for k, v in d.items()}


def maybe_no_sync(accelerator, model, sync: bool):
    """DDP: skip the gradient all-reduce on every chunk but the last of a step."""
    return nullcontext() if sync else accelerator.no_sync(model)
