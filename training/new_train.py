#!/usr/bin/env python3
"""
SabiYarn HF training — pretrain & SFT, single/multi-GPU/multi-node via
Accelerate + FSDP.

Launch:
  python -m training.new_train                                        # single GPU / CPU smoke test
  torchrun --standalone --nproc_per_node=4 -m training.new_train       # single node, multi-GPU
  # multi-node: run the same command on every node with per-node --node_rank
  torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \\
      --master_addr=<node0_ip> --master_port=29500 -m training.new_train
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime

import numpy as np
import structlog
import torch
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.constant_tokens import MASK, assistant_token, end_of_text_token, system_token, user_token
from training.label_masking import apply_label_mask
from training.load_config import TrainConfig, load_train_config, sampling_weights
from training.training_attention_mask import build_document_causal_mask

LOG = structlog.get_logger()

try:
    from cut_cross_entropy import linear_cross_entropy
    HAS_CCE = True
except ImportError:
    HAS_CCE = False

# lm_head/wte are excluded from FSDP wrapping (see _setup_accelerator) since
# they're tied weights -- sharding one while the other stays a plain
# nn.Parameter would break the tie. That also means raw.lm_head.weight below
# is always the full, un-sharded tensor; no DeepSpeed-style gather-before-use
# dance is needed the way ZeRO-3 required.
_FSDP_IGNORED_MODULES = r"lm_head|transformer\.wte"


# Parameter-name substrings that actually appear in GPTJXMoEForCausalLM, keyed by
# the train_config.yaml `model.weights.freeze_*` flag that should freeze them.
_FREEZE_PATTERNS = {
    "freeze_pos_layer_only": ("wpe",),
    "freeze_emb_layer_only": ("wte",),
    "freeze_router_layer_only": ("mlp.gate",),
    "freeze_experts_only": ("mlp.fc_bank", "mlp.proj_bank"),
    "freeze_ffn_layer_only": ("mlp.c_fc", "mlp.c_proj"),
    "freeze_attn_layer_only": ("attn.",),
}


def _freeze_layers(model, cfg: TrainConfig) -> None:
    """Freeze parameters matching configured layer patterns.

    Must run before accelerator.prepare(): flipping requires_grad after FSDP
    has flattened/sharded parameters is unreliable.
    """
    active = {
        flag: patterns
        for flag, patterns in _FREEZE_PATTERNS.items()
        if getattr(cfg, flag, False)
    }
    if not active:
        return
    frozen = 0
    for name, param in model.named_parameters():
        for patterns in active.values():
            if any(p in name for p in patterns):
                param.requires_grad = False
                frozen += 1
                break
    LOG.info("layers_frozen", count=frozen, active_flags=list(active.keys()))


class Trainer:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.iter_num = 0
        self.best_val = 1e9
        self._setup_accelerator()
        self._setup_dirs()
        self._setup_wandb()
        self._setup_data()
        self._build_model()
        self._build_optimizer()
        self._prepare_for_training()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _accelerate_precision(self) -> str:
        return {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}.get(self.cfg.dtype, "bf16")

    def _setup_accelerator(self):
        # Keep the effective global batch size (train_batch_size * world_size *
        # grad_accum_steps) invariant to world_size, same as the old DDP path.
        world_size_env = int(os.environ.get("WORLD_SIZE", 1))
        if world_size_env > 1 and self.cfg.gradient_accumulation_steps % world_size_env == 0:
            self.cfg.gradient_accumulation_steps //= world_size_env
        self.cfg.gradient_accumulation_steps = max(1, self.cfg.gradient_accumulation_steps)

        fsdp_plugin = None
        if world_size_env > 1 and self.cfg.fsdp_sharding_strategy != "NO_SHARD":
            fsdp_plugin = FullyShardedDataParallelPlugin(
                sharding_strategy=self.cfg.fsdp_sharding_strategy,
                auto_wrap_policy="transformer_based_wrap",
                transformer_cls_names_to_wrap=["BlockJ"],
                # lm_head/wte are tied weights -- see _FSDP_IGNORED_MODULES.
                ignored_modules=_FSDP_IGNORED_MODULES,
                state_dict_type="FULL_STATE_DICT",
                # Deliberately NOT using cpu_ram_efficient_loading/
                # sync_module_states: that pair only materializes the real
                # checkpoint on rank 0, then broadcasts to other ranks --
                # but FSDP's sync only broadcasts FSDP-*managed* parameters
                # (confirmed in torch/distributed/fsdp/_init_utils.py:
                # _sync_module_params_and_buffers is only given
                # managed_params, which excludes ignored_modules). Since
                # lm_head/wte are ignored_modules here, they would silently
                # stay uninitialized (meta-device) on every non-master rank.
                # Every rank loads the full real checkpoint independently
                # instead -- more host RAM per node, but guaranteed correct.
                # Required for the freeze-policy config (freeze_*_layer_only):
                # with the default use_orig_params=False, every parameter in
                # one wrapped unit (e.g. a whole BlockJ, or the un-wrapped
                # root containing wpe/ln_f) must share the same requires_grad,
                # which any partial freeze violates. use_orig_params=True lets
                # FSDP mix frozen and trainable parameters within a unit (this
                # is PyTorch's own documented fix for exactly this case).
                use_orig_params=True,
            )

        self.fsdp_plugin = fsdp_plugin
        self.accelerator = Accelerator(
            mixed_precision=self._accelerate_precision(),
            fsdp_plugin=fsdp_plugin,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
        )
        self.device = self.accelerator.device
        self.master = self.accelerator.is_main_process
        self.world_size = self.accelerator.num_processes
        torch.manual_seed(self.cfg.seed + self.accelerator.process_index)

    def _setup_dirs(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.cfg.out_dir, f"{ts}_{self.cfg.mode}")
        if self.cfg.init_from == "resume" and self.cfg.resume_run_dir:
            self.run_dir = self.cfg.resume_run_dir
        if self.master:
            os.makedirs(self.run_dir, exist_ok=True)

    def _setup_wandb(self):
        if not self.master or not self.cfg.wandb_log:
            return
        try:
            import wandb
        except Exception:
            LOG.warning("wandb_unavailable")
            self.cfg.wandb_log = False
            return

        try:
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{self.cfg.wandb_run_name}_{self.cfg.mode}",
                config=vars(self.cfg),
            )
        except Exception as exc:
            LOG.warning("wandb_init_failed", error=str(exc))
            self.cfg.wandb_log = False

    def _setup_data(self):
        if not self.cfg.train_data_paths:
            raise ValueError("No train_data_paths configured")

        missing = []
        for path in self.cfg.train_data_paths + [self.cfg.eval_data_path]:
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                missing.append(path)
        if missing:
            raise FileNotFoundError(
                "Missing or empty training data files: "
                f"{missing}. Prepare data first (e.g. `modal run data/prepare_modal.py`)."
            )

        self.train_bins = self.cfg.train_data_paths
        self.eval_bin = self.cfg.eval_data_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
        LOG.info(
            "data_ready",
            mode=self.cfg.mode,
            train_bins=self.train_bins,
            eval=self.eval_bin,
            sft_masking=self.cfg.is_sft,
        )

    def _build_model(self):
        LOG.info("loading_model", source=self.cfg.init_from, repo=self.cfg.model_name)
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16)

        # Every rank independently loads the full real checkpoint here (see
        # _setup_accelerator for why cpu_ram_efficient_loading/
        # sync_module_states aren't used despite the extra host RAM cost).
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name, trust_remote_code=True, torch_dtype=torch_dtype,
        )
        # from_pretrained's torch_dtype cast isn't always exhaustive for every
        # parameter (e.g. LayerNorm weights can be left in the checkpoint's
        # original dtype) -- FSDP's FlatParamHandle requires every parameter
        # within one wrapped unit to share a dtype, so force a uniform cast
        # here rather than relying on from_pretrained alone.
        self.model = self.model.to(torch_dtype)

        _freeze_layers(self.model, self.cfg)

        if self.cfg.compile_model:
            if self.fsdp_plugin is not None:
                LOG.warning("compile_skipped", reason="torch.compile + FSDP is unsupported/fragile")
            else:
                self.model = torch.compile(self.model)

    def _build_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable,
            lr=self.cfg.learning_rate,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )

    def _prepare_for_training(self):
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        if self.cfg.init_from == "resume" and self.cfg.resume_run_dir:
            meta_path = os.path.join(self.cfg.resume_run_dir, "trainer_state.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self.iter_num = meta.get("iter_num", 0)
                self.best_val = meta.get("best_val_loss", 1e9)
            resume_state_dir = os.path.join(self.cfg.resume_run_dir, "resume_state")
            if os.path.isdir(resume_state_dir):
                self.accelerator.load_state(resume_state_dir)
                LOG.info("resumed_from_checkpoint", path=resume_state_dir, iter=self.iter_num)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _read_memmap(self, path: str) -> np.memmap:
        return np.memmap(path, dtype=np.uint16, mode="r")

    def _sampling_weights(self) -> tuple[float, float]:
        return sampling_weights(
            self.cfg.eng_sampling_weight,
            self.cfg.afr_sampling_weight,
            self.iter_num,
            self.cfg.max_iters,
            self.cfg.use_scheduled_sampling,
        )

    def get_batch(self, split: str):
        if split == "train" and len(self.train_bins) > 1:
            # train_bins is [eng_train_data_path, afr_train_data_path], in that fixed
            # order (see load_config.load_train_config).
            eng_w, _ = self._sampling_weights()
            path = self.train_bins[0] if torch.rand(1).item() < eng_w else self.train_bins[1]
        else:
            path = self.train_bins[0] if split == "train" else self.eval_bin
        data = self._read_memmap(path)
        bs, sl = self.cfg.train_batch_size, self.cfg.block_size
        ix = torch.randint(len(data) - sl - 1, (bs,))
        x = torch.stack([torch.from_numpy(data[i : i + sl].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + sl + 1].astype(np.int64)) for i in ix])

        y = torch.stack([
            apply_label_mask(
                row.clone(), self.cfg.mode,
                user_token=user_token, assistant_token=assistant_token,
                system_token=system_token, mask=MASK,
            )
            for row in y
        ])

        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        return x, y

    # ------------------------------------------------------------------
    # Train / eval
    # ------------------------------------------------------------------

    def _lr(self, it: int) -> float:
        if it < self.cfg.warmup_iters:
            return self.cfg.learning_rate * it / max(1, self.cfg.warmup_iters)
        if it > self.cfg.lr_decay_iters:
            return self.cfg.min_lr
        decay = (it - self.cfg.warmup_iters) / max(1, self.cfg.lr_decay_iters - self.cfg.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
        return self.cfg.min_lr + coeff * (self.cfg.learning_rate - self.cfg.min_lr)

    def _forward_loss(self, x, y):
        raw = self.accelerator.unwrap_model(self.model)
        attention_mask = build_document_causal_mask(x, end_of_text_token)

        if self.cfg.use_cce and HAS_CCE:
            with self.accelerator.autocast():
                out = self.model(
                    input_ids=x, attention_mask=attention_mask,
                    output_hidden_states=True, compute_logits=False,
                )
            hidden = out.hidden_states
            if hidden is None:
                with self.accelerator.autocast():
                    out = self.model(input_ids=x, attention_mask=attention_mask, targets=y)
                ce_loss = out.loss
            else:
                # lm_head is FSDP-ignored (see _FSDP_IGNORED_MODULES), so
                # raw.lm_head.weight is always the full tensor already --
                # no gathering needed.
                weight = raw.lm_head.weight
                ce_loss = linear_cross_entropy(hidden, weight, y, shift=False, ignore_index=MASK)
        else:
            with self.accelerator.autocast():
                out = self.model(input_ids=x, attention_mask=attention_mask, targets=y)
            ce_loss = out.loss

        _, lb_loss = raw.get_expert_utilization()
        if lb_loss is not None:
            return ce_loss + self.cfg.moe_aux_loss_weight * lb_loss
        return ce_loss

    @torch.no_grad()
    def _generate_greedy(self, prompt_ids: torch.Tensor, max_new_tokens: int = 64) -> torch.Tensor:
        """Greedy autoregressive decode using the model's own forward() directly,
        not GenerationMixin.generate(). This custom model class doesn't implement
        the hooks generate() expects (prepare_inputs_for_generation, etc.), which
        under FSDP surfaced as "'weight' must be 2-D" -- calling forward()
        ourselves uses the exact path already proven correct during real
        training steps, at the cost of no KV-cache (recomputes the full
        sequence each step, fine for a short periodic sanity check)."""
        ids = prompt_ids
        for _ in range(max_new_tokens):
            out = self.model(input_ids=ids)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

    @torch.no_grad()
    def _log_sample_generation(self, prompt_ids: torch.Tensor, tag: str = "sample_generation"):
        """Generate continuations for a batch of real prompts and log them
        (master only). Every rank must call the model's forward() collectively
        -- FSDP does a per-layer all-gather, so a single rank calling this
        alone would deadlock waiting on the others."""
        self.model.eval()
        try:
            generated = self._generate_greedy(prompt_ids)
        except Exception as exc:
            if self.master:
                LOG.warning("sample_generation_failed", iter=self.iter_num, error=str(exc))
            self.model.train()
            return
        self.model.train()
        if not self.master:
            return

        n = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)
        header = f" Sample generation @ iter {self.iter_num} ({tag}) "
        print(f"\n{header:=^100}")
        for i in range(n):
            input_text = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=False)
            output_text = self.tokenizer.decode(generated[i, prompt_len:], skip_special_tokens=False)
            print(f"--- sample {i + 1}/{n} ---")
            print(f"[INPUT]  {input_text}")
            print(f"[OUTPUT] {output_text}")
        print("=" * 100 + "\n")

    @torch.no_grad()
    def estimate_loss(self):
        """Every rank evaluates a shard of eval_iters and results are averaged
        via an all-reduce, so all ranks do equal work and stay in lockstep
        (no straggler risk from an eval-only-on-master pattern)."""
        self.model.eval()
        out = {}
        local_iters = max(1, self.cfg.eval_iters // max(1, self.world_size))
        for split in ("train", "val"):
            losses = torch.zeros(local_iters, device=self.device)
            for k in range(local_iters):
                x, y = self.get_batch(split)
                losses[k] = self._forward_loss(x, y)
            local_mean = losses.mean()
            out[split] = self.accelerator.reduce(local_mean, reduction="mean").item()
        self.model.train()
        return out

    def _push_checkpoint_to_hf(self, ckpt_dir: str) -> None:
        if not self.cfg.hf_chkpt_path:
            return
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HF_API_KEY")
        )
        if not token:
            LOG.warning(
                "hf_checkpoint_push_skipped",
                reason="missing HF auth token",
                repo=self.cfg.hf_chkpt_path,
                path=ckpt_dir,
            )
            return

        try:
            from huggingface_hub import HfApi
        except ImportError:
            LOG.warning(
                "hf_checkpoint_push_skipped",
                reason="huggingface_hub not installed",
                repo=self.cfg.hf_chkpt_path,
            )
            return

        api = HfApi()
        try:
            api.create_repo(
                repo_id=self.cfg.hf_chkpt_path, token=token, exist_ok=True, repo_type="model",
            )
        except Exception as exc:
            LOG.info(
                "hf_checkpoint_repo_exists_or_create_failed",
                repo=self.cfg.hf_chkpt_path, reason=str(exc),
            )

        try:
            api.upload_folder(
                folder_path=ckpt_dir,
                repo_id=self.cfg.hf_chkpt_path,
                repo_type="model",
                token=token,
                commit_message=f"checkpoint at iter {self.iter_num}",
            )
            LOG.info("hf_checkpoint_uploaded", repo=self.cfg.hf_chkpt_path, iter=self.iter_num)
        except Exception as exc:
            LOG.error("hf_checkpoint_upload_failed", repo=self.cfg.hf_chkpt_path, reason=str(exc))

    def _save(self):
        # get_state_dict / save_state are collective under FSDP (all-gather
        # across ranks) — every rank must call them, not just master.
        unwrapped = self.accelerator.unwrap_model(self.model)
        state_dict = self.accelerator.get_state_dict(self.model)
        ckpt_dir = os.path.join(self.run_dir, f"ckpt_{self.iter_num}")

        if self.master:
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped.save_pretrained(
                ckpt_dir,
                is_main_process=True,
                save_function=self.accelerator.save,
                state_dict=state_dict,
            )
            with open(os.path.join(self.run_dir, "trainer_state.json"), "w") as f:
                json.dump({"iter_num": self.iter_num, "best_val_loss": self.best_val, "latest_ckpt": ckpt_dir}, f)
            LOG.info("checkpoint_saved", path=ckpt_dir, iter=self.iter_num)

        self.accelerator.save_state(os.path.join(self.run_dir, "resume_state"))
        self.accelerator.wait_for_everyone()

        if self.master:
            self._push_checkpoint_to_hf(ckpt_dir)

    def _maybe_log_wandb(self, losses, lr):
        if not self.cfg.wandb_log or not self.master:
            return
        try:
            import wandb
            wandb.log({"eval/train": losses["train"], "eval/val": losses["val"], "lr": lr}, step=self.iter_num)
        except Exception as exc:
            LOG.warning("wandb_log_failed", error=str(exc))
            self.cfg.wandb_log = False

    def _sample_prompt(self, x: torch.Tensor, num_samples: int = 5) -> torch.Tensor:
        """Real token ids straight from the current batch -- up to num_samples
        rows (fewer if train_batch_size is smaller), passed to the model
        together as one batch."""
        prompt_len = min(32, x.size(1))
        n = min(num_samples, x.size(0))
        return x[:n, :prompt_len]

    def train(self):
        if self.master:
            LOG.info("training_start", mode=self.cfg.mode, world_size=self.world_size)

        x, y = self.get_batch("train")

        # Sanity-check the loaded checkpoint (and FSDP wrapping) before
        # spending any real training time on it.
        self._log_sample_generation(self._sample_prompt(x), tag="startup_sample_generation")

        t0 = time.time()
        last_loss = None

        while self.iter_num <= self.cfg.max_iters:
            lr = self._lr(self.iter_num) if self.cfg.decay_lr else self.cfg.learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            if self.iter_num % self.cfg.eval_interval == 0:
                losses = self.estimate_loss()
                if self.master:
                    LOG.info("eval", iter=self.iter_num, **losses)
                    self._maybe_log_wandb(losses, lr)
                if losses["val"] < self.best_val or self.cfg.always_save_checkpoint:
                    self.best_val = min(self.best_val, losses["val"])
                    if self.iter_num > 0:
                        self._save()

            if self.iter_num == 0 and self.cfg.eval_only:
                break

            if (
                self.iter_num > 0
                and self.cfg.display_model_output_iter > 0
                and self.iter_num % self.cfg.display_model_output_iter == 0
            ):
                self._log_sample_generation(self._sample_prompt(x))

            for _ in range(self.cfg.gradient_accumulation_steps):
                with self.accelerator.accumulate(self.model):
                    loss = self._forward_loss(x, y)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients and self.cfg.grad_clip > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                x, y = self.get_batch("train")
            last_loss = loss

            if self.iter_num % self.cfg.log_interval == 0 and self.master:
                dt = time.time() - t0
                log_kwargs = {"iter": self.iter_num, "loss": last_loss.item(), "ms": dt * 1000}
                if self.cfg.use_scheduled_sampling and len(self.train_bins) > 1:
                    eng_w, afr_w = self._sampling_weights()
                    log_kwargs.update(eng_sampling_weight=eng_w, afr_sampling_weight=afr_w)
                LOG.info("step", **log_kwargs)
                t0 = time.time()

            self.iter_num += 1

        if self.master:
            LOG.info("training_done", iter=self.iter_num)


def main():
    config = load_train_config()
    Trainer(config).train()


if __name__ == "__main__":
    main()
