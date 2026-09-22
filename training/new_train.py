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

import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime

import numpy as np
import structlog
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.constant_tokens import MASK, assistant_token, end_of_text_token, system_token, user_token
from training.label_masking import apply_label_mask
from sabiyarn.hub import register_local_model_code  # noqa: F401  (re-exported for tests)
from training.load_config import TrainConfig, load_train_config, sampling_weights
from training.mfu import compute_mfu, model_flops_per_token, peak_flops_for_current_device
from training.lr_schedule import lr_at
from training.muon import Muon, split_muon_params
from training.data_sampler import MixedBlockSampler, n_blocks_for, read_blocks
from training.curated_eval import CuratedTotals, build_sequence, load_curated_samples, resolve_path
from training.tracking import MlflowTracker, bits_per_byte, build_token_byte_lengths
from training.s3_utils import (
    delete_prefix,
    download_folder,
    find_latest_remote_run_dir,
    is_mutable_checkpoint_file,
    list_immediate_subfolders,
    read_remote_json,
    upload_folder,
    upload_if_absent,
)
from training.training_attention_mask import build_document_block_mask, build_document_causal_mask

LOG = structlog.get_logger()

def _wandb_has_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import netrc

        return netrc.netrc().authenticators("api.wandb.ai") is not None
    except Exception:
        return False


# Config fields never sent to experiment trackers (S3 keys live on TrainConfig).
_SECRET_CFG_MARKERS = ("secret", "access_key", "token", "password")

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

# Val-loss band (see Trainer._should_push_to_hf): push to HF whenever this
# eval's val loss is within this band of what's already there -- i.e. it's
# "commensurate" with the current HF checkpoint, not an anomalous spike.
_HF_PUSH_LOSS_BAND = 0.4


def _find_latest_run_dir(out_dir: str, mode: str) -> str | None:
    """Scans out_dir for existing run directories named `{timestamp}_{mode}`
    (see Trainer._setup_dirs) that have a valid trainer_state.json, and
    returns the most recent one (directory names sort chronologically), or
    None if none exist yet.

    This is how training state (optimizer, iter_num, best_val, and -- since
    the LR and sampling-ratio schedules are pure functions of iter_num, not
    separate stateful objects -- their progress too) auto-resumes regardless
    of platform: out_dir is just a filesystem path, whether it's a Modal
    Volume mount, a vast.ai instance's local disk, or your own machine, so
    no platform-specific resume logic is needed as long as out_dir points at
    a location that actually persists across restarts there.
    """
    if not os.path.isdir(out_dir):
        return None
    suffix = f"_{mode}"
    candidates = []
    for name in os.listdir(out_dir):
        if not name.endswith(suffix):
            continue
        full = os.path.join(out_dir, name)
        if os.path.isfile(os.path.join(full, "trainer_state.json")):
            candidates.append(full)
    if not candidates:
        return None
    return sorted(candidates)[-1]


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
        self._last_hf_push_loss = None  # val loss at the last successful HF push, if any
        self._last_hf_push_iter = 0
        # The separate, less-frequently-updated "best" checkpoint -- see
        # _save's is_new_best handling and train_config.yaml's resume_from.
        # ckpt_best/resume_state_best only get (re)written when an eval's
        # val loss beats this run's own best so far, unlike the "latest"
        # ckpt_{iter_num}/resume_state pair saved on every eval.
        self._best_ckpt_dir = None
        self._best_iter_num = 0
        self._best_sanity_loss = None
        self._best_sanity_batch_hash = None
        # Name (e.g. "ckpt_2400") of the checkpoint folder currently pushed
        # to S3 for this run, if this process has pushed one yet -- lets
        # _push_checkpoint_to_s3 delete it once a newer one uploads, so S3
        # only ever holds the single latest checkpoint (see that method).
        self._last_s3_pushed_ckpt_name = None
        # True until the first S3 push this process actually runs: S3 may
        # already hold ckpt_N folder(s) from before this process started
        # (a prior run, or a manual full-history push via
        # push_checkpoint_to_s3_modal.py) that _last_s3_pushed_ckpt_name
        # above has no way to know about -- clean those up once, the first
        # time, rather than assuming a clean slate.
        self._s3_ckpt_cleanup_pending = True
        self._resume_dir = None  # set by _setup_dirs, used by _build_model/_prepare_for_training
        # Skip checkpointing/HF-push on the very first eval this PROCESS
        # runs -- on a resume, that eval lands at the same iter_num as the
        # checkpoint we just loaded, and saving again would overwrite it
        # before there's been any chance to inspect the resume (see
        # _verify_resume_sanity) and abort if something looks wrong. Cleared
        # after that first eval so every later one saves normally.
        self._suppress_first_save = True
        # Tracks the iter_num as of the last "step" log line, so the MFU/
        # throughput window (see train()) always covers exactly the iterations
        # actually elapsed since then, even for the very first log line (which
        # only covers iter 0 itself, not a full log_interval window).
        self._last_logged_iter = -1
        # MLflow tracking + the pieces bits-per-byte needs (see _target_stats).
        self.tracker = MlflowTracker()
        self._last_ce_loss = None   # CE-only loss (no MoE aux term) of the latest _forward_loss
        self._last_grad_norm = None  # pre-clip global grad norm of the latest optimizer step
        self._token_bytes = None    # id -> UTF-8 byte length lookup, built lazily on device
        # The static reference checkpoint (model.reference_repo), kept on
        # master between _verify_reference_weights and the startup
        # generation comparison that reuses it, then dropped.
        self._ref_model = None
        # Tokens consumed by optimizer steps (not by eval), persisted in trainer_state.json.
        # tokens_seen restarts at 0 on a fresh start; tokens_seen_offset carries the earlier runs' total
        # so "lifetime" tokens (offset + tokens_seen) never resets. By-bin counts (eng / afr) are
        # this rank's consumed micro-batches x world size, i.e. exact when ranks are symmetric.
        self.tokens_seen = 0
        self.tokens_seen_offset = 0
        self.tokens_seen_by_bin: dict[str, int] = {}
        self._tokens_seen_estimated = False
        self._last_train_bin = None
        self._fresh_start_active = False  # set by _setup_dirs: this process is the one-time step-0 restart
        self._fresh_started = False  # persisted in trainer_state.json: this run dir began as a fresh start
        self._setup_accelerator()
        self._resolve_max_iters()
        self._setup_dirs()
        self._backfill_ckpt_best()
        self._setup_wandb()
        self._setup_mlflow()
        self._setup_data()
        self._build_model()
        self._build_optimizer()
        self._prepare_for_training()
        self._verify_resume_sanity()
        self._verify_reference_weights()
        # Qualitative companion to the weight-deviation check above: what the
        # two models actually GENERATE, before a single training step runs.
        # When training.test_run is set, train() runs one eval after this and
        # then stops, without saving or pushing anything (_test_run_eval).
        self._startup_generation_comparison()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _accelerate_precision(self) -> str:
        return {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}.get(self.cfg.dtype, "bf16")

    def _setup_accelerator(self):
        # Keep the effective global batch size (train_batch_size * world_size *
        # grad_accum_steps) invariant to world_size, same as the old DDP path.
        #
        # Read with a fallback, not WORLD_SIZE alone: torchrun sets WORLD_SIZE
        # for every rank it spawns, but a 2026-08-01 run (naijaai workspace,
        # resuming ckpt_7560) read world_size_env<=1 here despite torchrun
        # actually launching 4 processes (confirmed by
        # self.accelerator.num_processes==4 later in that same run) --
        # silently skipping FSDP and falling back to plain DDP, which broke
        # resume_state loading (DDP expects "pytorch_model.bin"; the FSDP
        # checkpoint only has "pytorch_model_fsdp.bin") and then crashed on
        # torch.compile + DDP. LOCAL_WORLD_SIZE is torchrun's redundant
        # signal for exactly this; logging both so a repeat is diagnosable
        # directly instead of inferred after the fact from compile_skipped
        # being absent.
        raw_world_size = os.environ.get("WORLD_SIZE")
        raw_local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
        LOG.info(
            "distributed_env_check",
            WORLD_SIZE=raw_world_size, LOCAL_WORLD_SIZE=raw_local_world_size,
            RANK=os.environ.get("RANK"), LOCAL_RANK=os.environ.get("LOCAL_RANK"),
        )
        world_size_env = int(raw_world_size or raw_local_world_size or 1)
        if world_size_env > 1 and self.cfg.gradient_accumulation_steps % world_size_env == 0:
            self.cfg.gradient_accumulation_steps //= world_size_env
        self.cfg.gradient_accumulation_steps = max(1, self.cfg.gradient_accumulation_steps)

        fsdp_plugin = None
        # DDP is the default (cfg.distributed); FSDP only when explicitly requested. With no
        # fsdp_plugin, Accelerate wraps a multi-process run in plain DDP.
        if world_size_env > 1 and self.cfg.distributed == "fsdp" and self.cfg.fsdp_sharding_strategy != "NO_SHARD":
            # Accelerator.__init__ is documented to set this itself when a
            # fsdp_plugin is passed, but a 2026-08-01 run read WORLD_SIZE=4
            # correctly here (confirmed via distributed_env_check) yet still
            # ended up DDP-wrapped -- root cause not pinned down (couldn't
            # reproduce locally: this sandbox's networking blocks a real
            # multi-process rendezvous to test against). Setting it directly
            # ourselves, before construction, removes any dependency on
            # Accelerate's internal ordering doing this at the right time
            # relative to torch.distributed's own backend selection.
            os.environ["ACCELERATE_USE_FSDP"] = "true"
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
        LOG.info("distributed_type_resolved", distributed_type=str(self.accelerator.state.distributed_type))
        if fsdp_plugin is not None and self.accelerator.state.distributed_type != DistributedType.FSDP:
            # Fail loud and immediate rather than silently training under
            # plain DDP: DDP replicates the full model + Adam optimizer
            # state on every GPU instead of sharding it (what FSDP exists
            # to avoid), and this model's per-GPU memory budget assumes that
            # sharding -- a 2026-08-01 run that silently fell back to DDP
            # this way OOM'd on its very first training step after ~5
            # minutes of setup/S3 upload, burning real GPU time on a run
            # that could never have worked. Better to know in the first
            # second than after paying for that setup cost again.
            raise RuntimeError(
                f"Requested FSDP (fsdp_sharding_strategy={self.cfg.fsdp_sharding_strategy!r}, "
                f"world_size={world_size_env}) but Accelerate resolved distributed_type="
                f"{self.accelerator.state.distributed_type} instead of FSDP -- refusing to silently "
                "continue under plain DDP, which will OOM (DDP doesn't shard model/optimizer state "
                "across GPUs the way FSDP does). See distributed_env_check/distributed_type_resolved "
                "log lines above for the actual env state Accelerate saw."
            )
        self.device = self.accelerator.device
        self.master = self.accelerator.is_main_process
        self.world_size = self.accelerator.num_processes
        torch.manual_seed(self.cfg.seed + self.accelerator.process_index)

    def _tokens_per_step(self) -> int:
        """Tokens consumed by one optimizer step across all ranks (accumulation already / world size)."""
        return (
            self.cfg.train_batch_size * self.cfg.block_size
            * self.cfg.gradient_accumulation_steps * getattr(self, "world_size", 1)
        )

    def _build_samplers(self) -> None:
        """No-repeat epoch samplers (training/data_sampler.py): one global stream per training bin,
        each cut into non-overlapping block_size windows and reshuffled every epoch, identical on
        every rank and split across ranks. Two extra fixed samplers give evals the SAME blocks every
        time (the old evals drew fresh random windows, so eval curves carried sampling noise)."""
        sl, bs = self.cfg.block_size, self.cfg.train_batch_size
        rank = self.accelerator.process_index
        names = [os.path.splitext(os.path.basename(p))[0] for p in self.train_bins]
        blocks = [n_blocks_for(len(self._read_memmap(p)), sl) for p in self.train_bins]
        eval_name = os.path.splitext(os.path.basename(self.eval_bin))[0]
        eval_blocks = n_blocks_for(len(self._read_memmap(self.eval_bin)), sl)
        common = dict(batch_size=bs, world_size=self.world_size, rank=rank)
        self.sampler = MixedBlockSampler(names, blocks, seed=self.cfg.seed, **common)
        self._eval_train_sampler = MixedBlockSampler(names, blocks, seed=self.cfg.seed + 1, **common)
        self._eval_val_sampler = MixedBlockSampler([eval_name], [eval_blocks], seed=self.cfg.seed + 2, **common)
        if self.master:
            LOG.info(
                "data_blocks", block_size=sl,
                bins={n: {"blocks": b, "tokens_b": round(b * sl / 1e9, 3)} for n, b in zip(names, blocks)},
                eval_blocks=eval_blocks,
            )

    def _bin_weights(self) -> tuple[float, ...]:
        if len(self.train_bins) == 1:
            return (1.0,)
        if len(self.train_bins) == 2:
            return self._sampling_weights()
        return tuple([1.0 / len(self.train_bins)] * len(self.train_bins))

    def _resolve_max_iters(self) -> None:
        """If training.max_tokens / optimizer.max_tokens is set, derive max_iters from
        it using the REAL tokens per optimizer step (gradient_accumulation_steps has
        already been divided by world size at this point), so the LR schedule and the
        token budget can't drift apart through arithmetic slips."""
        if self.cfg.max_tokens <= 0:
            return
        world = getattr(self, "world_size", 1)
        tokens_per_step = self._tokens_per_step()
        self.cfg.max_iters = max(1, math.ceil(self.cfg.max_tokens / tokens_per_step))
        if self.master:
            LOG.info(
                "max_iters_from_token_budget", max_tokens=self.cfg.max_tokens,
                tokens_per_step=tokens_per_step, max_iters=self.cfg.max_iters,
                warmup_iters=self.cfg.warmup_iters, scheduler=self.cfg.scheduler,
            )

    def _load_lm(self, path, torch_dtype=None):
        """Load a SabiYarn causal LM: model code from this repo (model_code: local) or from the
        Hub next to the weights (model_code: hub); weights always come from `path`."""
        kwargs = {} if torch_dtype is None else {"torch_dtype": torch_dtype}
        if self.cfg.model_code == "local":
            from sabiyarn.model.modeling import GPTJXMoEForCausalLM

            register_local_model_code()
            return GPTJXMoEForCausalLM.from_pretrained(path, **kwargs)
        return AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, **kwargs)

    def _apply_model_options(self) -> None:
        """Apply yaml model options that are not stored in checkpoints."""
        sparse = self.cfg.moe_dispatch == "sparse"
        n = 0
        for module in self.model.modules():
            if hasattr(module, "sparse_dispatch"):
                module.sparse_dispatch = sparse
                n += 1
        self.model.config.moe_sparse_dispatch = sparse  # keeps mfu.py's FLOP accounting in step
        if self.master:
            LOG.info("model_options", model_code=self.cfg.model_code, moe_dispatch=self.cfg.moe_dispatch,
                     moe_layers=n, attention_impl=self.cfg.attention_impl)

    def _doc_attention_mask(self, x):
        if self.cfg.attention_impl == "flex":
            return build_document_block_mask(x, end_of_text_token)
        return build_document_causal_mask(x, end_of_text_token)

    def _param_torch_dtype(self):
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            self.cfg.param_dtype, torch.float32
        )

    def _setup_dirs(self):
        # Training state (optimizer, iter_num, best_val, schedule progress --
        # see _prepare_for_training) always auto-resumes from the latest
        # checkpoint when one exists, regardless of init_from; init_from only
        # controls where MODEL WEIGHTS come from (see _build_model).
        # resume_run_dir, if set, is an explicit override that wins over
        # everything below -- otherwise the most recent state is found
        # automatically, comparing local out_dir against S3 (see
        # _resolve_resume_dir_local_or_s3), which is what makes this work
        # across Modal container restarts *and* across switching between
        # different Modal accounts/volumes with the same S3 bucket as the
        # shared source of truth.
        self._resume_dir = None
        if self.cfg.resume_run_dir:
            if os.path.isfile(os.path.join(self.cfg.resume_run_dir, "trainer_state.json")):
                self._resume_dir = self.cfg.resume_run_dir
            else:
                LOG.warning("resume_run_dir_has_no_checkpoint", path=self.cfg.resume_run_dir)
        else:
            self._resume_dir = self._resolve_resume_dir_local_or_s3()

        if self._resume_dir and self.cfg.fresh_start:
            resumed_meta_path = os.path.join(self._resume_dir, "trainer_state.json")
            resumed_meta = {}
            if os.path.isfile(resumed_meta_path):
                with open(resumed_meta_path, "r") as f:
                    resumed_meta = json.load(f)
            if resumed_meta.get("fresh_started"):
                # Already the run dir a previous fresh start created: resume it normally.
                self._fresh_started = True
                LOG.info("fresh_start_already_applied", path=self._resume_dir)
            else:
                self._fresh_start_active = self._fresh_started = True

        if self._resume_dir and self._fresh_start_active:
            # Weights/optimizer come from _resume_dir, but everything new is written to a
            # NEW run dir so the old run's checkpoints and iteration numbers are untouched.
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(self.cfg.out_dir, f"{ts}_{self.cfg.mode}")
            LOG.info("fresh_start_new_run_dir", loading_from=self._resume_dir, writing_to=self.run_dir)
        elif self._resume_dir:
            self.run_dir = self._resume_dir
            LOG.info("found_existing_checkpoint_dir", path=self.run_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(self.cfg.out_dir, f"{ts}_{self.cfg.mode}")

        if self.master:
            os.makedirs(self.run_dir, exist_ok=True)

    def _backfill_ckpt_best(self) -> None:
        """One-time migration for a run_dir saved before ckpt_best existed
        (e.g. ckpt_7560, saved 2026-07-30): seeds ckpt_best/resume_state_best
        from the current latest_ckpt/resume_state and records that in
        trainer_state.json, so both are immediately valid, independent
        resume targets (see resume_from) from this point forward -- rather
        than ckpt_best silently staying absent until the next real
        improvement. No-ops once best_ckpt is already recorded. Pushes the
        seeded ckpt_best to S3 too, so a fresh account/volume resuming from
        S3 sees it immediately, not just whichever account happens to run
        the backfill locally.

        Master does the actual I/O; every rank still calls this (and hits
        the barrier at the end) so the rest of __init__ can safely assume
        trainer_state.json is already backfilled on every rank by the time
        _build_model/_prepare_for_training read it.
        """
        if self.cfg.test_run:
            # test_run promises nothing is written or pushed anywhere -- and
            # this backfill both copies ckpt_best/resume_state_best locally
            # AND pushes them to S3. Skipped entirely; it's a one-time
            # migration that the next real run will do.
            self.accelerator.wait_for_everyone()
            return
        if self.master and self._resume_dir and not self._fresh_start_active:
            meta_path = os.path.join(self._resume_dir, "trainer_state.json")
            if os.path.isfile(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                latest_ckpt = meta.get("latest_ckpt")
                if not meta.get("best_ckpt") and latest_ckpt and os.path.isdir(latest_ckpt):
                    best_ckpt_dir = os.path.join(self._resume_dir, "ckpt_best")
                    shutil.copytree(latest_ckpt, best_ckpt_dir, dirs_exist_ok=True)

                    resume_state_dir = os.path.join(self._resume_dir, "resume_state")
                    resume_state_best_dir = os.path.join(self._resume_dir, "resume_state_best")
                    if os.path.isdir(resume_state_dir):
                        shutil.copytree(resume_state_dir, resume_state_best_dir, dirs_exist_ok=True)

                    meta["best_ckpt"] = best_ckpt_dir
                    meta["best_iter_num"] = meta.get("iter_num", 0)
                    meta["best_sanity_loss"] = meta.get("sanity_loss")
                    meta["best_sanity_batch_hash"] = meta.get("sanity_batch_hash")
                    with open(meta_path, "w") as f:
                        json.dump(meta, f)
                    LOG.info(
                        "ckpt_best_backfilled", path=best_ckpt_dir, iter=meta["best_iter_num"],
                        note="seeded ckpt_best from latest_ckpt -- this run_dir predates best-checkpoint tracking",
                    )
                    # self.run_dir == self._resume_dir already (set by
                    # _setup_dirs above), which _push_checkpoint_to_s3 reads.
                    # iter_num isn't resumed until _prepare_for_training runs
                    # later -- set it here too, just so this push's log line
                    # reports the real iter instead of the init default (0).
                    self.iter_num = meta.get("iter_num", 0)
                    self._push_checkpoint_to_s3(latest_ckpt, is_new_best=True)

        self.accelerator.wait_for_everyone()

    def _run_name_timestamp(self, run_name: str) -> str:
        """Strips the trailing `_{mode}` off a run dir name (`{timestamp}_{mode}`),
        leaving just the sortable creation timestamp."""
        suffix = f"_{self.cfg.mode}"
        return run_name[: -len(suffix)] if run_name.endswith(suffix) else run_name

    def _local_run_recency(self, local_run_dir: str | None) -> tuple[int, str]:
        """(iter_num, creation_timestamp) for the local candidate, or the
        lowest-possible sentinel if there isn't one -- so it always loses a
        comparison against any real S3 checkpoint."""
        if not local_run_dir:
            return (-1, "")
        meta_path = os.path.join(local_run_dir, "trainer_state.json")
        if not os.path.isfile(meta_path):
            return (-1, "")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        name = os.path.basename(local_run_dir.rstrip("/"))
        return (int(meta.get("iter_num", 0)), self._run_name_timestamp(name))

    def _remote_run_recency(self, remote_run_prefix: str, s3_kwargs: dict) -> tuple[int, str]:
        """(iter_num, creation_timestamp) for the S3 candidate, read from
        just its trainer_state.json -- no need to download the (potentially
        large) rest of the checkpoint just to compare recency."""
        try:
            meta = read_remote_json(f"{remote_run_prefix}trainer_state.json", **s3_kwargs)
        except Exception as e:
            LOG.warning("s3_trainer_state_read_failed", path=remote_run_prefix, error=str(e))
            meta = None
        if meta is None:
            return (-1, "")
        name = remote_run_prefix.rstrip("/").rsplit("/", 1)[-1]
        return (int(meta.get("iter_num", 0)), self._run_name_timestamp(name))

    def _resolve_resume_dir_local_or_s3(self) -> str | None:
        """Auto-discovery when resume_run_dir isn't explicitly set: compares
        the latest local checkpoint against the latest one pushed to S3 (via
        training/push_checkpoint_to_s3_modal.py) and uses whichever is more
        recent -- by iter_num first, then by creation timestamp as a
        tiebreaker. This is what lets training continue correctly no matter
        which Modal account/volume you're currently running on, as long as
        checkpoints get pushed to the same S3 bucket: an account with a
        stale or empty local out_dir still picks up the real latest state
        from S3 instead of silently restarting from scratch or resuming an
        outdated local checkpoint.

        cfg.force_download_from_s3 skips the comparison entirely and always
        uses S3 when it has anything, regardless of what's local -- for
        cases where you know local is wrong/irrelevant and just want a clean
        pull from the shared source of truth.
        """
        local_run_dir = _find_latest_run_dir(self.cfg.out_dir, self.cfg.mode)

        s3_ready = bool(
            self.cfg.s3_bucket and self.cfg.s3_endpoint and self.cfg.s3_access_key and self.cfg.s3_secret_key
        )
        if not s3_ready:
            return local_run_dir

        s3_kwargs = dict(
            bucket=self.cfg.s3_bucket, endpoint=self.cfg.s3_endpoint,
            access_key=self.cfg.s3_access_key, secret_key=self.cfg.s3_secret_key,
        )
        remote_root = f"checkpoints/{os.path.basename(self.cfg.out_dir.rstrip('/'))}"
        try:
            remote_run_prefix = find_latest_remote_run_dir(
                remote_root, self.cfg.mode, prefix=self.cfg.s3_prefix, **s3_kwargs,
            )
        except Exception as e:
            LOG.warning("s3_latest_run_lookup_failed", error=str(e))
            remote_run_prefix = None

        if remote_run_prefix is None:
            return local_run_dir

        use_s3 = bool(self.cfg.force_download_from_s3)
        if not use_s3:
            remote_recency = self._remote_run_recency(remote_run_prefix, s3_kwargs)
            local_recency = self._local_run_recency(local_run_dir)
            use_s3 = remote_recency > local_recency
            LOG.info(
                "resume_source_comparison",
                local_dir=local_run_dir, local_iter=local_recency[0], local_ts=local_recency[1],
                remote_dir=remote_run_prefix, remote_iter=remote_recency[0], remote_ts=remote_recency[1],
                chosen="s3" if use_s3 else "local",
            )

        if not use_s3:
            return local_run_dir

        return self._download_remote_run_dir(remote_run_prefix, s3_kwargs)

    def _download_remote_run_dir(self, remote_run_prefix: str, s3_kwargs: dict) -> str | None:
        """Downloads a full remote run dir (weights + resume_state +
        trainer_state.json, as pushed by training/push_checkpoint_to_s3_modal.py
        with --folder <run_dir_name>) into a local directory under out_dir
        with the same run-dir name, so the rest of the resume machinery
        (_resolve_resume_weights_path, _prepare_for_training's
        accelerator.load_state and its incompatible-checkpoint graceful
        degradation) can treat it exactly like a locally-found run_dir -- no
        separate code path needed there.

        Only the master rank downloads, to avoid every rank racing to write
        the same local files when ranks share a filesystem (the single-node
        multi-GPU case, which is what out_dir being a shared path already
        assumes for local-checkpoint resume too). Other ranks wait at the
        barrier below, then check the shared filesystem directly -- more
        reliable than trying to propagate success/failure through a
        Python-local variable that only master actually set.
        """
        run_name = remote_run_prefix.rstrip("/").rsplit("/", 1)[-1]
        local_run_dir = os.path.join(self.cfg.out_dir, run_name)

        if self.master:
            os.makedirs(local_run_dir, exist_ok=True)
            try:
                # remote_run_prefix already folds in cfg.s3_prefix (it came
                # straight from find_latest_remote_run_dir's listing), so
                # pass prefix="" here to avoid applying it a second time.
                # force_redownload_paths: trainer_state.json/resume_state
                # are rewritten in place remotely on every push -- never
                # trust a local copy already sitting under this run_dir
                # name for those, even if one exists from an earlier run.
                download_folder(
                    remote_run_prefix, local_run_dir, prefix="",
                    force_redownload_paths=is_mutable_checkpoint_file, **s3_kwargs,
                )

                meta_path = os.path.join(local_run_dir, "trainer_state.json")
                if os.path.isfile(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    latest_ckpt = meta.get("latest_ckpt")
                    best_ckpt = meta.get("best_ckpt")
                    if latest_ckpt or best_ckpt:
                        # latest_ckpt/best_ckpt in the downloaded json are
                        # absolute paths from wherever they were originally
                        # saved (e.g. a different machine/volume) -- repoint
                        # them at this machine's actual local copies.
                        if latest_ckpt:
                            meta["latest_ckpt"] = os.path.join(local_run_dir, os.path.basename(latest_ckpt.rstrip("/")))
                        if best_ckpt:
                            meta["best_ckpt"] = os.path.join(local_run_dir, os.path.basename(best_ckpt.rstrip("/")))
                        with open(meta_path, "w") as f:
                            json.dump(meta, f)
                LOG.info("resume_state_downloaded_from_s3", remote=remote_run_prefix, local=local_run_dir)
            except Exception as e:
                LOG.warning("resume_state_s3_download_failed", remote=remote_run_prefix, error=str(e))

        self.accelerator.wait_for_everyone()

        if os.path.isfile(os.path.join(local_run_dir, "trainer_state.json")):
            return local_run_dir
        return None

    def _setup_wandb(self):
        if not self.master or not self.cfg.wandb_log:
            return
        try:
            import wandb
        except Exception:
            LOG.warning("wandb_unavailable")
            self.cfg.wandb_log = False
            return

        if not _wandb_has_credentials():
            # wandb.init() with no key opens an interactive login prompt on a real terminal,
            # blocking the run at startup (Modal never hit this: no stdin, it just errored).
            LOG.warning("wandb_skipped", reason="no WANDB_API_KEY or ~/.netrc entry; set one or wandb.log: false")
            self.cfg.wandb_log = False
            return

        try:
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{self.cfg.wandb_run_name}_{self.cfg.mode}",
                # Never send credentials (S3 keys live on TrainConfig) to a third party.
                config={k: v for k, v in vars(self.cfg).items() if not any(t in k.lower() for t in _SECRET_CFG_MARKERS)},
            )
        except Exception as exc:
            LOG.warning("wandb_init_failed", error=str(exc))
            self.cfg.wandb_log = False

    def _setup_mlflow(self):
        if not self.master or not self.cfg.mlflow_log:
            return
        run_name = self.cfg.mlflow_run_name or f"{self.cfg.wandb_run_name}_{self.cfg.mode}"
        config_path = os.environ.get("TRAIN_CONFIG_PATH") or os.path.join(os.path.dirname(__file__), "train_config.yaml")
        self.tracker.start(
            tracking_uri=self.cfg.mlflow_tracking_uri,
            experiment_name=self.cfg.mlflow_experiment,
            run_name=run_name,
            run_id=self.cfg.mlflow_run_id,
            params={
                **{k: v for k, v in vars(self.cfg).items() if not any(t in k.lower() for t in _SECRET_CFG_MARKERS)},
                "world_size": self.world_size,
            },
            tags={"mode": self.cfg.mode},
            log_system_metrics=self.cfg.mlflow_log_system_metrics,
            artifacts=[config_path],
        )
        if self.cfg.mlflow_ui:
            self.tracker.start_ui(self.cfg.mlflow_ui_host, self.cfg.mlflow_ui_port)

    def _byte_table(self) -> torch.Tensor:
        if self._token_bytes is None:
            special = set(self.tokenizer.all_special_ids) | set(self.tokenizer.added_tokens_decoder.keys())
            table = build_token_byte_lengths(self.tokenizer.get_vocab(), special)
            self._token_bytes = torch.tensor(table, dtype=torch.int64, device=self.device)
        return self._token_bytes

    def _target_stats(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(supervised token count, bytes those tokens decode to) for a target
        batch -- the two denominators bits-per-byte needs. Masked positions
        (MASK) count for neither; special tokens count 0 bytes."""
        table = self._byte_table()
        valid = y != MASK
        ids = y.clamp(min=0, max=table.numel() - 1)  # ids past the tokenizer vocab hit the trailing 0 slot
        return valid.sum(), (table[ids] * valid).sum()

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
                f"{missing}. Download the bins first: `python -m data.prefetch_bins --mode {self.cfg.mode} --write-env` "
                "(bare box / vast.ai), or `modal run data/prepare_modal.py` to build them (Modal)."
            )

        self.train_bins = self.cfg.train_data_paths
        self.eval_bin = self.cfg.eval_data_path
        # trust_remote_code=True: see training/constant_tokens.py (interactive prompt on a tty).
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name, trust_remote_code=True)
        self._build_samplers()
        LOG.info(
            "data_ready",
            mode=self.cfg.mode,
            train_bins=self.train_bins,
            eval=self.eval_bin,
            sft_masking=self.cfg.is_sft,
        )

    def _resolve_resume_weights_path(self) -> str | None:
        """The local model-weights directory to load from when
        init_from=="resume": latest_ckpt or best_ckpt (per cfg.resume_from)
        recorded in the resumed run's trainer_state.json (see _setup_dirs
        for how _resume_dir itself is found), if that checkpoint has
        actually been saved there yet."""
        if not self._resume_dir:
            return None
        meta_path = os.path.join(self._resume_dir, "trainer_state.json")
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path, "r") as f:
            meta = json.load(f)
        ckpt_path = meta.get("best_ckpt") if self.cfg.resume_from == "best" else None
        if self.cfg.resume_from == "best" and not ckpt_path:
            LOG.warning(
                "resume_from_best_unavailable", path=self._resume_dir,
                reason="no best_ckpt recorded yet -- falling back to latest_ckpt",
            )
        if not ckpt_path:
            ckpt_path = meta.get("latest_ckpt")
        if ckpt_path and os.path.isdir(ckpt_path):
            return ckpt_path
        return None

    def _build_model(self):
        # Parameters are held in cfg.param_dtype (fp32 master weights by default); bf16
        # autocast (cfg.dtype) still runs the math in bf16. See TrainConfig.param_dtype.
        torch_dtype = self._param_torch_dtype()

        # init_from controls MODEL WEIGHTS only -- optimizer/iter_num/best_val
        # always auto-resume separately regardless of this setting (see
        # _prepare_for_training).
        #
        # The base architecture always comes from the HF Hub (model.repo_name)
        # -- this guarantees a complete, canonical set of config/generation/
        # tokenizer files regardless of what a local checkpoint directory
        # happens to contain. init_from=="resume" then overlays that
        # architecture's weights with the last local checkpoint's state dict
        # (see _load_checkpoint_weights) rather than instantiating
        # from_pretrained directly against the local checkpoint dir.
        resume_weights = self._resolve_resume_weights_path() if self.cfg.init_from == "resume" else None
        if self.cfg.init_from == "resume" and resume_weights is None:
            LOG.warning(
                "resume_requested_but_no_checkpoint_weights_found",
                out_dir=self.cfg.out_dir, mode=self.cfg.mode,
                fallback=f"loading model_name={self.cfg.model_name!r} from HF instead",
            )
        load_desc = "hf base + local checkpoint weights (resume)" if resume_weights else self.cfg.init_from

        LOG.info("loading_model", source=load_desc, repo=self.cfg.model_name)

        # Every rank independently loads the full real checkpoint here (see
        # _setup_accelerator for why cpu_ram_efficient_loading/
        # sync_module_states aren't used despite the extra host RAM cost).
        self.model = self._load_lm(self.cfg.model_name, torch_dtype=torch_dtype)
        self._apply_model_options()
        # from_pretrained's torch_dtype cast isn't always exhaustive for every
        # parameter (e.g. LayerNorm weights can be left in the checkpoint's
        # original dtype) -- FSDP's FlatParamHandle requires every parameter
        # within one wrapped unit to share a dtype, so force a uniform cast
        # here rather than relying on from_pretrained alone.
        self.model = self.model.to(torch_dtype)

        if resume_weights is not None:
            self._load_checkpoint_weights(resume_weights, torch_dtype)

        _freeze_layers(self.model, self.cfg)

        # Computed from the real (post from_pretrained) config -- expert_per_layer
        # is only populated once the HF loading path runs (_prepare_config), so this
        # must happen after from_pretrained above, not from train_config.yaml alone.
        self._flops_per_token = model_flops_per_token(self.model.config, self.cfg.block_size)
        self._peak_flops_per_gpu = peak_flops_for_current_device()
        if self.master:
            LOG.info(
                "mfu_setup",
                flops_per_token=self._flops_per_token,
                gpu=torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu/mps",
                peak_tflops_per_gpu=(self._peak_flops_per_gpu / 1e12 if self._peak_flops_per_gpu else None),
                note=(
                    "flops_per_token counts EVERY expert per MoE layer (not just "
                    "num_experts_per_tok) since MoE.forward computes all experts "
                    "densely before top-k gather -- see training/mfu.py"
                    if getattr(self.model.config, "use_moe", False) else None
                ),
            )
            if self._peak_flops_per_gpu is None and torch.cuda.is_available():
                LOG.warning(
                    "mfu_peak_flops_unknown",
                    gpu=torch.cuda.get_device_name(torch.cuda.current_device()),
                    action="add this GPU to _GPU_PEAK_TFLOPS in training/mfu.py to get an mfu %; "
                           "tflops_per_gpu will still be logged",
                )

        if self.cfg.compile_model:
            if self.accelerator.num_processes > 1:
                # Confirmed fragile under BOTH wrapping strategies, not just
                # FSDP: a 2026-08-01 run that fell back to DDP (see
                # _setup_accelerator) crashed torch._dynamo's DDPOptimizer
                # graph-splitting pass with AttributeError: 'int' object has
                # no attribute 'meta' on the very first backward pass.
                LOG.warning(
                    "compile_skipped",
                    reason="torch.compile is unsupported/fragile under multi-process distributed "
                           "training (FSDP or DDP) for this model",
                )
            else:
                self.model = torch.compile(self.model)

    def _load_checkpoint_weights(self, ckpt_dir: str, torch_dtype) -> None:
        """Overlays self.model's weights (already built from the HF Hub
        architecture) with a local checkpoint's state dict. Loads the
        checkpoint via from_pretrained (the same trust_remote_code path
        already proven to work) purely to obtain its state dict, then
        discards that temporary model -- avoids hand-parsing the checkpoint's
        safetensors/bin shards directly."""
        ckpt_model = self._load_lm(ckpt_dir, torch_dtype=torch_dtype)
        self.model.load_state_dict(ckpt_model.state_dict(), strict=True)
        del ckpt_model
        LOG.info("resume_checkpoint_weights_loaded", path=ckpt_dir)

    def _build_optimizer(self):
        if self.cfg.optimizer_name == "muon":
            if self.fsdp_plugin is not None:
                raise ValueError(
                    "optimizer.name: muon needs whole weight matrices, which FSDP shards away. Set "
                    "ddp.enabled: true (the default) or use a single GPU, or set "
                    "optimizer.name: adamw."
                )
            n_embd = self.model.config.n_embd
            muon, adam_2d, adam_1d = split_muon_params(self.model, n_embd)
            wd = self.cfg.weight_decay
            self.optimizer = Muon(
                [
                    {"params": muon, "use_muon": True, "weight_decay": wd},
                    {"params": adam_2d, "use_muon": False, "weight_decay": wd},
                    {"params": adam_1d, "use_muon": False, "weight_decay": 0.0},
                ],
                n_embd=n_embd, lr=self.cfg.learning_rate, weight_decay=wd,
                momentum=self.cfg.muon_momentum, ns_steps=self.cfg.muon_ns_steps, rms=self.cfg.muon_rms,
                betas=(self.cfg.beta1, self.cfg.beta2),
            )
            if self.master:
                LOG.info(
                    "optimizer_muon", muon_tensors=len(muon), muon_params=sum(p.numel() for p in muon),
                    adamw_2d_tensors=len(adam_2d), adamw_1d_tensors=len(adam_1d),
                )
            return
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable,
            lr=self.cfg.learning_rate,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )

    def _prepare_for_training(self):
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # Always attempt to resume optimizer state / iter_num / best_val from
        # the latest checkpoint (self._resume_dir, found in _setup_dirs),
        # regardless of init_from -- init_from only controls where MODEL
        # WEIGHTS come from (see _build_model). The LR schedule (_lr) and the
        # dynamic eng/afr sampling-ratio schedule (sampling_weights) are both
        # pure functions of iter_num, not separate stateful objects, so
        # restoring iter_num alone is what continues them correctly.
        if self._resume_dir:
            meta_path = os.path.join(self._resume_dir, "trainer_state.json")
            resume_from_best = False
            restore_optimizer_state = self.cfg.resume_optimizer_state
            if os.path.isfile(meta_path):
                # iter_num/best_val/hf-push tracking are plain facts recorded
                # at save time -- resume them unconditionally whenever
                # trainer_state.json is present, regardless of whether the
                # optimizer's exact momentum can also be restored below.
                # Previously these were only set inside the resume_state
                # try/except's success branch, so ANY resume_state issue
                # (missing, incompatible, or deliberately omitted -- e.g. to
                # warm-start weights from one checkpoint with a fresh
                # optimizer rather than reusing another checkpoint's
                # mismatched momentum) silently threw away iter_num too,
                # even though it was sitting right there in the same file.
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                saved_optimizer = meta.get("optimizer_name", "adamw")
                if saved_optimizer != self.cfg.optimizer_name:
                    # Moments/momentum from a different optimizer can't be reused.
                    restore_optimizer_state = False
                    LOG.info(
                        "optimizer_state_not_restored", saved=saved_optimizer, current=self.cfg.optimizer_name,
                        note="optimizer changed since this checkpoint; starting it fresh",
                    )
                resume_from_best = self.cfg.resume_from == "best" and bool(meta.get("best_ckpt"))
                # "best" rewinds iter_num to wherever that best eval
                # happened -- the LR/sampling schedule is a pure function
                # of iter_num, so this genuinely resumes AT that earlier
                # point, not just its weights with today's schedule value.
                self.iter_num = (
                    meta.get("best_iter_num", meta.get("iter_num", 0))
                    if resume_from_best else meta.get("iter_num", 0)
                )
                # Token accounting: exact if the checkpoint recorded it, else estimated from iter_num
                # x this run's tokens/step (older checkpoints; wrong if the batch size changed).
                if meta.get("tokens_seen") is not None and not resume_from_best:
                    self.tokens_seen = int(meta["tokens_seen"])
                    self.tokens_seen_by_bin = {k: int(v) for k, v in meta.get("tokens_seen_by_bin", {}).items()}
                else:
                    self.tokens_seen = int(self.iter_num * meta.get("tokens_per_step", self._tokens_per_step()))
                    self.tokens_seen_by_bin = {}
                    self._tokens_seen_estimated = True
                self.tokens_seen_offset = int(meta.get("tokens_seen_offset", 0))
                # Data position: continue mid-epoch instead of replaying batches already trained on.
                # Not restored for a fresh start (new run, new epoch 0), for resume-from-best (its
                # data position wasn't recorded), or for checkpoints from before this sampler existed.
                if meta.get("sampler") and not resume_from_best and not self._fresh_start_active:
                    skipped = self.sampler.load_state_dict(meta["sampler"])
                    LOG.info("sampler_state_restored", epochs=self.sampler.epochs_done(), reset_bins=skipped)
                else:
                    LOG.info(
                        "sampler_state_fresh",
                        reason="fresh_start" if self._fresh_start_active else (
                            "resume_from_best" if resume_from_best else "checkpoint has no sampler state"),
                    )
                self.best_val = meta.get("best_val_loss", 1e9)
                self._best_ckpt_dir = meta.get("best_ckpt")
                self._best_iter_num = meta.get("best_iter_num", self.iter_num)
                self._best_sanity_loss = meta.get("best_sanity_loss")
                self._best_sanity_batch_hash = meta.get("best_sanity_batch_hash")
                self._last_hf_push_loss = meta.get("last_hf_push_loss")
                self._last_hf_push_iter = meta.get("last_hf_push_iter", 0)

            resume_state_dir = os.path.join(
                self._resume_dir, "resume_state_best" if resume_from_best else "resume_state",
            )
            if not restore_optimizer_state:
                LOG.info(
                    "resume_state_skipped", path=resume_state_dir,
                    note="optimizer/model restore from resume_state disabled; weights come from init_from, optimizer starts fresh",
                )
            elif os.path.isdir(resume_state_dir):
                # Bisects WHERE a post-resume sanity-loss gap gets introduced:
                # accelerator.load_state() restores BOTH the FSDP model state
                # AND the optimizer state together from resume_state (it's
                # the one step common to every instance of the "loss jumps
                # after resume" pattern chased across this whole session --
                # always last, always after accelerator.prepare() has already
                # FSDP-wrapped the model). Comparing sanity_loss right before
                # vs. right after this specific call tells us definitively
                # whether THIS is where the gap appears (pointing at a real
                # bug in the FSDP full-state-dict save/restore round trip --
                # note the FSDP.state_dict_type()/set_state_dict_type()
                # deprecation warnings this exact call triggers every run)
                # or whether it's already present beforehand (pointing at
                # FSDP wrapping/mixed-precision itself, or the initial
                # weights, instead).
                pre_load_state_sanity_loss = self._sanity_loss()
                # accelerator.save_state/load_state captures optimizer state
                # (and RNG generator state) for whatever was passed to
                # accelerator.prepare() -- self.optimizer here. This can fail
                # if the discovered run_dir belongs to an incompatible run
                # (e.g. a leftover checkpoint from an earlier smoke test with
                # different freeze_*/model settings, so the optimizer's
                # trainable-param groups don't line up) -- degrade to a fresh
                # optimizer rather than crashing the whole launch, since a
                # stale directory under out_dir shouldn't be able to take
                # down a real run. iter_num/best_val above are unaffected
                # either way.
                try:
                    self.accelerator.load_state(resume_state_dir)
                except Exception as e:
                    LOG.warning(
                        "resume_state_incompatible", path=resume_state_dir, error=str(e),
                        action="continuing with a fresh optimizer state; iter_num/best_val still resumed from trainer_state.json",
                    )
                else:
                    post_load_state_sanity_loss = self._sanity_loss()
                    load_state_delta = post_load_state_sanity_loss - pre_load_state_sanity_loss
                    log_fn = LOG.warning if abs(load_state_delta) > 0.1 else LOG.info
                    log_fn(
                        "load_state_sanity_check",
                        pre_load_state_sanity_loss=pre_load_state_sanity_loss,
                        post_load_state_sanity_loss=post_load_state_sanity_loss,
                        load_state_delta=load_state_delta,
                        interpretation=(
                            "large |load_state_delta| -> accelerator.load_state()'s model restoration "
                            "(not the optimizer, not the initial weight load, not FSDP wrapping itself) "
                            "is where the gap gets introduced -- a real bug in the FSDP full-state-dict "
                            "save/restore round trip. small |load_state_delta| -> the gap (if any, see "
                            "resume_sanity_check below) was already present before this call, so look at "
                            "FSDP wrapping/mixed-precision or the initial weight source instead."
                        ),
                    )
                    LOG.info(
                        "resumed_training_state", path=resume_state_dir, iter=self.iter_num,
                        source="best" if resume_from_best else "latest",
                    )

        if self._fresh_start_active:
            # Weights and (if compatible) optimizer state are loaded; everything that
            # tracks PROGRESS restarts: step counter (so warmup / LR schedule / scheduled
            # sampling begin at 0) and the best-val / HF-push bookkeeping, which would
            # otherwise compare a new run's high early losses against the old run's best.
            LOG.warning(
                "fresh_start", previous_iter_num=self.iter_num, previous_best_val=self.best_val,
                note="training.fresh_start is set: iter_num=0, best_val/HF-push tracking reset, new run dir",
            )
            self.tokens_seen_offset += self.tokens_seen  # earlier runs' tokens live on as the lifetime offset
            self.tokens_seen = 0
            self.tokens_seen_by_bin = {}
            self.iter_num = 0
            self.best_val = 1e9
            self._best_ckpt_dir = None
            self._best_iter_num = 0
            self._best_sanity_loss = None
            self._best_sanity_batch_hash = None
            self._last_hf_push_loss = None
            self._last_hf_push_iter = 0

        # Manual last-resort override -- forces iter_num regardless of
        # whatever the block above found (or didn't find at all), for when
        # a checkpoint's weights made it to HF but its matching training
        # state never made it to S3. Applies unconditionally, not just as a
        # fallback, per training.last_step's contract; see its definition
        # in load_config.py for why this must be manually cleared afterward.
        if self.cfg.last_step is not None:
            LOG.warning(
                "last_step_override", previous_iter_num=self.iter_num, forced_iter_num=self.cfg.last_step,
                action="training.last_step is set in train_config.yaml -- clear it back to blank once "
                       "checkpointing is confirmed working again, or every future restart will keep "
                       "resetting iter_num to this same value and discard real progress made since.",
            )
            self.iter_num = self.cfg.last_step

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _read_memmap(self, path: str) -> np.memmap:
        return np.memmap(path, dtype=np.uint16, mode="r")

    def _sanity_indices_and_data(self) -> tuple[np.memmap, np.ndarray]:
        """The (data, indices) pair _sanity_batch/_sanity_batch_hash both
        draw from -- factored out so the hash and the actual batch are
        guaranteed to be computed from the exact same bytes."""
        rng = np.random.default_rng(1234567)
        bs, sl = self.cfg.train_batch_size, self.cfg.block_size
        data = self._read_memmap(self.eval_bin)
        ix = rng.integers(0, len(data) - sl - 1, size=bs)
        return data, ix

    def _sanity_batch(self):
        """A FIXED, deterministic batch -- drawn via a fixed numpy seed
        independent of the live torch RNG stream (which accelerator.load_state
        touches on resume) -- so the same tokens get drawn every time this is
        called, whether right before a save or right after a resume. Read
        from eval_bin so it doesn't depend on the (dynamic, iter_num-driven)
        train sampling ratio either.
        """
        data, ix = self._sanity_indices_and_data()
        sl = self.cfg.block_size
        x = torch.stack([torch.from_numpy(data[i : i + sl].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + sl + 1].astype(np.int64)) for i in ix])
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def _sanity_batch_hash(self) -> str:
        """Content hash of the RAW TOKEN BYTES _sanity_batch draws, computed
        independent of the model -- lets _verify_resume_sanity tell apart
        "eval_bin itself differs from what was used at save time" (this
        hash differs -- e.g. a different Modal account/volume's synced copy
        of validation.bin) from "the file is identical but the model's
        forward pass on it differs" (hash matches, sanity_loss doesn't --
        THAT'S the real weight-loading bug signal). eval_bin is synced
        per-account from S3 via download_if_missing/using_cached_file, with
        no guarantee its bytes are identical across every account/volume
        that's ever touched this run, so this check matters in exactly the
        multi-account setup this training pipeline runs under.
        """
        data, ix = self._sanity_indices_and_data()
        sl = self.cfg.block_size
        raw = b"".join(data[i : i + sl + 1].tobytes() for i in ix)
        return hashlib.sha256(raw).hexdigest()[:16]

    @torch.no_grad()
    def _sanity_loss(self) -> float:
        """Loss on the fixed sanity batch -- a pure forward pass, no
        gradient step, so it isolates the MODEL/WEIGHTS' functional
        behavior from the optimizer entirely. Recorded at every save and
        re-checked right after every resume (see _verify_resume_sanity) so
        a mismatch directly proves a weight-loading bug, rather than
        inferring it indirectly from training-loss trends. Collective
        under FSDP -- every rank must call this, not just master.
        """
        self.model.eval()
        x, y = self._sanity_batch()
        loss = self._forward_loss(x, y)
        self.model.train()
        return self.accelerator.reduce(loss, reduction="mean").item()

    def _verify_resume_sanity(self) -> None:
        """Compares the fixed sanity-batch loss right now (weights loaded,
        FSDP fully wrapped, optimizer resumed if applicable) against the
        value recorded at the last save. A large delta is direct proof the
        reloaded model is NOT functionally identical to what was saved,
        despite resume reporting success elsewhere -- a real weight-loading
        bug. A small delta proves the weights ARE fine, meaning any
        observed post-resume loss spike is coming from somewhere else
        entirely (optimizer dynamics, data, schedule), not corrupted state.

        BUT that interpretation only holds if the sanity batch itself is
        the same underlying bytes both times -- eval_bin is synced
        per-account from S3 (see _sync_data/download_if_missing), with no
        guarantee it's byte-identical across every Modal account/volume
        this run_dir has ever been resumed on. sanity_batch_hash checks
        that directly: if it differs from what was recorded at save time,
        a large delta just means "different underlying text", not a real
        bug, and gets reported as such instead of the misleading default
        interpretation.

        Also calls _sanity_loss() a SECOND time immediately after the
        first, with zero real training steps elapsed in between (no
        gradient step, no data touched, nothing that should change the
        model at all): current_sanity_loss should equal repeat_sanity_loss
        exactly if the forward pass is deterministic and fully "settled"
        right after accelerator.prepare()/load_state(). A real
        repeat_delta here -- even though nothing legitimately changed
        between the two calls -- would be direct proof of some
        non-determinism or lazy warm-up state (MoE router noise is
        eval()-gated off so shouldn't be it, but FSDP/mixed-precision
        internals settling on first use are a real category of this) that
        could also be inflating (or fully explaining) the save-vs-resume
        delta above, rather than the weights genuinely differing.
        """
        if not self._resume_dir:
            return
        meta_path = os.path.join(self._resume_dir, "trainer_state.json")
        if not os.path.isfile(meta_path):
            return
        with open(meta_path, "r") as f:
            meta = json.load(f)
        resume_from_best = self.cfg.resume_from == "best" and bool(meta.get("best_ckpt"))
        saved_sanity_loss = meta.get("best_sanity_loss") if resume_from_best else meta.get("sanity_loss")
        saved_batch_hash = meta.get("best_sanity_batch_hash") if resume_from_best else meta.get("sanity_batch_hash")
        if saved_sanity_loss is None:
            return

        current = self._sanity_loss()
        current_repeat = self._sanity_loss()
        current_batch_hash = self._sanity_batch_hash()
        delta = current - saved_sanity_loss
        repeat_delta = current_repeat - current
        batch_matches = saved_batch_hash is None or current_batch_hash == saved_batch_hash
        if not batch_matches:
            interpretation = (
                "sanity_batch_hash differs from the saved value -- eval_bin's CONTENT is not the "
                "same as when this checkpoint was saved (likely a different Modal account/volume's "
                "synced copy of the eval data), so this delta is NOT a valid weight-loading check: "
                "it's comparing loss on different underlying text, not the same text through "
                "different weights. Re-run once eval_bin is confirmed identical to get a real answer."
            )
        elif abs(repeat_delta) > 0.01:
            interpretation = (
                "repeat_delta is non-negligible despite ZERO real training steps between the two "
                "calls -- the forward pass itself is non-deterministic or some FSDP/mixed-precision "
                "state hadn't settled on first use. That non-determinism could be inflating or fully "
                "explaining the save-vs-resume delta too, rather than the weights actually differing."
            )
        else:
            interpretation = (
                "sanity_batch_hash matches and repeat_delta is negligible (same forward pass twice in "
                "a row, zero real steps between, agrees) -- so delta IS a valid weight-loading check. "
                "large |delta| -> the reloaded model is NOT functionally identical to what was saved "
                "(a real weight-loading bug); small |delta| -> weights are fine, any post-resume loss "
                "spike is coming from elsewhere (optimizer dynamics, data, schedule)"
            )
        log_fn = LOG.warning if (not batch_matches or abs(repeat_delta) > 0.01 or abs(delta) > 0.1) else LOG.info
        log_fn(
            "resume_sanity_check",
            saved_sanity_loss=saved_sanity_loss, current_sanity_loss=current, delta=delta,
            current_repeat_sanity_loss=current_repeat, repeat_delta=repeat_delta,
            batch_hash_matches=batch_matches, saved_batch_hash=saved_batch_hash, current_batch_hash=current_batch_hash,
            interpretation=interpretation,
        )

    def _verify_reference_weights(self) -> None:
        """Last-line weight sanity check, run once at startup regardless of
        init_from (resume from a checkpoint OR fresh weights from the HF
        Hub) -- compares the just-loaded model's weights against a STATIC,
        manually-maintained reference checkpoint (model.reference_repo).
        Catches a gross weight-loading failure (silently falling back to
        random/base init, a corrupted checkpoint, a wrong repo) that the
        loss-based checks above (_verify_resume_sanity) might miss or
        misattribute to something else.

        Per layer, computes relative L2 norm: ||current - reference|| /
        ||reference|| -- scale-invariant, so a huge embedding matrix and a
        tiny LayerNorm bias are judged on the same footing. These are then
        reduced to ONE number via a parameter-count-weighted global L2 norm
        (equivalent to flattening every checked tensor into one giant vector
        and taking its relative L2 norm as a whole), NOT a plain average of
        the per-layer ratios: a plain average would let a handful of small,
        noisy tensors (whose ratios are naturally larger/less stable, since
        they have little mass to average over) swing the result as much as
        the big matrices that actually hold most of the model's parameters
        -- weighting by element count is what makes "the model overall looks
        different" mean the same thing regardless of how many tiny tensors
        happen to be in the state dict. A fully-random/independently-
        reinitialized model would land around a ratio of sqrt(2) ~= 1.41
        (uncorrelated tensors of the same scale); THRESHOLD (0.5) sits well
        below that but above the drift expected from healthy continued
        training at this LR, so it should trip on catastrophic failures
        without following normal training progress into false positives --
        recalibrate from real observed values if it doesn't hold up.

        model.reference_repo is NOT auto-updated by training code -- refresh
        it yourself periodically, or normal training progress will
        eventually push the ratio past reference_weight_deviation_threshold
        too. No-ops if reference_model_repo isn't set. Every rank
        independently downloads+loads the reference (same tradeoff as
        _load_checkpoint_weights: more bandwidth/host RAM per node, simpler
        and guaranteed-correct vs. a broadcast dance). get_state_dict is
        collective under FSDP, so every rank must call it.
        """
        if not self.cfg.reference_model_repo:
            return
        try:
            ref_model = self._load_lm(self.cfg.reference_model_repo)
        except Exception as exc:
            LOG.warning("reference_weights_load_failed", repo=self.cfg.reference_model_repo, error=str(exc))
            return
        ref_state = ref_model.state_dict()
        # Kept (master only) for _startup_generation_comparison below rather
        # than reloaded there -- same weights, and a second from_pretrained
        # would re-read the whole checkpoint for nothing.
        if self.master:
            self._ref_model = ref_model
        # See _save's identical migration -- modern, non-deprecated
        # FSDP1-and-FSDP2-unified API instead of accelerate's FSDP1 path
        # through the legacy FSDP.state_dict_type() context manager.
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        current_state = get_model_state_dict(
            self.model,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=True),
        )
        del ref_model

        if not self.master:
            return

        threshold = self.cfg.reference_weight_deviation_threshold
        per_layer = []
        diff_sq_sum = 0.0
        ref_sq_sum = 0.0
        checked = 0
        for name, ref_tensor in ref_state.items():
            cur_tensor = current_state.get(name)
            if cur_tensor is None or cur_tensor.shape != ref_tensor.shape:
                continue
            checked += 1
            ref_f = ref_tensor.detach().float().cpu()
            diff_f = cur_tensor.detach().float().cpu() - ref_f
            diff_sq = diff_f.pow(2).sum().item()
            ref_sq = ref_f.pow(2).sum().item()
            diff_sq_sum += diff_sq
            ref_sq_sum += ref_sq
            layer_rel_l2 = (diff_sq ** 0.5) / (ref_sq ** 0.5) if ref_sq > 0 else float("inf")
            per_layer.append((name, round(layer_rel_l2, 5)))

        aggregate_rel_l2 = (diff_sq_sum ** 0.5) / (ref_sq_sum ** 0.5) if ref_sq_sum > 0 else float("inf")
        flagged = aggregate_rel_l2 > threshold
        per_layer.sort(key=lambda item: item[1], reverse=True)

        log_fn = LOG.warning if flagged else LOG.info
        log_fn(
            "reference_weight_check",
            repo=self.cfg.reference_model_repo, layers_checked=checked,
            aggregate_rel_l2=round(aggregate_rel_l2, 5), threshold=threshold, flagged=flagged,
            top_layers_by_deviation=per_layer[:10],  # highest-deviation layers, for pinpointing if flagged
            interpretation=(
                "aggregate_rel_l2 is a parameter-count-weighted relative L2 norm across every "
                "matched layer (not a plain average) -- flagged=true means it exceeds threshold, "
                "which could be a real loading bug (base/random weights, wrong checkpoint) OR just "
                "genuine training progress since reference_model_repo was last refreshed. "
                "top_layers_by_deviation shows where the deviation concentrates. Cross-check against "
                "resume_sanity_check and generation quality before concluding it's a bug."
            ),
        )

    def _sampling_weights(self) -> tuple[float, float]:
        return sampling_weights(
            self.cfg.eng_sampling_weight,
            self.cfg.afr_sampling_weight,
            self.iter_num,
            self.cfg.max_iters,
            self.cfg.use_scheduled_sampling,
            self.cfg.sampling_schedule,
        )

    def _eval_bin_weights(self) -> tuple[float, ...]:
        """Fixed (iteration-independent) mixture for the train-loss probe, so successive evals score
        the same blocks and the curve stays comparable while the training mixture is scheduled."""
        n = len(self.train_bins)
        if n == 2:
            return sampling_weights(self.cfg.eng_sampling_weight, self.cfg.afr_sampling_weight, 0, 1, False)
        return tuple([1.0 / n] * n)

    def get_batch(self, split: str, track: bool = False):
        """One micro-batch for this rank.

        track=True is the TRAINING draw: the next blocks of the persistent no-repeat epoch stream
        (each block used once per epoch, reshuffled per epoch, state saved in trainer_state.json).
        track=False is the EVAL draw: a fixed set of blocks (split "train" mixes the training bins,
        "val" uses the eval bin); estimate_loss resets these samplers so every eval scores the same
        windows and never advances the training stream.
        """
        if track:
            b, ids = self.sampler.next_batch(self._bin_weights())
            path, self._last_train_bin = self.train_bins[b], self.sampler.names[b]
        elif split == "train":
            b, ids = self._eval_train_sampler.next_batch(self._eval_bin_weights())
            path = self.train_bins[b]
        else:
            _, ids = self._eval_val_sampler.next_batch((1.0,))
            path = self.eval_bin
        x_np, y_np = read_blocks(self._read_memmap(path), ids, self.cfg.block_size)
        x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)

        if self.cfg.use_loss_mask:
            y = torch.stack([
                apply_label_mask(
                    row.clone(), self.cfg.mode,
                    user_token=user_token, assistant_token=assistant_token,
                    system_token=system_token, mask=MASK,
                )
                for row in y
            ])
        # else: every token contributes to the loss, unmasked -- no prompt/
        # action-span or SFT prompt-vs-response masking at all.

        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        return x, y

    # ------------------------------------------------------------------
    # Train / eval
    # ------------------------------------------------------------------

    def _lr(self, it: int) -> float:
        return lr_at(it, self.cfg)

    def _forward_loss(self, x, y):
        raw = self.accelerator.unwrap_model(self.model)
        attention_mask = self._doc_attention_mask(x)

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

        self._last_ce_loss = ce_loss.detach()
        _, lb_loss = raw.get_expert_utilization()
        if lb_loss is not None:
            return ce_loss + self.cfg.moe_aux_loss_weight * lb_loss
        return ce_loss

    # Matches the config the checkpoint was manually verified against outside
    # this pipeline (plain single-GPU/CPU, no FSDP), with two changes:
    #   - do_sample=True (was False): deterministic decoding is prone to
    #     repetitive-loop degeneration, especially while the model's
    #     next-token distribution isn't yet sharply peaked.
    #   - num_beams=1 (was 5): num_beams>1 combined with do_sample=True is
    #     NOT "no beam search" -- it's beam-sample decoding, which still runs
    #     full beam search (multiple beams, cumulative-score pruning, KV-cache
    #     reordering every step) and still exhibits beam search's well-known
    #     mode-seeking/repetition-loop tendency, just with sampled token
    #     choices layered on top. num_beams=1 is what actually turns beam
    #     search off entirely, leaving plain top-k/top-p sampling.
    #     length_penalty/early_stopping are beam-search-only knobs (they
    #     govern beam score normalization/termination) -- dropped since
    #     they're inert with num_beams=1.
    #   - top_k=40 (was 50): anywhere in 20-50 is reasonable here; 40 keeps
    #     enough of the distribution for the model to still sound varied
    #     across five languages, while trimming more of the low-probability
    #     tail that a mid-training 280M model still puts mass on (that tail
    #     is where most of the obvious "wrong language / nonsense token"
    #     samples come from). 20 would be tighter but starts hiding genuine
    #     diversity problems behind the truncation.
    #   - repetition_penalty=1.15 (was 4.0): 4.0 is far outside the usual
    #     1.05-1.3 range -- it divides the logit of every already-seen token
    #     by 4, which at that strength doesn't just discourage loops, it
    #     effectively forbids reusing common function words and punctuation,
    #     so samples drift off-topic and off-language within a few dozen
    #     tokens. That makes these samples useless as a read on the model:
    #     they'd look broken whether or not the model is. 1.15 still damps
    #     degenerate loops. This is display-only -- no effect on the loss,
    #     gradients, or anything that gets checkpointed.
    _GENERATION_CONFIG = dict(
        max_new_tokens=100,
        num_beams=1,
        do_sample=True,
        temperature=0.99,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.15,
    )

    # One-off startup comparison (see _startup_generation_comparison), run
    # once before training against model.reference_repo. Sampling reuses the
    # exact training-time config above (same decoding the periodic
    # display_model_output_iter samples use, just longer) so what you see
    # here is what you'll see mid-run; beam search is the deterministic
    # counterpart, with the sampling-only knobs dropped since they're inert
    # under do_sample=False and transformers warns about them.
    _STARTUP_MAX_NEW_TOKENS = 150
    _STARTUP_SAMPLE_CONFIG = dict(_GENERATION_CONFIG, max_new_tokens=_STARTUP_MAX_NEW_TOKENS)
    _STARTUP_BEAM_CONFIG = dict(
        max_new_tokens=_STARTUP_MAX_NEW_TOKENS,
        num_beams=5,
        do_sample=False,
        early_stopping=True,
        length_penalty=1.0,
        repetition_penalty=_GENERATION_CONFIG["repetition_penalty"],
    )

    # Five fixed prompts (~20-30 words each), one per major pretraining
    # language. Deliberately PLAIN TEXT -- no <eng>/<yor>/<ibo>/... language
    # tag, and no task tag either (see training/constant_tokens.py for the
    # tags the data itself was tokenized with). So this probes the model
    # cold, the way an untagged user prompt would arrive at inference; if
    # the model only produces coherent text WITH its tags, that shows up
    # here as visibly worse output rather than staying hidden. Nothing
    # downstream depends on these exact strings -- edit them freely.
    _STARTUP_PROMPTS = (
        (
            "The rapid growth of artificial intelligence research across Africa has opened new "
            "opportunities for local startups and universities building language technology for their "
            "own communities."
        ),
        (
            "Ìjọba ìpínlẹ̀ Èkó sọ pé àwọn ọ̀nà tuntun yóò ṣí sílẹ̀ fún àwọn oníṣòwò kékeré, "
            "kí ọrọ̀ ajé ìlú lè tẹ̀síwájú."
        ),
        (
            "Ndị ọchịchị steeti Anambra kwuru na ha ga-emezi ụzọ na ụlọ akwụkwọ dị n'ime obodo, "
            "ka ụmụ akwụkwọ nwee ike ịga akwụkwọ n'udo."
        ),
        (
            "Gwamnatin jihar Kano ta ce za ta gina sabbin hanyoyi da makarantu a ƙauyuka da dama, "
            "domin inganta rayuwar manoma da yara."
        ),
        (
            "Plenty people for Lagos dey talk say the new transport policy go make traffic better, "
            "but some drivers still dey complain well well."
        ),
        # Very short, open-ended English stub: the hardest case for a partly-trained model. A healthy
        # checkpoint continues it as a sentence; degenerate repetition or tag spam shows up immediately.
        "Technology is ",
    )

    @torch.no_grad()
    def _generate_greedy(self, prompt_ids: torch.Tensor, max_new_tokens: int = 64) -> torch.Tensor:
        """Greedy autoregressive decode using the model's own forward() directly,
        not GenerationMixin.generate(). Fallback only -- see _log_sample_generation."""
        ids = prompt_ids
        for _ in range(max_new_tokens):
            out = self.model(input_ids=ids)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

    @torch.no_grad()
    def _generate_with_config(self, prompt_ids: torch.Tensor, gen_config: dict | None = None) -> torch.Tensor:
        """Real GenerationMixin.generate() with _GENERATION_CONFIG (or an
        explicit gen_config -- see _startup_generation_comparison), temporarily
        un-sharding parameters via FSDP.summon_full_params so generate()'s
        internal machinery (prepare_inputs_for_generation, beam search, etc.)
        sees ordinary full 2-D weight tensors instead of FSDP's flat shards --
        calling generate() directly on the FSDP-wrapped model without this
        raised "'weight' must be 2-D". Collective: every rank must enter the
        context and call generate() together, matching FSDP's per-layer
        all-gather requirement."""
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        cfg = gen_config if gen_config is not None else self._GENERATION_CONFIG
        if self.fsdp_plugin is not None:
            with FSDP.summon_full_params(self.model, writeback=False, recurse=True):
                return self.model.generate(prompt_ids, pad_token_id=pad_id, **cfg)
        # DDP: every rank already holds the full model, but DistributedDataParallel does not
        # forward .generate() ("no attribute 'generate'"), so call it on the unwrapped module.
        return self.accelerator.unwrap_model(self.model).generate(prompt_ids, pad_token_id=pad_id, **cfg)

    @torch.no_grad()
    def _log_sample_generation(self, prompt_ids: torch.Tensor, tag: str = "sample_generation"):
        """Generate continuations for a batch of real prompts and log them
        (master only). Every rank must participate in generation collectively
        -- FSDP does a per-layer all-gather, so a single rank calling this
        alone would deadlock waiting on the others."""
        self.model.eval()
        method = "generate"
        try:
            generated = self._generate_with_config(prompt_ids)
        except Exception as exc:
            if self.master:
                LOG.warning("generate_failed_falling_back_to_greedy", iter=self.iter_num, error=str(exc))
            try:
                generated = self._generate_greedy(prompt_ids)
                method = "greedy_fallback"
            except Exception as exc2:
                if self.master:
                    LOG.warning("sample_generation_failed", iter=self.iter_num, error=str(exc2))
                self.model.train()
                return
        self.model.train()
        if not self.master:
            return

        n = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)
        header = f" Sample generation @ iter {self.iter_num} ({tag}, method={method}) "
        print(f"\n{header:=^100}")
        for i in range(n):
            input_text = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=False)
            output_text = self.tokenizer.decode(generated[i, prompt_len:], skip_special_tokens=False)
            print(f"--- sample {i + 1}/{n} ---")
            print(f"[INPUT]  {input_text}")
            print(f"[OUTPUT] {output_text}")
        print("=" * 100 + "\n")

    @torch.no_grad()
    def _generate_reference(self, prompt_ids: torch.Tensor, gen_config: dict) -> torch.Tensor | None:
        """Generate from the static reference checkpoint (model.reference_repo,
        loaded in _verify_reference_weights). Master-only and NOT collective:
        this is a plain, unwrapped HF model, so no FSDP all-gather is
        involved. Moved onto the training device on first use for speed,
        falling back to CPU if there isn't room for it alongside the
        training model."""
        if self._ref_model is None:
            return None
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        try:
            self._ref_model.to(self.device)
            ids = prompt_ids.to(self.device)
        except Exception as exc:
            LOG.warning("reference_model_to_device_failed", error=str(exc), fallback="cpu")
            self._ref_model.to("cpu")
            ids = prompt_ids.to("cpu")
        try:
            return self._ref_model.generate(ids, pad_token_id=pad_id, **gen_config)
        except Exception as exc:
            LOG.warning("reference_generation_failed", error=str(exc))
            return None

    @torch.no_grad()
    def _startup_generation_comparison(self) -> None:
        """One-off, before the first training step: generate from
        _STARTUP_PROMPTS with BOTH the model about to be trained and the
        static reference checkpoint (model.reference_repo), under both
        sampling and beam search, and print them side by side.

        This is the qualitative counterpart to _verify_reference_weights'
        single aggregate_rel_l2 number: that says how FAR the weights have
        moved from the reference, this shows what that movement actually did
        to the model's output -- real progress and a broken/mis-loaded
        checkpoint can produce a similar deviation number, but they don't
        read the same.

        Generation with the training model is collective (FSDP all-gathers
        per layer -- see _generate_with_config), so EVERY rank must run this
        loop in lockstep; only master generates with the reference model and
        If training.test_run is set, train() follows this with a single
        eval and then stops without saving or pushing (see _test_run_eval).
        """
        modes = (
            (f"do_sample (top_k={self._STARTUP_SAMPLE_CONFIG['top_k']}, "
             f"top_p={self._STARTUP_SAMPLE_CONFIG['top_p']}, "
             f"temperature={self._STARTUP_SAMPLE_CONFIG['temperature']})", self._STARTUP_SAMPLE_CONFIG),
            (f"beam_search (num_beams={self._STARTUP_BEAM_CONFIG['num_beams']}, do_sample=False)",
             self._STARTUP_BEAM_CONFIG),
        )
        trained_label = (
            f"MODEL BEING TRAINED  [{self.cfg.model_name}"
            f"{', resumed from ' + self.cfg.init_from if self.cfg.init_from else ''}, iter {self.iter_num}]"
        )
        ref_label = f"REFERENCE MODEL      [{self.cfg.reference_model_repo or 'not configured'}]"

        self.model.eval()
        if self.master:
            header = " STARTUP GENERATION COMPARISON (before training) "
            print(f"\n{header:#^110}")
            print(f"# prompts: {len(self._STARTUP_PROMPTS)} | max_new_tokens: {self._STARTUP_MAX_NEW_TOKENS} "
                  f"| test_run: {self.cfg.test_run}")
            print(f"# {trained_label}")
            print(f"# {ref_label}")
            print("#" * 110)

        for idx, prompt in enumerate(self._STARTUP_PROMPTS, 1):
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            prompt_len = prompt_ids.size(1)
            if self.master:
                print(f"\n{'=' * 110}")
                print(f"=== PROMPT {idx}/{len(self._STARTUP_PROMPTS)} ({prompt_len} tokens)")
                print(f"{'=' * 110}")
                print(f"[PROMPT] {prompt}")

            for mode_label, gen_config in modes:
                # Collective -- every rank calls this, master and non-master alike.
                try:
                    trained_out = self._generate_with_config(prompt_ids, gen_config)
                except Exception as exc:
                    if self.master:
                        LOG.warning("startup_generation_failed", prompt=idx, mode=mode_label, error=str(exc))
                    trained_out = None
                ref_out = self._generate_reference(prompt_ids, gen_config) if self.master else None

                if not self.master:
                    continue
                print(f"\n--- PROMPT {idx} | DECODING: {mode_label} ---")
                for label, out in ((trained_label, trained_out), (ref_label, ref_out)):
                    if out is None:
                        print(f"  [{label}]\n    <no output>")
                        continue
                    text = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=False)
                    print(f"  [{label}]\n    {text}")

        if self.master:
            print(f"\n{'#' * 110}\n")
        self._ref_model = None  # free the reference model; only needed for this comparison
        self.model.train()
        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def _curated_samples(self) -> list[dict]:
        """Tokenized curated probe set (cached), or [] when disabled/missing."""
        if getattr(self, "_curated_cache", None) is not None:
            return self._curated_cache
        self._curated_cache = []
        if not self.cfg.curated_eval_path:
            return self._curated_cache
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = resolve_path(self.cfg.curated_eval_path, repo_root)
        if not os.path.isfile(path):
            LOG.warning("curated_eval_file_missing", path=path)
            return self._curated_cache
        eos, bos = self.tokenizer.eos_token_id, self.tokenizer.bos_token_id
        for row in load_curated_samples(path):
            ids = self.tokenizer.encode(row["text"], add_special_tokens=False)
            self._curated_cache.append({"lang": row["lang"], "seq": build_sequence(ids, eos, bos)})
        LOG.info("curated_eval_loaded", path=path, samples=len(self._curated_cache))
        return self._curated_cache

    @torch.no_grad()
    def _curated_eval(self) -> dict | None:
        """CE / bits-per-byte on the curated probe set, overall and per language.

        One forward per passage (batch of 1) so short texts aren't diluted by
        padding. Collective under FSDP: every rank runs the identical forwards,
        so no reduce is needed and every rank gets the same numbers.
        """
        samples = self._curated_samples()
        if not samples:
            return None
        self.model.eval()
        totals = CuratedTotals()
        # One fixed shape for every passage (padding masked out of the loss) so
        # torch.compile, when enabled, compiles once instead of once per length.
        width = -(-(max(len(s["seq"]) for s in samples) - 1) // 128) * 128  # flex block size is 128
        eos = self.tokenizer.eos_token_id
        for sample in samples:
            seq = sample["seq"]
            pad = width - (len(seq) - 1)
            x = torch.tensor([seq[:-1] + [eos] * pad], dtype=torch.long, device=self.device)
            y = torch.tensor([seq[1:] + [MASK] * pad], dtype=torch.long, device=self.device)
            self._forward_loss(x, y)
            n_tok, n_bytes = self._target_stats(y)
            totals.add(sample["lang"], self._last_ce_loss.item(), n_tok.item(), n_bytes.item())
        self.model.train()
        return totals.summary()

    def _log_curated_eval(self, curated: dict | None) -> None:
        if curated is None or not self.master:
            return
        LOG.info("curated_eval", iter=self.iter_num, **curated)
        metrics = {"curated/ce": curated["overall"]["ce"], "curated/bpb": curated["overall"]["bpb"]}
        for lang, stats in curated["by_language"].items():
            metrics[f"curated/{lang}_ce"] = stats["ce"]
            metrics[f"curated/{lang}_bpb"] = stats["bpb"]
        self.tracker.log_metrics(metrics, step=self.iter_num)

    def estimate_loss(self):
        """Every rank evaluates a shard of eval_iters and results are averaged
        via an all-reduce, so all ranks do equal work and stay in lockstep
        (no straggler risk from an eval-only-on-master pattern)."""
        self.model.eval()
        out = {}
        local_iters = max(1, self.cfg.eval_iters // max(1, self.world_size))
        for split in ("train", "val"):
            losses = torch.zeros(local_iters, device=self.device)
            # Running [sum of CE nats, supervised tokens, decoded bytes] for bits-per-byte.
            totals = torch.zeros(3, device=self.device, dtype=torch.float64)
            (self._eval_train_sampler if split == "train" else self._eval_val_sampler).reset()  # same blocks every eval
            for k in range(local_iters):
                x, y = self.get_batch(split)
                losses[k] = self._forward_loss(x, y)
                n_tok, n_bytes = self._target_stats(y)
                totals += torch.stack([self._last_ce_loss.double() * n_tok, n_tok.double(), n_bytes.double()])
            local_mean = losses.mean()
            out[split] = self.accelerator.reduce(local_mean, reduction="mean").item()
            ce_total, tok_total, bytes_total = self.accelerator.reduce(totals, reduction="sum").tolist()
            if tok_total > 0:
                out[f"{split}_ce"] = ce_total / tok_total  # pure CE, without the MoE aux term in out[split]
                out[f"{split}_bpb"] = bits_per_byte(out[f"{split}_ce"], tok_total, bytes_total)
        self.model.train()
        return out

    def _should_push_to_hf(self, val_loss: float) -> bool:
        """Push to HF whenever this eval's val loss is "commensurate" with
        what's already on HF -- within _HF_PUSH_LOSS_BAND of the loss at the
        last push. This is a safety gate, not a throttle: local checkpoints
        already save every eval regardless of loss (see train()); this only
        decides whether THIS one is safe to publish. Refusing a push when
        val_loss has spiked/diverged from the last push is exactly what
        would have protected the HF repo from the post-resume loss spikes
        chased earlier this session (e.g. 3.6 -> 7.5) -- a spike that large
        is >> _HF_PUSH_LOSS_BAND, so it gets skipped instead of overwriting
        a good checkpoint with a bad one.
        """
        if not self.cfg.hf_chkpt_path:
            return False
        if self._last_hf_push_loss is None:
            return True
        # One-sided on purpose: an eval whose loss IMPROVED on the last push
        # is exactly what this repo should be publishing, however large the
        # improvement. Only a regression beyond the band (a post-resume spike,
        # a diverging run) is worth refusing -- the earlier abs() form also
        # blocked big improvements, which silently starved the Hub repo of
        # updates on precisely the runs that were going well.
        return val_loss <= self._last_hf_push_loss + _HF_PUSH_LOSS_BAND

    def _push_checkpoint_to_hf(self, ckpt_dir: str) -> None:
        if not self.cfg.hf_chkpt_path:
            return
        token = (
            os.environ.get("HF_WRITE_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HF_API_KEY")  # last resort: the read token in some setups
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

    def _push_checkpoint_to_s3(self, ckpt_dir: str, is_new_best: bool = False) -> None:
        """Pushes the checkpoint just saved locally (this ckpt_N's weights,
        resume_state/, trainer_state.json) to S3 on every save, so state is
        always recoverable even if this session gets torn down (e.g. a
        free-tier Modal account running out of credit) before a manual
        training/push_checkpoint_to_s3_modal.py push happens.

        Unlike that manual script, this REPLACES IN PLACE rather than
        accumulating history: once the new ckpt_N uploads successfully, the
        previously-pushed one for this run is deleted from S3, so S3 only
        ever holds a single "latest" checkpoint's weights per run (plus the
        always-current trainer_state.json/resume_state) -- not a growing
        archive. ckpt_best/resume_state_best are a SEPARATE, also
        replace-in-place slot -- fixed names, so no explicit delete needed,
        just overwrite -- only touched when is_new_best (a real improvement,
        or the one-time _backfill_ckpt_best seed). Master-only; skips
        silently if S3 isn't configured.
        """
        if not (self.cfg.s3_bucket and self.cfg.s3_endpoint and self.cfg.s3_access_key and self.cfg.s3_secret_key):
            return

        s3_kwargs = dict(
            bucket=self.cfg.s3_bucket, endpoint=self.cfg.s3_endpoint,
            access_key=self.cfg.s3_access_key, secret_key=self.cfg.s3_secret_key,
        )
        out_dir_name = os.path.basename(self.cfg.out_dir.rstrip("/"))
        run_dir_name = os.path.basename(self.run_dir.rstrip("/"))
        remote_root = f"checkpoints/{out_dir_name}/{run_dir_name}"
        ckpt_name = os.path.basename(ckpt_dir.rstrip("/"))

        try:
            if self._s3_ckpt_cleanup_pending:
                for folder in list_immediate_subfolders(remote_root, prefix=self.cfg.s3_prefix, **s3_kwargs):
                    name = folder.rstrip("/").rsplit("/", 1)[-1]
                    # ckpt_best is a fixed-name slot maintained separately
                    # below, not a numbered "latest" checkpoint -- exclude it
                    # or this cleanup would delete it on the very next push.
                    if name.startswith("ckpt_") and name != ckpt_name and name != "ckpt_best":
                        delete_prefix(f"{remote_root}/{name}", prefix=self.cfg.s3_prefix, **s3_kwargs)
                self._s3_ckpt_cleanup_pending = False

            upload_folder(ckpt_dir, f"{remote_root}/{ckpt_name}", prefix=self.cfg.s3_prefix, **s3_kwargs)

            resume_state_dir = os.path.join(self.run_dir, "resume_state")
            if os.path.isdir(resume_state_dir):
                upload_folder(
                    resume_state_dir, f"{remote_root}/resume_state",
                    prefix=self.cfg.s3_prefix, override=True, **s3_kwargs,
                )

            if is_new_best:
                best_ckpt_dir = os.path.join(self.run_dir, "ckpt_best")
                if os.path.isdir(best_ckpt_dir):
                    upload_folder(
                        best_ckpt_dir, f"{remote_root}/ckpt_best",
                        prefix=self.cfg.s3_prefix, override=True, **s3_kwargs,
                    )
                resume_state_best_dir = os.path.join(self.run_dir, "resume_state_best")
                if os.path.isdir(resume_state_best_dir):
                    upload_folder(
                        resume_state_best_dir, f"{remote_root}/resume_state_best",
                        prefix=self.cfg.s3_prefix, override=True, **s3_kwargs,
                    )

            meta_path = os.path.join(self.run_dir, "trainer_state.json")
            if os.path.isfile(meta_path):
                upload_if_absent(
                    meta_path, f"{remote_root}/trainer_state.json",
                    prefix=self.cfg.s3_prefix, override=True, **s3_kwargs,
                )

            if self._last_s3_pushed_ckpt_name and self._last_s3_pushed_ckpt_name != ckpt_name:
                delete_prefix(
                    f"{remote_root}/{self._last_s3_pushed_ckpt_name}", prefix=self.cfg.s3_prefix, **s3_kwargs,
                )
            self._last_s3_pushed_ckpt_name = ckpt_name
            LOG.info(
                "s3_checkpoint_pushed", remote=f"{remote_root}/{ckpt_name}",
                iter=self.iter_num, is_new_best=is_new_best,
            )
        except Exception as exc:
            LOG.error("s3_checkpoint_push_failed", error=str(exc))

    def _save_roundtrip_check(self, ckpt_dir: str, live_sanity_loss: float) -> None:
        """Tests a DIFFERENT hypothesis than _verify_resume_sanity: not "did
        resume correctly reconstruct the saved state" but "did the SAVE
        itself correctly capture what was live in memory." live_sanity_loss
        (just computed via a real forward pass through the live, FSDP-
        wrapped self.model) is compared against a forward pass through a
        FRESH, plain (non-FSDP) reload of the checkpoint JUST written to
        ckpt_dir -- same fixed sanity batch, same moment in time, no
        elapsed training, no distribution drift possible. If these two
        already disagree, that's direct proof get_state_dict()/
        save_pretrained() aren't faithfully capturing the live model's
        actual behavior, independent of anything resume-side. Master-only,
        deliberately not collective/FSDP -- this is specifically checking
        the plain, non-distributed reload path, matching how init_from
        actually reloads checkpoints later.
        """
        torch_dtype = self._param_torch_dtype()
        try:
            verify_model = self._load_lm(ckpt_dir, torch_dtype=torch_dtype)
            # from_pretrained's torch_dtype cast isn't always exhaustive for
            # every parameter (e.g. LayerNorm weights can be left in the
            # checkpoint's original dtype -- see _build_model's identical
            # comment) -- without this, mixed dtypes between e.g. bf16
            # linear weights and fp32 LayerNorm weights crash matmul with
            # "expected mat1 and mat2 to have the same dtype" (confirmed:
            # every save this session hit exactly that until this fix).
            verify_model = verify_model.to(torch_dtype).to(self.device)
            verify_model.eval()
            x, y = self._sanity_batch()
            attention_mask = self._doc_attention_mask(x)
            with torch.no_grad():
                out = verify_model(input_ids=x, attention_mask=attention_mask, targets=y)
                ce_loss = out.loss
                _, lb_loss = verify_model.get_expert_utilization()
                roundtrip_loss = (ce_loss + self.cfg.moe_aux_loss_weight * lb_loss if lb_loss is not None else ce_loss).item()
            del verify_model
        except Exception as exc:
            LOG.warning("save_roundtrip_check_failed", path=ckpt_dir, error=str(exc))
            return

        delta = roundtrip_loss - live_sanity_loss
        log_fn = LOG.warning if abs(delta) > 0.1 else LOG.info
        log_fn(
            "save_roundtrip_check",
            live_sanity_loss=live_sanity_loss, roundtrip_sanity_loss=roundtrip_loss, delta=delta,
            interpretation=(
                "large |delta| -> save_pretrained/get_state_dict did NOT faithfully capture what was "
                "live in self.model at save time -- the checkpoint written to disk is functionally "
                "different from the model that computed live_sanity_loss, independent of resume, "
                "distribution drift, or elapsed training entirely. small |delta| -> the save process "
                "is faithful; any later resume-time gap comes from something else (data/schedule "
                "drift, not the save itself)."
            ),
        )

    def _save(self, val_loss: float, is_new_best: bool):
        # get_state_dict / save_state are collective under FSDP (all-gather
        # across ranks) — every rank must call them, not just master.
        unwrapped = self.accelerator.unwrap_model(self.model)
        # Uses torch.distributed.checkpoint.state_dict.get_model_state_dict
        # directly instead of self.accelerator.get_state_dict(self.model):
        # for FSDP1 (what this codebase uses), accelerate's own
        # get_state_dict internally goes through the legacy
        # FSDP.state_dict_type()/FullStateDictConfig context-manager path --
        # the exact API PyTorch's own FutureWarning on every run says is
        # being deprecated in favor of this one. get_model_state_dict is the
        # same modern, unified API accelerate ITSELF uses for FSDP2 (see
        # accelerate.Accelerator.get_state_dict's source), and it explicitly
        # supports FSDP1 modules too -- not a speculative fix, a direct
        # migration off the one save-path API that's flagged as legacy.
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        state_dict = get_model_state_dict(
            self.model,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=True),
        )
        ckpt_dir = os.path.join(self.run_dir, f"ckpt_{self.iter_num}")
        best_ckpt_dir = os.path.join(self.run_dir, "ckpt_best")

        # Also collective (see _sanity_loss) -- recorded so a future resume
        # can directly verify the reloaded model is functionally identical
        # to what's saved here, rather than inferring it from training-loss
        # trends alone.
        sanity_loss = self._sanity_loss()
        # Cheap (no model/GPU involved) -- recorded alongside sanity_loss so
        # a future resume can tell "eval_bin differs" apart from "weights
        # differ" instead of assuming the latter (see _verify_resume_sanity).
        sanity_batch_hash = self._sanity_batch_hash()

        push_now = self.master and self._should_push_to_hf(val_loss)
        if self.master and self.cfg.hf_chkpt_path and not push_now:
            # Otherwise a run that never publishes looks identical to one
            # that does -- the skip was previously silent.
            LOG.warning(
                "hf_push_skipped", iter=self.iter_num, val_loss=val_loss,
                last_push_loss=self._last_hf_push_loss, band=_HF_PUSH_LOSS_BAND,
                reason="val loss regressed more than _HF_PUSH_LOSS_BAND beyond the last pushed checkpoint",
            )

        if self.master:
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped.save_pretrained(
                ckpt_dir,
                is_main_process=True,
                save_function=self.accelerator.save,
                state_dict=state_dict,
            )
            self._save_roundtrip_check(ckpt_dir, sanity_loss)
            if is_new_best:
                # Reuses the state_dict already gathered above -- another
                # local write, no extra FSDP all-gather.
                os.makedirs(best_ckpt_dir, exist_ok=True)
                unwrapped.save_pretrained(
                    best_ckpt_dir,
                    is_main_process=True,
                    save_function=self.accelerator.save,
                    state_dict=state_dict,
                )
                self._best_ckpt_dir = best_ckpt_dir
                self._best_iter_num = self.iter_num
                self._best_sanity_loss = sanity_loss
                self._best_sanity_batch_hash = sanity_batch_hash
            if push_now:
                self._last_hf_push_loss = val_loss
                self._last_hf_push_iter = self.iter_num
            with open(os.path.join(self.run_dir, "trainer_state.json"), "w") as f:
                json.dump({
                    "iter_num": self.iter_num,
                    "best_val_loss": self.best_val,
                    "best_iter_num": self._best_iter_num,
                    "latest_ckpt": ckpt_dir,
                    "best_ckpt": self._best_ckpt_dir,
                    "sanity_loss": sanity_loss,
                    "sanity_batch_hash": sanity_batch_hash,
                    "best_sanity_loss": self._best_sanity_loss,
                    "best_sanity_batch_hash": self._best_sanity_batch_hash,
                    "last_hf_push_loss": self._last_hf_push_loss,
                    "last_hf_push_iter": self._last_hf_push_iter,
                    "optimizer_name": self.cfg.optimizer_name,
                    "fresh_started": self._fresh_started,
                    "tokens_seen": self.tokens_seen,
                    "tokens_seen_offset": self.tokens_seen_offset,
                    "tokens_seen_by_bin": self.tokens_seen_by_bin,
                    "tokens_per_step": self._tokens_per_step(),
                    "sampler": self.sampler.state_dict(),
                }, f)
            LOG.info("checkpoint_saved", path=ckpt_dir, iter=self.iter_num, is_new_best=is_new_best, tokens_seen=self.tokens_seen)

        self.accelerator.save_state(os.path.join(self.run_dir, "resume_state"))
        self.accelerator.wait_for_everyone()

        if self.master and is_new_best:
            # Plain file copy, not a second collective save_state() call --
            # resume_state/ just written above IS this exact iter's state,
            # so copying it sideways is equivalent at a fraction of the cost.
            resume_state_dir = os.path.join(self.run_dir, "resume_state")
            resume_state_best_dir = os.path.join(self.run_dir, "resume_state_best")
            if os.path.isdir(resume_state_dir):
                shutil.copytree(resume_state_dir, resume_state_best_dir, dirs_exist_ok=True)

        if push_now:
            self._push_checkpoint_to_hf(ckpt_dir)
        if self.master:
            self._push_checkpoint_to_s3(ckpt_dir, is_new_best=is_new_best)

    def _log_moe_stats(self) -> None:
        """Surfaces per-layer expert utilization (load) and average router
        probability -- the actual numbers needed to see whether the router
        has collapsed onto a handful of experts, rather than inferring it
        indirectly from the scalar aux loss alone. Reads the same
        block.mlp._expert_utilization/_router_probs attributes
        get_expert_utilization() already reads for the aux loss (set during
        the model's own forward -- this doesn't depend on which modeling.py
        is actually loaded via trust_remote_code, just that it sets those
        attributes, which it demonstrably does since aux loss training
        already works). Reflects the last forward call before this point,
        i.e. the last eval batch. Master-only; every rank runs the same
        forward pass so the stats don't differ across ranks.
        """
        if not self.master:
            return
        raw = self.accelerator.unwrap_model(self.model)
        if not getattr(raw.config, "use_moe", False):
            return

        layers = []
        for i, block in enumerate(raw.transformer.h):
            if not (hasattr(block, "use_moe") and block.use_moe and hasattr(block.mlp, "_expert_utilization")):
                continue
            util = block.mlp._expert_utilization.tolist()
            probs = block.mlp._router_probs.tolist() if hasattr(block.mlp, "_router_probs") else None
            num_experts = len(util)
            # Normalized entropy of the router probability distribution:
            # 1.0 = perfectly balanced across all experts, 0.0 = fully
            # collapsed onto a single expert -- the single-number collapse
            # signal to watch; the raw per-expert arrays are there for
            # actually seeing which experts are starved.
            entropy = None
            if probs and num_experts > 1:
                p = torch.tensor(probs).clamp_min(1e-12)
                entropy = float(-(p * p.log()).sum() / math.log(num_experts))
            layers.append({
                "layer": i,
                "max_util": round(max(util), 4),
                "entropy": round(entropy, 4) if entropy is not None else None,
                "utilization": [round(u, 4) for u in util],
                "router_probs": [round(p, 4) for p in probs] if probs else None,
            })

        if not layers:
            return
        # Re-reads the same per-layer _aux_lb attributes get_expert_utilization()
        # already aggregates for the training loss -- free (no extra forward
        # pass), just surfaces a number that was previously computed every
        # step but never actually logged anywhere.
        _, lb_loss = raw.get_expert_utilization()
        lb_loss_value = float(lb_loss) if lb_loss is not None and not isinstance(lb_loss, int) else None

        LOG.info("moe_expert_stats", iter=self.iter_num, lb_loss=lb_loss_value, layers=layers)

        if self.tracker.enabled:
            m = {"moe/lb_loss": lb_loss_value}
            for l in layers:
                m[f"moe/layer{l['layer']}_max_util"] = l["max_util"]
                m[f"moe/layer{l['layer']}_entropy"] = l["entropy"]
            self.tracker.log_metrics(m, step=self.iter_num)

        if self.cfg.wandb_log:
            try:
                import wandb
                log_dict = {}
                if lb_loss_value is not None:
                    log_dict["moe/lb_loss"] = lb_loss_value
                for l in layers:
                    log_dict[f"moe/layer{l['layer']}_max_util"] = l["max_util"]
                    if l["entropy"] is not None:
                        log_dict[f"moe/layer{l['layer']}_entropy"] = l["entropy"]
                wandb.log(log_dict, step=self.iter_num)
            except Exception as exc:
                LOG.warning("wandb_moe_log_failed", error=str(exc))

    def _maybe_log_wandb(self, losses, lr):
        if not self.cfg.wandb_log or not self.master:
            return
        try:
            import wandb
            wandb.log({"eval/train": losses["train"], "eval/val": losses["val"], "lr": lr}, step=self.iter_num)
        except Exception as exc:
            LOG.warning("wandb_log_failed", error=str(exc))
            self.cfg.wandb_log = False

    def _log_eval_mlflow(self, losses: dict, lr: float) -> None:
        m = {"lr": lr, "eval/train_loss": losses["train"], "eval/val_loss": losses["val"]}
        for split in ("train", "val"):
            ce = losses.get(f"{split}_ce")
            m[f"eval/{split}_ce"] = ce
            m[f"eval/{split}_bpb"] = losses.get(f"{split}_bpb")
            m[f"eval/{split}_ppl"] = math.exp(min(ce, 50.0)) if ce is not None else None
        m.update(self._token_metrics())
        self.tracker.log_metrics(m, step=self.iter_num)

    def _token_log_fields(self) -> dict:
        fields = {"tokens_seen": self.tokens_seen, "tokens_seen_b": round(self.tokens_seen / 1e9, 4)}
        if self.tokens_seen_offset:
            fields["lifetime_tokens_seen_b"] = round((self.tokens_seen_offset + self.tokens_seen) / 1e9, 4)
        if len(self.tokens_seen_by_bin) > 1:
            fields["tokens_seen_by_bin"] = dict(self.tokens_seen_by_bin)
        if getattr(self, "sampler", None) is not None:
            fields["epochs_by_bin"] = self.sampler.epochs_done()
        return fields

    def _token_metrics(self) -> dict:
        m = {"train/tokens_seen": self.tokens_seen}
        if self.tokens_seen_offset:
            m["train/lifetime_tokens_seen"] = self.tokens_seen_offset + self.tokens_seen
        for name, n in self.tokens_seen_by_bin.items():
            m[f"train/tokens_seen_{name}"] = n
        if getattr(self, "sampler", None) is not None:
            for name, e in self.sampler.epochs_done().items():
                m[f"train/epoch_{name}"] = e
        return m

    def _log_step_mlflow(self, loss, last_y, lr, tokens_per_sec, tflops_per_gpu, mfu, log_kwargs) -> None:
        if not self.tracker.enabled:
            return
        ce = self._last_ce_loss.item() if self._last_ce_loss is not None else None
        n_tok, n_bytes = (t.item() for t in self._target_stats(last_y))
        self.tracker.log_metrics({
            "train/loss": loss,  # CE + weighted MoE aux loss: what is actually optimised
            "train/ce_loss": ce,
            "train/bpb": bits_per_byte(ce, n_tok, n_bytes) if ce is not None else None,
            "train/ppl": math.exp(min(ce, 50.0)) if ce is not None else None,
            "train/grad_norm": float(self._last_grad_norm) if self._last_grad_norm is not None else None,
            "lr": lr,
            **self._token_metrics(),
            "train/tokens_per_sec": tokens_per_sec,
            "train/tflops_per_gpu": tflops_per_gpu,
            "train/mfu": mfu,
            "train/peak_mem_gib": log_kwargs.get("peak_mem_gib"),
            "train/eng_sampling_weight": log_kwargs.get("eng_sampling_weight"),
            "train/afr_sampling_weight": log_kwargs.get("afr_sampling_weight"),
        }, step=self.iter_num)

    def _log_step_wandb(self, loss: float, tokens_per_sec: float, tflops_per_gpu: float, mfu: float | None) -> None:
        if not self.cfg.wandb_log or not self.master:
            return
        try:
            import wandb
            log_dict = {
                "train/loss": loss,
                "train/tokens_per_sec": tokens_per_sec,
                "train/tflops_per_gpu": tflops_per_gpu,
            }
            if mfu is not None:
                log_dict["train/mfu"] = mfu
            wandb.log(log_dict, step=self.iter_num)
        except Exception as exc:
            LOG.warning("wandb_step_log_failed", error=str(exc))

    def _sample_prompt(self, x: torch.Tensor, num_samples: int = 5) -> torch.Tensor:
        """Real token ids straight from the current batch -- up to num_samples
        rows (fewer if train_batch_size is smaller), passed to the model
        together as one batch."""
        prompt_len = min(32, x.size(1))
        n = min(num_samples, x.size(0))
        return x[:n, :prompt_len]

    def _test_run_eval(self) -> None:
        """training.test_run: one eval, logged, then stop -- BEFORE anything
        is written or published.

        Runs after __init__ has already logged the two deviation checks
        (_verify_resume_sanity's saved-vs-current sanity loss, and
        _verify_reference_weights' aggregate_rel_l2 against
        model.reference_repo) and the startup generation comparison. This
        adds the actual val/train loss for the loaded checkpoint, plus how
        it compares to the best_val_loss recorded in the checkpoint's own
        trainer_state.json, and returns immediately.

        Deliberately does NOT call _save: no ckpt_N/ or resume_state/
        written locally, nothing uploaded to S3, nothing pushed to the HF
        Hub. estimate_loss is collective, so every rank runs it.
        """
        losses = self.estimate_loss()
        self._log_curated_eval(self._curated_eval())
        if self.master:
            lr = self._lr(self.iter_num) if self.cfg.decay_lr else self.cfg.learning_rate
            LOG.info("eval", iter=self.iter_num, tokens_seen=self.tokens_seen, **losses)
            self._maybe_log_wandb(losses, lr)
            self._log_eval_mlflow(losses, lr)
            self._log_moe_stats()
            # self.best_val is whatever the resumed trainer_state.json
            # recorded (1e9 if this is a fresh run with no checkpoint).
            resumed_best = self.best_val if self.best_val < 1e8 else None
            LOG.info(
                "test_run_complete",
                iter=self.iter_num,
                val_loss=losses["val"],
                train_loss=losses["train"],
                checkpoint_best_val_loss=resumed_best,
                val_loss_delta_vs_checkpoint_best=(
                    losses["val"] - resumed_best if resumed_best is not None else None
                ),
                note="training.test_run is true -- evaluated the loaded checkpoint and stopped. "
                     "NOTHING was saved locally, pushed to S3, or pushed to the HF Hub. "
                     "Set test_run: false to train. A positive delta just means this eval batch "
                     "scored worse than the best eval recorded in the checkpoint's trainer_state.json.",
            )
        self.tracker.end()

    def train(self):
        if self.master:
            LOG.info(
                "training_start", mode=self.cfg.mode, world_size=self.world_size, iter=self.iter_num,
                tokens_per_step=self._tokens_per_step(), max_iters=self.cfg.max_iters,
                tokens_seen_estimated=self._tokens_seen_estimated, **self._token_log_fields(),
            )

        x, y = self.get_batch("train", track=True)
        cur_bin = self._last_train_bin

        # Sanity-check the loaded checkpoint (and FSDP wrapping) before
        # spending any real training time on it.
        self._log_sample_generation(self._sample_prompt(x), tag="startup_sample_generation")

        if self.cfg.test_run:
            self._test_run_eval()
            return

        t0 = time.time()
        last_loss = None

        while self.iter_num <= self.cfg.max_iters:
            lr = self._lr(self.iter_num) if self.cfg.decay_lr else self.cfg.learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            if self.iter_num % self.cfg.eval_interval == 0:
                losses = self.estimate_loss()
                self._log_curated_eval(self._curated_eval())
                if self.master:
                    LOG.info("eval", iter=self.iter_num, tokens_seen=self.tokens_seen, **losses)
                    self._maybe_log_wandb(losses, lr)
                    self._log_eval_mlflow(losses, lr)
                    self._log_moe_stats()
                # Checkpoint ("latest") on every eval regardless of whether
                # val loss improved. Whether it's ALSO a new best additionally
                # gates a separate ckpt_best/resume_state_best save (see
                # _save) -- must be computed before best_val is updated below,
                # or every eval would trivially look like a "new best" against
                # its own already-folded-in value. Pushing to the HF Hub
                # remains selective (_should_push_to_hf).
                is_new_best = losses["val"] < self.best_val
                self.best_val = min(self.best_val, losses["val"])
                if self.iter_num > 0 and not self._suppress_first_save:
                    self._save(losses["val"], is_new_best)
                elif self.master and self._suppress_first_save:
                    LOG.info(
                        "first_eval_save_suppressed", iter=self.iter_num, val_loss=losses["val"],
                        note="not checkpointed/pushed -- inspect this eval and stop now if it looks wrong",
                    )
                self._suppress_first_save = False

            if self.iter_num == 0 and self.cfg.eval_only:
                break

            if (
                self.iter_num > 0
                and self.cfg.display_model_output_iter > 0
                and self.iter_num % self.cfg.display_model_output_iter == 0
            ):
                self._log_sample_generation(self._sample_prompt(x))

            for _ in range(self.cfg.gradient_accumulation_steps):
                self.tokens_seen_by_bin[cur_bin] = (
                    self.tokens_seen_by_bin.get(cur_bin, 0)
                    + self.cfg.train_batch_size * self.cfg.block_size * self.world_size
                )
                with self.accelerator.accumulate(self.model):
                    loss = self._forward_loss(x, y)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients and self.cfg.grad_clip > 0:
                        self._last_grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                last_y = y
                x, y = self.get_batch("train", track=True)
                cur_bin = self._last_train_bin
            self.tokens_seen += self._tokens_per_step()
            last_loss = loss

            if self.iter_num % self.cfg.log_interval == 0 and self.master:
                dt = time.time() - t0
                iters_elapsed = max(1, self.iter_num - self._last_logged_iter)
                self._last_logged_iter = self.iter_num
                tokens_processed = (
                    self.cfg.train_batch_size * self.cfg.block_size
                    * self.cfg.gradient_accumulation_steps * self.world_size * iters_elapsed
                )
                tokens_per_sec = tokens_processed / dt if dt > 0 else 0.0
                tflops_per_gpu, mfu = compute_mfu(
                    self._flops_per_token, tokens_processed, dt, self.world_size, self._peak_flops_per_gpu,
                )
                log_kwargs = {
                    "iter": self.iter_num, "loss": last_loss.item(), "ms": dt * 1000,
                    "tokens_per_sec": round(tokens_per_sec, 1),
                    "tflops_per_gpu": round(tflops_per_gpu, 2),
                }
                if mfu is not None:
                    log_kwargs["mfu"] = round(mfu, 4)
                log_kwargs.update(self._token_log_fields())
                if torch.cuda.is_available():
                    # Peak allocated since the previous log line: how close train_batch_size is to
                    # this GPU's limit. Raise the batch if it is far below, lower it before an OOM.
                    log_kwargs["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    torch.cuda.reset_peak_memory_stats()
                if self.cfg.use_scheduled_sampling and len(self.train_bins) > 1:
                    eng_w, afr_w = self._sampling_weights()
                    log_kwargs.update(eng_sampling_weight=eng_w, afr_sampling_weight=afr_w)
                LOG.info("step", **log_kwargs)
                self._log_step_wandb(last_loss.item(), tokens_per_sec, tflops_per_gpu, mfu)
                self._log_step_mlflow(
                    last_loss.item(), last_y, lr, tokens_per_sec, tflops_per_gpu, mfu, log_kwargs,
                )
                t0 = time.time()

            self.iter_num += 1

        if self.master:
            LOG.info("training_done", iter=self.iter_num, **self._token_log_fields())
        self.tracker.end()


def main():
    config = load_train_config()
    trainer = Trainer(config)
    try:
        trainer.train()
    except BaseException:
        trainer.tracker.end(status="FAILED")  # no-op unless MLflow is active (master only)
        raise


if __name__ == "__main__":
    main()
