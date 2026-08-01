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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.constant_tokens import MASK, assistant_token, end_of_text_token, system_token, user_token
from training.label_masking import apply_label_mask
from training.load_config import TrainConfig, load_train_config, sampling_weights
from training.mfu import compute_mfu, model_flops_per_token, peak_flops_for_current_device
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

# Val-loss band (see Trainer._should_push_to_hf) within which loss is
# considered "oscillating"/plateaued rather than having definitely moved.
_HF_PUSH_LOSS_BAND = 0.25


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
        self._setup_accelerator()
        self._setup_dirs()
        self._setup_wandb()
        self._setup_data()
        self._build_model()
        self._build_optimizer()
        self._prepare_for_training()
        self._verify_resume_sanity()

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

        if self._resume_dir:
            self.run_dir = self._resume_dir
            LOG.info("found_existing_checkpoint_dir", path=self.run_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(self.cfg.out_dir, f"{ts}_{self.cfg.mode}")

        if self.master:
            os.makedirs(self.run_dir, exist_ok=True)

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
                    if latest_ckpt:
                        # latest_ckpt in the downloaded json is an absolute
                        # path from wherever it was originally saved (e.g. a
                        # different machine/volume) -- repoint it at this
                        # machine's actual local copy.
                        meta["latest_ckpt"] = os.path.join(local_run_dir, os.path.basename(latest_ckpt.rstrip("/")))
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

    def _resolve_resume_weights_path(self) -> str | None:
        """The local model-weights directory to load from when
        init_from=="resume": the latest_ckpt recorded in the resumed run's
        trainer_state.json (see _setup_dirs for how _resume_dir itself is
        found), if any checkpoint has actually been saved there yet."""
        if not self._resume_dir:
            return None
        meta_path = os.path.join(self._resume_dir, "trainer_state.json")
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path, "r") as f:
            meta = json.load(f)
        latest_ckpt = meta.get("latest_ckpt")
        if latest_ckpt and os.path.isdir(latest_ckpt):
            return latest_ckpt
        return None

    def _build_model(self):
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16)

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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name, trust_remote_code=True, torch_dtype=torch_dtype,
        )
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
        ckpt_model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, trust_remote_code=True, torch_dtype=torch_dtype,
        )
        self.model.load_state_dict(ckpt_model.state_dict(), strict=True)
        del ckpt_model
        LOG.info("resume_checkpoint_weights_loaded", path=ckpt_dir)

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

        # Always attempt to resume optimizer state / iter_num / best_val from
        # the latest checkpoint (self._resume_dir, found in _setup_dirs),
        # regardless of init_from -- init_from only controls where MODEL
        # WEIGHTS come from (see _build_model). The LR schedule (_lr) and the
        # dynamic eng/afr sampling-ratio schedule (sampling_weights) are both
        # pure functions of iter_num, not separate stateful objects, so
        # restoring iter_num alone is what continues them correctly.
        if self._resume_dir:
            meta_path = os.path.join(self._resume_dir, "trainer_state.json")
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
                self.iter_num = meta.get("iter_num", 0)
                self.best_val = meta.get("best_val_loss", 1e9)
                self._last_hf_push_loss = meta.get("last_hf_push_loss")
                self._last_hf_push_iter = meta.get("last_hf_push_iter", 0)

            resume_state_dir = os.path.join(self._resume_dir, "resume_state")
            if os.path.isdir(resume_state_dir):
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
                    LOG.info("resumed_training_state", path=resume_state_dir, iter=self.iter_num)

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

    def _sanity_batch(self):
        """A FIXED, deterministic batch -- drawn via a fixed numpy seed
        independent of the live torch RNG stream (which accelerator.load_state
        touches on resume) -- so the same tokens get drawn every time this is
        called, whether right before a save or right after a resume. Read
        from eval_bin so it doesn't depend on the (dynamic, iter_num-driven)
        train sampling ratio either.
        """
        rng = np.random.default_rng(1234567)
        bs, sl = self.cfg.train_batch_size, self.cfg.block_size
        data = self._read_memmap(self.eval_bin)
        ix = rng.integers(0, len(data) - sl - 1, size=bs)
        x = torch.stack([torch.from_numpy(data[i : i + sl].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + sl + 1].astype(np.int64)) for i in ix])
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

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
        """
        if not self._resume_dir:
            return
        meta_path = os.path.join(self._resume_dir, "trainer_state.json")
        if not os.path.isfile(meta_path):
            return
        with open(meta_path, "r") as f:
            meta = json.load(f)
        saved_sanity_loss = meta.get("sanity_loss")
        if saved_sanity_loss is None:
            return

        current = self._sanity_loss()
        delta = current - saved_sanity_loss
        log_fn = LOG.warning if abs(delta) > 0.1 else LOG.info
        log_fn(
            "resume_sanity_check",
            saved_sanity_loss=saved_sanity_loss, current_sanity_loss=current, delta=delta,
            interpretation=(
                "large |delta| -> the reloaded model is NOT functionally identical to what was "
                "saved (a real weight-loading bug); small |delta| -> weights are fine, any "
                "post-resume loss spike is coming from elsewhere (optimizer dynamics, data, schedule)"
            ),
        )

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
    _GENERATION_CONFIG = dict(
        max_new_tokens=100,
        num_beams=1,
        do_sample=True,
        temperature=0.99,
        top_k=50,
        top_p=0.95,
        repetition_penalty=4.0,
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
    def _generate_with_config(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Real GenerationMixin.generate() with _GENERATION_CONFIG, temporarily
        un-sharding parameters via FSDP.summon_full_params so generate()'s
        internal machinery (prepare_inputs_for_generation, beam search, etc.)
        sees ordinary full 2-D weight tensors instead of FSDP's flat shards --
        calling generate() directly on the FSDP-wrapped model without this
        raised "'weight' must be 2-D". Collective: every rank must enter the
        context and call generate() together, matching FSDP's per-layer
        all-gather requirement."""
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if self.fsdp_plugin is not None:
            with FSDP.summon_full_params(self.model, writeback=False, recurse=True):
                return self.model.generate(prompt_ids, pad_token_id=pad_id, **self._GENERATION_CONFIG)
        return self.model.generate(prompt_ids, pad_token_id=pad_id, **self._GENERATION_CONFIG)

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

    def _should_push_to_hf(self, val_loss: float) -> bool:
        """Local checkpoints now save on every eval regardless of val loss
        (see train()), but pushing every one of those to the HF Hub would be
        wasteful. Push when:
          - nothing has been pushed yet this run, or
          - val loss has moved by more than _HF_PUSH_LOSS_BAND from the loss
            at the last push (a real, decisive change worth recording), or
          - hf_push_interval iters have elapsed since the last push AND loss
            has stayed within that band the whole time (i.e. it's plateaued/
            oscillating rather than trending) -- keeps the HF repo from going
            stale during a long plateau without pushing on every single eval.
        """
        if not self.cfg.hf_chkpt_path:
            return False
        if self._last_hf_push_loss is None:
            return True
        moved = abs(val_loss - self._last_hf_push_loss) > _HF_PUSH_LOSS_BAND
        if moved:
            return True
        return (self.iter_num - self._last_hf_push_iter) >= self.cfg.hf_push_interval

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

    def _push_checkpoint_to_s3(self, ckpt_dir: str) -> None:
        """Pushes the checkpoint just saved locally (this ckpt_N's weights,
        resume_state/, trainer_state.json) to S3 on every save, so state is
        always recoverable even if this session gets torn down (e.g. a
        free-tier Modal account running out of credit) before a manual
        training/push_checkpoint_to_s3_modal.py push happens.

        Unlike that manual script, this REPLACES IN PLACE rather than
        accumulating history: once the new ckpt_N uploads successfully, the
        previously-pushed one for this run is deleted from S3, so S3 only
        ever holds a single checkpoint's weights per run (plus the always-
        current trainer_state.json/resume_state) -- not a growing archive.
        Master-only; skips silently if S3 isn't configured.
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
                    if name.startswith("ckpt_") and name != ckpt_name:
                        delete_prefix(f"{remote_root}/{name}", prefix=self.cfg.s3_prefix, **s3_kwargs)
                self._s3_ckpt_cleanup_pending = False

            upload_folder(ckpt_dir, f"{remote_root}/{ckpt_name}", prefix=self.cfg.s3_prefix, **s3_kwargs)

            resume_state_dir = os.path.join(self.run_dir, "resume_state")
            if os.path.isdir(resume_state_dir):
                upload_folder(
                    resume_state_dir, f"{remote_root}/resume_state",
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
            LOG.info("s3_checkpoint_pushed", remote=f"{remote_root}/{ckpt_name}", iter=self.iter_num)
        except Exception as exc:
            LOG.error("s3_checkpoint_push_failed", error=str(exc))

    def _save(self, val_loss: float):
        # get_state_dict / save_state are collective under FSDP (all-gather
        # across ranks) — every rank must call them, not just master.
        unwrapped = self.accelerator.unwrap_model(self.model)
        state_dict = self.accelerator.get_state_dict(self.model)
        ckpt_dir = os.path.join(self.run_dir, f"ckpt_{self.iter_num}")

        # Also collective (see _sanity_loss) -- recorded so a future resume
        # can directly verify the reloaded model is functionally identical
        # to what's saved here, rather than inferring it from training-loss
        # trends alone.
        sanity_loss = self._sanity_loss()

        push_now = self.master and self._should_push_to_hf(val_loss)

        if self.master:
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped.save_pretrained(
                ckpt_dir,
                is_main_process=True,
                save_function=self.accelerator.save,
                state_dict=state_dict,
            )
            if push_now:
                self._last_hf_push_loss = val_loss
                self._last_hf_push_iter = self.iter_num
            with open(os.path.join(self.run_dir, "trainer_state.json"), "w") as f:
                json.dump({
                    "iter_num": self.iter_num,
                    "best_val_loss": self.best_val,
                    "latest_ckpt": ckpt_dir,
                    "sanity_loss": sanity_loss,
                    "last_hf_push_loss": self._last_hf_push_loss,
                    "last_hf_push_iter": self._last_hf_push_iter,
                }, f)
            LOG.info("checkpoint_saved", path=ckpt_dir, iter=self.iter_num)

        self.accelerator.save_state(os.path.join(self.run_dir, "resume_state"))
        self.accelerator.wait_for_everyone()

        if push_now:
            self._push_checkpoint_to_hf(ckpt_dir)
        if self.master:
            self._push_checkpoint_to_s3(ckpt_dir)

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
                    self._log_moe_stats()
                # Checkpoint on every eval regardless of whether val loss
                # improved -- best_val is still tracked (for metadata/logging),
                # but no longer gates whether a checkpoint is written.
                # Pushing to the HF Hub remains selective (_should_push_to_hf).
                self.best_val = min(self.best_val, losses["val"])
                if self.iter_num > 0 and not self._suppress_first_save:
                    self._save(losses["val"])
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
                if self.cfg.use_scheduled_sampling and len(self.train_bins) > 1:
                    eng_w, afr_w = self._sampling_weights()
                    log_kwargs.update(eng_sampling_weight=eng_w, afr_sampling_weight=afr_w)
                LOG.info("step", **log_kwargs)
                self._log_step_wandb(last_loss.item(), tokens_per_sec, tflops_per_gpu, mfu)
                t0 = time.time()

            self.iter_num += 1

        if self.master:
            LOG.info("training_done", iter=self.iter_num)


def main():
    config = load_train_config()
    Trainer(config).train()


if __name__ == "__main__":
    main()
