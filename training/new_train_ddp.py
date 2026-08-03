#!/usr/bin/env python3
"""Plain DDP (DistributedDataParallel, no FSDP) variant of new_train.py's
Trainer, for isolating whether the resume loss-spike bug chased at length
in new_train.py's diagnostics (resume_sanity_check, load_state_sanity_check,
save_roundtrip_check, reference_weight_check) is specific to FSDP's
checkpoint machinery. Every GPU holds a full, unsharded copy of the model
under DDP, so there's no sharded-state-dict gather/collate step at all --
if the resume loss spike disappears here, that's direct evidence the bug
lives in FSDP's save/restore path, not in this codebase's data, schedule,
or masking logic (which this script shares byte-for-byte with new_train.py).

Does NOT modify new_train.py -- DDPTrainer subclasses Trainer and overrides
only _setup_accelerator (build a plain Accelerator, no fsdp_plugin) and
_generate_with_config (DistributedDataParallel doesn't forward .generate()
the way FSDP does -- confirmed this session: calling it directly raises
"'DistributedDataParallel' object has no attribute 'generate'" and silently
falls back to slower/lower-quality greedy decoding every time). Every other
method -- data loading, checkpoint save/push, all the sanity/reference/
round-trip diagnostics, eval, the training loop itself -- is inherited
unchanged, so this is a pure "how is the model wrapped for compute"
swap, not a different training pipeline. Reads the same train_config.yaml.

Memory warning: DDP does NOT shard optimizer/gradient/parameter state
across GPUs the way FSDP does -- every GPU needs enough memory for the
FULL model + full Adam optimizer state. This is exactly what caused an
OOM crash earlier this session when FSDP silently failed to activate and
DDP kicked in by accident at the same batch_size/block_size the FSDP path
was tuned for. Expect to need a smaller train_batch_size/block_size (or
fewer GPUs' worth of gradient_accumulation_steps) than your normal FSDP
config to fit -- this is meant as a short, targeted diagnostic run, not a
drop-in replacement for the FSDP training path.

Launch (same pattern as new_train.py):
  python -m training.new_train_ddp                                        # single GPU / CPU smoke test
  torchrun --standalone --nproc_per_node=4 -m training.new_train_ddp       # single node, multi-GPU
"""

from __future__ import annotations

import os

import structlog
import torch
from accelerate import Accelerator

from training.load_config import load_train_config
from training.new_train import Trainer

LOG = structlog.get_logger()


class DDPTrainer(Trainer):
    """Trainer, but always plain DDP -- see module docstring."""

    def _setup_accelerator(self):
        # Same world_size/gradient_accumulation_steps bookkeeping as
        # Trainer._setup_accelerator (kept in sync deliberately -- this
        # part has nothing to do with FSDP vs DDP), just without ever
        # building an fsdp_plugin.
        raw_world_size = os.environ.get("WORLD_SIZE")
        raw_local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
        LOG.info(
            "distributed_env_check",
            WORLD_SIZE=raw_world_size, LOCAL_WORLD_SIZE=raw_local_world_size,
            RANK=os.environ.get("RANK"), LOCAL_RANK=os.environ.get("LOCAL_RANK"),
            backend="ddp",
        )
        world_size_env = int(raw_world_size or raw_local_world_size or 1)
        if world_size_env > 1 and self.cfg.gradient_accumulation_steps % world_size_env == 0:
            self.cfg.gradient_accumulation_steps //= world_size_env
        self.cfg.gradient_accumulation_steps = max(1, self.cfg.gradient_accumulation_steps)

        # Deliberately no fsdp_plugin at all -- Accelerate wraps with plain
        # DDP whenever world_size > 1 and no fsdp_plugin/ACCELERATE_USE_FSDP
        # is present. self.fsdp_plugin stays None, which every FSDP-specific
        # branch elsewhere in Trainer (compile_skipped's multi-process
        # check, _generate_with_config's summon_full_params) already checks
        # for -- no changes needed there.
        self.fsdp_plugin = None
        self.accelerator = Accelerator(
            mixed_precision=self._accelerate_precision(),
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
        )
        LOG.info("distributed_type_resolved", distributed_type=str(self.accelerator.state.distributed_type))
        self.device = self.accelerator.device
        self.master = self.accelerator.is_main_process
        self.world_size = self.accelerator.num_processes
        torch.manual_seed(self.cfg.seed + self.accelerator.process_index)

    def _generate_with_config(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """DDP has no FSDP.summon_full_params concern (every rank already
        holds the full, unsharded model), but .generate() isn't forwarded
        through DistributedDataParallel's wrapper the way it is through
        FSDP's -- confirmed this session, calling it directly raises
        "'DistributedDataParallel' object has no attribute 'generate'" and
        silently falls back to greedy decoding (Trainer._log_sample_generation's
        try/except) every single time. Call it on the unwrapped module
        instead so DDP runs actually get real top-k/top-p sampled output
        to compare against the FSDP path's, not a permanently-degraded
        fallback.
        """
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        raw = self.accelerator.unwrap_model(self.model)
        return raw.generate(prompt_ids, pad_token_id=pad_id, **self._GENERATION_CONFIG)


def main():
    config = load_train_config()
    DDPTrainer(config).train()


if __name__ == "__main__":
    main()
