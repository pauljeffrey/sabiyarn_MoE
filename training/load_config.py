"""Load and normalize training/train_config.yaml."""

from __future__ import annotations

import ast
import math
import operator
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# Loaded once at import time so every entrypoint that reads TrainConfig (new_train.py,
# modal_train.py, prepare.py, etc.) picks up secrets (S3 keys, HF/wandb tokens) from a
# local .env file without each script needing its own load_dotenv() call.
load_dotenv()


def _normalize_list_sections(text: str) -> str:
    """Normalize YAML list-of-mapping sections like `tokenizer: [name, {k: v}]`."""
    lines = text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            out.append(line)
            i += 1
            continue

        match = re.match(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_.\-/]+):\s*$", line)
        if match:
            indent = len(match.group("indent"))
            key = match.group("key")
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                next_indent = len(lines[j]) - len(lines[j].lstrip(" "))
                if next_indent > indent and lines[j].lstrip().startswith("- "):
                    entries: List[str] = []
                    k = j
                    while k < len(lines):
                        cur = lines[k]
                        if not cur.strip() or cur.lstrip().startswith("#"):
                            # Blank lines and standalone comment lines don't end
                            # the list block -- they're just skipped, not
                            # treated as data or as a terminator.
                            k += 1
                            continue
                        cur_indent = len(cur) - len(cur.lstrip(" "))
                        if cur_indent <= indent:
                            break
                        if not cur.lstrip().startswith("- "):
                            break
                        entries.append(cur)
                        k += 1

                    if entries:
                        out.append(line)
                        child_indent = " " * (indent + 2)
                        for entry in entries:
                            entry_text = entry.lstrip()[2:].strip()
                            if not entry_text:
                                continue
                            if ":" in entry_text:
                                sub_key, _, value = entry_text.partition(':')
                                sub_key = sub_key.strip()
                                value = value.strip()
                                if value:
                                    out.append(f"{child_indent}{sub_key}: {value}")
                                else:
                                    out.append(f"{child_indent}{sub_key}:")
                            else:
                                out.append(f"{child_indent}name: {entry_text}")
                        i = k
                        continue

        out.append(line)
        i += 1

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _merge_list_section(items: Any) -> Dict[str, Any]:
    """Merge YAML list sections like `[model_name, {k: v}, ...]` or a plain mapping."""
    if items is None:
        return {}
    if isinstance(items, dict):
        return dict(items)

    out: Dict[str, Any] = {}
    for item in items or []:
        if isinstance(item, str):
            out.setdefault("name", item)
        elif isinstance(item, dict):
            out.update(item)
    return out


def _nested_list_dict(items: Any) -> Dict[str, Any]:
    if items is None:
        return {}
    if isinstance(items, dict):
        return dict(items)

    out: Dict[str, Any] = {}
    for item in items or []:
        if isinstance(item, dict):
            out.update(item)
    return out


def _safe_int(value: Any, default: int) -> int:
    """Parse ints from YAML values like 40 or '5 * 8'."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return int(ast.literal_eval(raw))
        except (ValueError, SyntaxError):
            pass
        allowed_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.FloorDiv: operator.floordiv,
            ast.Div: operator.truediv,
        }

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
                return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -_eval(node.operand)
            raise ValueError(f"unsupported expression: {raw}")

        try:
            return int(_eval(ast.parse(raw, mode="eval")))
        except Exception:
            return default
    return int(value)


@dataclass
class TrainConfig:
    # training
    mode: str = "pretrain"
    train_batch_size: int = 8
    block_size: int = 4096
    gradient_accumulation_steps: int = 40
    max_iters: int = 600_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    moe_aux_loss_weight: float = 0.01
    decay_lr: bool = True
    warmup_iters: int = 1500
    lr_decay_iters: int = 600_000
    min_lr: float = 6e-5
    compile_model: bool = False
    dtype: str = "bfloat16"
    use_cce: bool = False
    init_from: str = "hf"  # hf | resume
    resume_run_dir: Optional[str] = None
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 100
    eval_iters: int = 200
    display_model_output_iter: int = 0  # 0 disables periodic sample generation
    eval_only: bool = False
    always_save_checkpoint: bool = True
    hf_chkpt_path: Optional[str] = None
    seed: int = 42
    world_size: int = 1
    rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"

    # model / tokenizer
    model_name: str = ""
    tokenizer_name: str = ""
    tokenizer_num_proc: int = 8
    process_one_file_at_a_time: bool = True

    # data
    datasets: List[str] = field(default_factory=list)
    train_data_paths: List[str] = field(default_factory=list)
    eval_data_path: str = "data/val.bin"
    overwrite_data: bool = False

    # data sampling (mixing train_data_paths[0]=eng / [1]=afr into each batch)
    use_scheduled_sampling: bool = False
    eng_sampling_weight: float = 0.5
    afr_sampling_weight: float = 0.5

    # s3
    s3_endpoint: Optional[str] = None
    s3_bucket: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_prefix: str = ""

    # wandb
    wandb_log: bool = False
    wandb_project: str = "sabiyarn"
    wandb_run_name: str = "run"
    save_model_to_wandb: bool = False

    # ddp / modal
    ddp_backend: str = "nccl"
    gpus_per_node: int = 1
    num_nodes: int = 1
    mixed_precision: str = "bf16"

    # accelerate / fsdp
    fsdp_sharding_strategy: str = "FULL_SHARD"  # NO_SHARD | SHARD_GRAD_OP | FULL_SHARD | HYBRID_SHARD
    gradient_clipping: float = 1.0

    # weight-freeze policy
    freeze_experts_only: bool = False
    freeze_pos_layer_only: bool = False
    freeze_emb_layer_only: bool = False
    freeze_router_layer_only: bool = False
    freeze_ffn_layer_only: bool = False
    freeze_attn_layer_only: bool = False

    @property
    def is_sft(self) -> bool:
        return self.mode.lower() == "sft"

    @property
    def is_pretrain(self) -> bool:
        return self.mode.lower() == "pretrain"


def sampling_weights(
    eng_weight: float,
    afr_weight: float,
    iter_num: int,
    max_iters: int,
    use_scheduled_sampling: bool,
) -> tuple[float, float]:
    """(eng_weight, afr_weight) for picking between the eng/afr training bins.

    Fixed mode holds the configured preset (normalized to sum to 1) for the whole
    run. Scheduled mode starts at that preset and cosine-anneals toward the
    swapped ratio by max_iters -- early training leans on whichever language
    starts heavier (typically English, for linguistic grounding) and gradually
    shifts sampling weight onto the other as training progresses.
    """
    total = eng_weight + afr_weight
    eng0, afr0 = (0.5, 0.5) if total <= 0 else (eng_weight / total, afr_weight / total)
    if not use_scheduled_sampling:
        return eng0, afr0
    progress = min(1.0, iter_num / max(1, max_iters))
    coeff = 0.5 * (1.0 - math.cos(math.pi * progress))  # 0 -> 1 over training
    eng_w = eng0 + (afr0 - eng0) * coeff
    return eng_w, 1.0 - eng_w


def load_train_config(path: Optional[str] = None) -> TrainConfig:
    if path is None:
        path = os.environ.get(
            "TRAIN_CONFIG_PATH",
            str(Path(__file__).parent / "train_config.yaml"),
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(_normalize_list_sections(f.read())) or {}

    training = raw.get("training", {}) or {}
    optimizer = raw.get("optimizer", {}) or {}
    wandb_cfg = raw.get("wandb", {}) or {}
    data = raw.get("data", {}) or {}
    ddp = raw.get("ddp", {}) or {}
    accelerate = raw.get("accelerate", {}) or {}
    env = raw.get("env", {}) or {}
    modal_cfg = raw.get("modal", {}) or {}

    tokenizer = _merge_list_section(raw.get("tokenizer"))
    model_cfg = _merge_list_section(raw.get("model"))
    weights_cfg = _merge_list_section(model_cfg.get("weights"))
    s3 = _nested_list_dict(raw.get("s3"))
    sampling_cfg = _nested_list_dict(data.get("sampling"))

    mode = str(os.environ.get("TRAIN_MODE", training.get("mode", "pretrain"))).lower()
    mode_data = _nested_list_dict(data.get(mode, data.get("pretrain", [])))
    if not mode_data and mode != "pretrain":
        mode_data = _nested_list_dict(data.get("pretrain", []))

    # TRAIN_DATA_PATHS_LOCAL (comma-separated, same eng-then-afr order as below) is set
    # by modal_train.py after it downloads the yaml-configured S3 objects to local disk --
    # it must win over the yaml so training reads the local copies, not the S3 keys.
    env_train_paths = os.getenv("TRAIN_DATA_PATHS_LOCAL")
    if env_train_paths:
        train_paths = [p for p in env_train_paths.split(",") if p]
    else:
        train_paths = []
        for key in ("eng_train_data_path", "afr_train_data_path"):
            if key in mode_data:
                train_paths.append(mode_data[key])
        if not train_paths and "train_data_path" in data:
            train_paths.append(data["train_data_path"])
        if not train_paths:
            env_train = os.getenv("TRAIN_DATA_PATH")
            if env_train:
                train_paths.append(env_train)

    dataset_section = data.get(f"{mode}_datasets") or data.get("pretrain_datasets") or data.get("sft_datasets") or data.get("rl_datasets") or {}
    datasets: List[str] = []
    if isinstance(dataset_section, dict):
        for values in dataset_section.values():
            if isinstance(values, list):
                datasets.extend(values)
            elif isinstance(values, str):
                datasets.append(values)
    datasets = list(dict.fromkeys(datasets))

    eval_data_path = os.getenv("VAL_DATA_PATH") or str(mode_data.get("eval_data_path", data.get("eval_data_path", "data/val.bin")))
    mixed_precision = str(
        training.get("dtype", accelerate.get("mixed_precision", "bf16"))
    ).replace("fp16", "float16").replace("bf16", "bfloat16")

    return TrainConfig(
        mode=mode,
        train_batch_size=int(training.get("train_batch_size", 8)),
        block_size=int(training.get("block_size", 4096)),
        gradient_accumulation_steps=_safe_int(training.get("gradient_accumulation_steps"), 40),
        max_iters=int(optimizer.get("max_iters", training.get("max_iters", 600_000))),
        learning_rate=float(optimizer.get("learning_rate", 3e-4)),
        weight_decay=float(optimizer.get("weight_decay", 0.1)),
        beta1=float(optimizer.get("beta1", 0.9)),
        beta2=float(optimizer.get("beta2", 0.95)),
        grad_clip=float(optimizer.get("grad_clip", 1.0)),
        moe_aux_loss_weight=float(optimizer.get("moe_aux_loss_weight", 0.01)),
        decay_lr=bool(training.get("decay_lr", True)),
        warmup_iters=int(training.get("warmup_iters", 1500)),
        lr_decay_iters=int(training.get("lr_decay_iters", 600_000)),
        min_lr=float(training.get("min_lr", 6e-5)),
        compile_model=bool(training.get("compile", False)),
        dtype=mixed_precision,
        use_cce=bool(training.get("use_cce", False)),
        init_from=str(training.get("init_from", "hf")),
        resume_run_dir=training.get("resume_run_dir"),
        # TRAIN_OUT_DIR (set by modal_train.py to a path under the mounted Modal volume)
        # wins over the yaml so checkpoints persist and are visible to other containers
        # (eval, test_generation) instead of living in the training container's ephemeral disk.
        out_dir=str(os.getenv("TRAIN_OUT_DIR") or training.get("out_dir", "out")),
        eval_interval=int(training.get("eval_interval", 2000)),
        log_interval=int(training.get("log_interval", 100)),
        eval_iters=int(training.get("eval_iters", 200)),
        display_model_output_iter=int(training.get("display_model_output_iter", 0)),
        eval_only=bool(training.get("eval_only", False)),
        always_save_checkpoint=bool(training.get("always_save_checkpoint", True)),
        seed=int(wandb_cfg.get("seed", 42)),
        world_size=int(env.get("world_size", 1) or 1),
        rank=int(env.get("rank", 0) or 0),
        master_addr=str(env.get("master_addr", "127.0.0.1")),
        master_port=str(env.get("master_port", "29500")),
        model_name=str(model_cfg.get("repo_name") or model_cfg.get("name", "")),
        tokenizer_name=str(tokenizer.get("name", "")),
        tokenizer_num_proc=int(tokenizer.get("num_proc", 8)),
        process_one_file_at_a_time=bool(tokenizer.get("process_one_file_at_a_time", True)),
        datasets=datasets,
        train_data_paths=train_paths,
        eval_data_path=eval_data_path,
        overwrite_data=bool(data.get("overwrite_data", False)),
        use_scheduled_sampling=bool(sampling_cfg.get("use_scheduled_sampling", False)),
        eng_sampling_weight=float(sampling_cfg.get("eng_sampling_weight", 0.5)),
        afr_sampling_weight=float(sampling_cfg.get("afr_sampling_weight", 0.5)),
        s3_endpoint=s3.get("s3_endpoint") or os.getenv("S3_ENDPOINT"),
        s3_bucket=s3.get("s3_bucket_name") or os.getenv("S3_BUCKET"),
        s3_access_key=s3.get("s3_access_key_id") or os.getenv("S3_ACCESS_KEY_ID"),
        s3_secret_key=s3.get("s3_secret_access_key") or os.getenv("S3_SECRET_ACCESS_KEY"),
        s3_prefix=str(s3.get("prefix", "")),
        wandb_log=bool(wandb_cfg.get("log", False)),
        wandb_project=str(wandb_cfg.get("project", "sabiyarn")),
        wandb_run_name=str(wandb_cfg.get("run_name", "run")),
        save_model_to_wandb=bool(wandb_cfg.get("save_model_to_wandb", False)),
        hf_chkpt_path=training.get("hf_chkpt_path") or None,
        ddp_backend=str(ddp.get("backend", "nccl")),
        gpus_per_node=int(modal_cfg.get("gpus_per_node", env.get("world_size", 1))),
        num_nodes=int(modal_cfg.get("num_nodes", 1)),
        mixed_precision=mixed_precision,
        fsdp_sharding_strategy=str(accelerate.get("fsdp_sharding_strategy", "FULL_SHARD")).upper(),
        gradient_clipping=float(accelerate.get("gradient_clipping", training.get("grad_clip", 1.0))),
        freeze_experts_only=bool(weights_cfg.get("freeze_experts_only", False)),
        freeze_pos_layer_only=bool(weights_cfg.get("freeze_pos_layer_only", False)),
        freeze_emb_layer_only=bool(weights_cfg.get("freeze_emb_layer_only", False)),
        freeze_router_layer_only=bool(weights_cfg.get("freeze_router_layer_only", False)),
        freeze_ffn_layer_only=bool(weights_cfg.get("freeze_ffn_layer_only", False)),
        freeze_attn_layer_only=bool(weights_cfg.get("freeze_attn_layer_only", False)),
    )
