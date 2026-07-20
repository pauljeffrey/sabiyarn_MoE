# # saves the openwebtext dataset to a binary file for training. following was helpful:
# # https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import argparse

# import tiktoken
from datasets import load_dataset, DownloadConfig
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import re
from huggingface_hub import list_repo_files, hf_hub_download
import json
from training import constant_tokens
import structlog
from dotenv import load_dotenv
from datasets import Dataset
import hashlib
import json
from pathlib import Path
import hashlib, lmdb
# remove cache for a specific dataset
# from datasets import set_caching_enabled

load_dotenv()

# set_caching_enabled(False)
LOG = structlog.stdlib.get_logger()

os.environ["TOKENIZERS_PARALLELISM"] = "true"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

config_path = os.environ.get(
    "TRAIN_CONFIG_PATH",
    os.path.join(project_root, "training", "train_config.yaml"),
)
config = OmegaConf.load(config_path)


def _merge_list_section(items) -> dict:
    """Merge YAML list sections like `[name, {k: v}, ...]`."""
    out = {}
    for item in items or []:
        if isinstance(item, str):
            out.setdefault("name", item)
        elif OmegaConf.is_dict(item) or isinstance(item, dict):
            out.update(OmegaConf.to_container(item, resolve=True))
    return out


def _nested_list_dict(items) -> dict:
    out = {}
    for item in items or []:
        if OmegaConf.is_dict(item) or isinstance(item, dict):
            out.update(OmegaConf.to_container(item, resolve=True))
    return out


_tokenizer_cfg = _merge_list_section(config.get("tokenizer"))
TRAINING_MODE = str(
    os.environ.get("TRAIN_MODE", config.get("training", {}).get("mode", "pretrain"))
).lower()

READ_TOKEN = os.getenv("HF_API_KEY")
num_proc = min(_tokenizer_cfg.get("num_proc", 8), os.cpu_count())
DATASET_REVISION = os.getenv("HF_DATASET_REVISION")  # Optional pin to commit/tag for stable cache keys

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though

DATASETS = config.data.datasets

PROCESS_ONE_FILE_AT_A_TIME = _tokenizer_cfg.get("process_one_file_at_a_time", True)

class HashRegistry:
    """Disk-backed hash registry using LMDB for billions of items."""
    def __init__(self, db_path: str, map_size_gb: int = 50):
        # map_size ~ maximum DB size. 50GB default; increase if needed.
        self.env = lmdb.open(
            db_path,
            map_size=map_size_gb * 1024 ** 3,
            subdir=True,
            max_dbs=1,
            readonly=False,
            lock=True,
            readahead=False,
            meminit=False
        )

    def add_if_new(self, h: bytes) -> bool:
        """Return True if new hash was added, False if it already existed."""
        with self.env.begin(write=True) as txn:
            if txn.get(h) is not None:
                return False
            txn.put(h, b"1", dupdata=False)
            return True

    def close(self):
        self.env.close()


def dedup_dataset_streaming(
    dataset,
    text_column,
    registry: HashRegistry,
    hash_algo="md5",
    key_fn=None,
):
    """Return only new rows by consulting the global registry."""
    hasher = getattr(hashlib, hash_algo)
    keep_idx = []
    for idx, rec in enumerate(dataset):
        text = key_fn(rec) if key_fn else rec[text_column].strip()
        h = hasher(text.encode("utf-8")).digest()
        if registry.add_if_new(h):
            keep_idx.append(idx)
    return dataset.select(keep_idx)

def get_tokenizer_and_eot(tokenizer_name):
    """Initializes and returns the tokenizer and end_of_text_token."""
    enc = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eot_token = enc.eos_token_id if enc.eos_token_id is not None else constant_tokens.end_of_text_token
    return enc, eot_token

def calculate_test_size(dataset_length):
    """Calculates the test split size based on dataset length."""
    if dataset_length <= 50: # Handle very small datasets, no test split
        return 0
    elif dataset_length <= 80000:
        return int(0.1 * dataset_length)
    elif dataset_length <= 800000:
        return int(0.01 * dataset_length)
    elif dataset_length <= 3500000:
        return int(0.0025 * dataset_length)
    else: # dataset_length > 3500000
        return int(0.0005 * dataset_length)

def _strip_leading_bos(ids):
    return ids[1:] if ids and ids[0] == 128000 else ids


def get_remove_columns(dset, mode: str):
    cols = set(dset.column_names)
    if mode == "sft":
        return [c for c in ("data", "messages", "text") if c in cols]
    return ["text"] if "text" in cols else []


def get_mode_data_paths(mode: str) -> dict:
    return _nested_list_dict(config.get("data", {}).get(mode, []))


def resolve_train_bin_path(dataset_name: str, mode: str, default_path: str) -> str:
    if mode != "sft":
        return default_path
    paths = get_mode_data_paths(mode)
    name_lower = dataset_name.lower()
    if "english" in name_lower or "eng" in name_lower:
        return paths.get("eng_train_data_path", default_path)
    if "african" in name_lower or "afr" in name_lower or "nigerian" in name_lower:
        return paths.get("afr_train_data_path", default_path)
    return default_path


def process_example(example, tokenizer, eot_token):
    """Tokenizes a single example from the pre-rendered text column."""
    ids = tokenizer.encode(example["text"])
    ids = _strip_leading_bos(ids)
    ids.extend([eot_token, eot_token])
    return {"ids": ids, "len": len(ids)}

def write_to_memmap(dset, filename, dtype, log_prefix="",):
    """
    Writes a dataset split to a memory-mapped file.
    """
    dtype = np.uint16
    
    # Add a length column if it doesn't exist
    LOG.info(f'column names: {dset.column_names}')
    # if "len" not in dset.column_names:
    dset = dset.map(lambda x: {"len": len(x["ids"])})
        
    arr_len = np.sum(dset["len"], dtype=np.uint64)

    # Check if file exists to determine starting index for appending
    current_file_size_bytes = 0
    if os.path.exists(filename):
        current_file_size_bytes = os.path.getsize(filename)

    # Calculate initial idx based on existing file size in terms of elements
    initial_idx = current_file_size_bytes // np.dtype(dtype).itemsize

    # Check actual elements in the file to get `initial_idx` for 'r+'
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        existing_arr = np.memmap(filename, dtype=dtype, mode="r")
        initial_idx = len(existing_arr)
        del existing_arr # Release file handle
        # Total size will be current size + new data size
        total_arr_len = initial_idx + arr_len
        arr = np.memmap(filename, dtype=dtype, mode="r+", shape=(total_arr_len,))
    else:
        initial_idx = 0
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

    LOG.info(f"{log_prefix} created/opened bin file (current size: {initial_idx} elements, new data: {arr_len} elements)...")

    n_shards = min(1024, len(dset)) # Use min to avoid n_shards > len(dset) when dset is small

    current_idx = initial_idx
    LOG.info(f"{log_prefix} writing to bin file from index {initial_idx}...")

    for batch_idx in tqdm(range(n_shards), desc=f"{log_prefix} writing {filename}"):
        batch = dset.shard(
            num_shards=n_shards, index=batch_idx, contiguous=True
        ).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])
        arr[current_idx : current_idx + len(arr_batch)] = arr_batch
        current_idx += len(arr_batch)
    arr.flush()
    LOG.info(f"{log_prefix} write to bin file complete...")

def remove_duplicate_samples(
    dataset: Dataset, text_column: str = "text", hash_algo: str = "md5") -> Dataset:
    """
    Stream through a HuggingFace Dataset and drop rows with duplicate text.

    - Uses only a set of short hashes in memory.
    - Processes each example exactly once.

    Args:
        dataset: HuggingFace Dataset
        text_column: Column name containing the text
        hash_algo: Hash function ('md5', 'sha1', etc.)

    Returns:
        Deduplicated Dataset
    """
    hasher = getattr(hashlib, hash_algo)
    seen = set()
    keep_idx = []

    # iterate without materializing the column
    for idx, record in enumerate(dataset):
        h = hasher(record[text_column].strip().encode("utf-8")).digest()
        if h not in seen:
            seen.add(h)
            keep_idx.append(idx)

    return dataset.select(keep_idx)


def remove_duplicate_samples_across_files(
    dataset: Dataset, text_column: str, hash_registry: set,  hash_algo: str = "md5",) -> Dataset:
    """
    Deduplicate dataset rows against a global hash_registry.

    Args:
        dataset: Hugging Face Dataset
        text_column: Column to deduplicate on
        hash_registry: set of seen hashes (modified in place)
        hash_algo: Hash function ('md5', 'sha1', etc.)

    Returns:
        Deduplicated Dataset
    """
    hasher = getattr(hashlib, hash_algo)
    keep_idx = []

    for idx, record in enumerate(dataset):
        text = record[text_column].strip()
        h = hasher(text.encode("utf-8")).digest()
        if h not in hash_registry:
            hash_registry.add(h)
            keep_idx.append(idx)

    return dataset.select(keep_idx)


def count_tag_occurrences(path: str, tag: str = "<twi>", tokenizer_name: str = None) -> int:
    """Count occurrences of a tag in a tokenized .bin file (np.uint16 memmap of token ids).

    The tag string is resolved to token id(s) via the tokenizer, then the memmap is
    scanned for that (sub)sequence. This operates on the actual binary token stream,
    not raw text -- the .bin files written by write_to_memmap() are not UTF-8 text.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0

    tok_name = tokenizer_name or _tokenizer_cfg.get("name", "BeardedMonster/SabiYarn-32k")
    enc, _ = get_tokenizer_and_eot(tok_name)
    tag_ids = enc.encode(tag, add_special_tokens=False)
    if not tag_ids:
        return 0

    arr = np.memmap(path, dtype=np.uint16, mode="r")
    if len(tag_ids) == 1:
        return int(np.count_nonzero(arr == tag_ids[0]))

    # Multi-token tag: count non-overlapping subsequence matches.
    tag_arr = np.array(tag_ids, dtype=np.uint16)
    matches = 0
    i = 0
    n, m = len(arr), len(tag_arr)
    while i <= n - m:
        if np.array_equal(arr[i : i + m], tag_arr):
            matches += 1
            i += m
        else:
            i += 1
    return matches


def run(
    datasets_list: list = DATASETS,
    data_files: dict = {},
    num_proc_load_dataset: int = num_proc,
    n_samples: int = 5_000_000,
    seed: int = 42,
    hash_algo: str = "md5",
    registry_cache: str = "global_hash_registry.lmdb",
    map_size_gb: int = 50,
):
    """
    Process, deduplicate (globally), tokenize and binarize many HF datasets.
    Dedup uses a disk-backed LMDB registry to handle billions of samples.
    """
    mode = str(
        os.environ.get("TRAIN_MODE", config.get("training", {}).get("mode", "pretrain"))
    ).lower()
    # ------- Setup paths and state -----------------------------------------
    env_train_path = os.getenv("TRAIN_DATA_PATH")
    env_val_path = os.getenv("VAL_DATA_PATH")
    default_train_path = getattr(config, "train_data_path", "data/train.bin")
    default_val_path = getattr(config.data, "eval_data_path", "data/val.bin")
    TRAIN_BIN_PATH = env_train_path or default_train_path
    VAL_BIN_PATH = env_val_path or default_val_path
    os.makedirs(os.path.dirname(TRAIN_BIN_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(VAL_BIN_PATH) or ".", exist_ok=True)

    state_path_env = os.getenv("PREP_STATE_PATH")
    STATE_PATH = state_path_env or os.path.join(os.path.dirname(TRAIN_BIN_PATH), "data_struct.json")

    num_proc_load_dataset = min(num_proc_load_dataset, max(1, os.cpu_count() - 1))

    override = os.environ.get("OVERRIDE_DATA", "0") == "1"
    if override:
        LOG.info("override_enabled", train_bin=TRAIN_BIN_PATH, val_bin=VAL_BIN_PATH)
        for p in (TRAIN_BIN_PATH, VAL_BIN_PATH, STATE_PATH):
            if os.path.exists(p):
                os.remove(p)
        files_processed = {}
    else:
        files_processed = {}
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r") as t:
                files_processed = json.load(t)

    tokenizer_name = _tokenizer_cfg.get("name", "BeardedMonster/SabiYarn-32k")
    tokenizer, eot = get_tokenizer_and_eot(tokenizer_name)
    process_fn = lambda example: process_example(example, tokenizer, eot)

    LOG.info("prepare_start", mode=mode, tokenizer=tokenizer_name)

    # ------- GLOBAL HASH REGISTRY (Disk-backed) ----------------------------
    try:
        registry = HashRegistry(db_path=registry_cache, map_size_gb=map_size_gb)
    except Exception as e:
        LOG.error("LMDB open failed: %s", e)
        raise

    # ------- Main loop -----------------------------------------------------
    for dataset_name in datasets_list:
        current_dataset_processed = files_processed.get(dataset_name, [])
        dataset_train_path = resolve_train_bin_path(dataset_name, mode, TRAIN_BIN_PATH)
        os.makedirs(os.path.dirname(dataset_train_path) or ".", exist_ok=True)

        if not PROCESS_ONE_FILE_AT_A_TIME:
            LOG.info(f"Loading dataset '{dataset_name}'…")
            LOG.info(f"Datafiles: {data_files.get(dataset_name)}")
            load_kwargs = dict(
                num_proc=num_proc_load_dataset,
                trust_remote_code=True,
                token=READ_TOKEN,
                verification_mode="no_checks",
            )
            if DATASET_REVISION:
                load_kwargs["revision"] = DATASET_REVISION

            ds = load_dataset(dataset_name,
                              data_files=data_files.get(dataset_name),download_mode="force_redownload",
                              **load_kwargs)["train"]

            ds = ds.shuffle(seed=seed)
            if n_samples != -1:
                ds = ds.select(range(min(n_samples, len(ds))))

            
            LOG.info(f"Dataset summary: {ds}…")
            # 1 GLOBAL DEDUP
            LOG.info(f"Deduplicating {dataset_name}…")
            ds = dedup_dataset_streaming(ds, "text", registry, hash_algo)
            if len(ds) == 0:
                LOG.warning(f"All samples in dataset '{dataset_name}' were duplicates. Skipping...")
                continue

            # Split, tokenize, save
            test_size = calculate_test_size(len(ds))
            split_ds = ds if test_size == 0 else ds.train_test_split(test_size=test_size, seed=2357)
            if test_size != 0:
                split_ds["val"] = split_ds.pop("test")
            else:
                split_ds = {"train": ds, "val": ds}

            remove_cols = get_remove_columns(ds, mode)
            tokenized = split_ds.map(
                process_fn,
                remove_columns=remove_cols,
                desc=f"tokenizing {dataset_name}",
                num_proc=num_proc,
            )

            for split, d in tokenized.items():
                out_file = dataset_train_path if split == "train" else VAL_BIN_PATH
                LOG.info(f"Writing {split} with {d} to {out_file} …")

                write_to_memmap(d, out_file, np.uint16, log_prefix=f"[{dataset_name} - {split}]")

            all_files = list_repo_files(dataset_name, repo_type="dataset", token=READ_TOKEN)
            files_processed[dataset_name] = list(set(current_dataset_processed + all_files))

        else:
            # ---------- One-file-at-a-time branch ----------
            all_files = data_files.get(dataset_name) or list_repo_files(dataset_name, repo_type="dataset", token=READ_TOKEN)
            files_to_process = [f for f in all_files if f not in current_dataset_processed and f.endswith(".parquet")]

            for fpath in files_to_process:
                LOG.info(f"Processing {fpath} …")
                local_path = hf_hub_download(dataset_name, filename=fpath,
                                            repo_type="dataset",
                                            revision=DATASET_REVISION or "main",
                                            token=READ_TOKEN,
                                            force_download=True, )
                dset = load_dataset("parquet", data_files={"train": local_path})["train"]
                LOG.info(f"Datafiles: {data_files.get(dataset_name)}")

                LOG.info(f"Dataset summary: {dset}…")

                # 1️⃣ GLOBAL DEDUP for this shard
                dset = dedup_dataset_streaming(dset, "text", registry, hash_algo)

                if len(dset) == 0:
                    LOG.warning(f"All samples in file '{fpath}' were duplicates. Skipping...")
                    continue
                test_size = calculate_test_size(len(dset))
                if len(dset) < 50 or test_size == 0:
                    split_d = {"train": dset, "val": dset}
                else:
                    tmp = dset.train_test_split(test_size=test_size, seed=2357)
                    split_d = {"train": tmp["train"], "val": tmp["test"]}

                remove_cols = get_remove_columns(dset, mode)
                tokenized = {
                    k: v.map(
                        process_fn,
                        remove_columns=remove_cols,
                        num_proc=num_proc,
                        desc=f"tokenizing {fpath} {k}",
                    )
                    for k, v in split_d.items()
                }

                for split, d in tokenized.items():
                    out_file = dataset_train_path if split == "train" else VAL_BIN_PATH
                    LOG.info(f"Writing {split} with {d} to {out_file} …")
                    write_to_memmap(
                        d, out_file, np.uint16, log_prefix=f"[{dataset_name}:{fpath}-{split}]"
                    )

                current_dataset_processed.append(fpath)
                files_processed[dataset_name] = current_dataset_processed
                with open(STATE_PATH, "w") as fp:
                    json.dump(files_processed, fp, indent=2)

        # Save ledger after each dataset
        with open(STATE_PATH, "w") as fp:
            json.dump(files_processed, fp, indent=2)
        LOG.info(f"Finished dataset {dataset_name}")

    registry.close()
    LOG.info("All datasets processed and globally deduplicated.")

    _push_outputs_to_s3(TRAIN_BIN_PATH, VAL_BIN_PATH, STATE_PATH, override=override)


def _push_outputs_to_s3(train_path: str, val_path: str, state_path: str, override: bool) -> None:
    """Upload processed .bin files + the processed-files ledger to S3, if S3 is configured."""
    s3_cfg = _nested_list_dict(config.get("s3", {}))
    bucket = s3_cfg.get("s3_bucket_name") or os.getenv("S3_BUCKET")
    endpoint = s3_cfg.get("s3_endpoint") or os.getenv("S3_ENDPOINT")
    access_key = s3_cfg.get("s3_access_key_id") or os.getenv("S3_ACCESS_KEY_ID")
    secret_key = s3_cfg.get("s3_secret_access_key") or os.getenv("S3_SECRET_ACCESS_KEY")
    if not (bucket and endpoint and access_key and secret_key):
        LOG.info("s3_push_skipped", reason="s3 not configured (missing bucket/endpoint/credentials)")
        return

    from training.s3_utils import upload_if_absent

    prefix = str(s3_cfg.get("prefix", ""))
    for local_path in (train_path, val_path, state_path):
        if not local_path or not os.path.exists(local_path):
            continue
        upload_if_absent(
            local_path,
            os.path.basename(local_path),
            bucket=bucket,
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            prefix=prefix,
            override=override,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=os.environ.get("TRAIN_MODE", TRAINING_MODE), choices=["pretrain", "sft", "rl"])
    parser.add_argument("--data-type", default="eng", choices=["eng", "african"])
    parser.add_argument("--tag", default="<twi>")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    os.environ["TRAIN_MODE"] = args.mode
    os.environ["OVERRIDE_DATA"] = "1" if args.override else "0"
    if args.data_type == "eng":
        datasets = config.get("data", {}).get(f"{args.mode}_datasets", {}).get("english", [])
    else:
        datasets = config.get("data", {}).get(f"{args.mode}_datasets", {}).get("african", [])
    if datasets:
        run(datasets_list=datasets)
    LOG.info("data tokenization and processing complete.....")

    train_path = os.getenv("TRAIN_DATA_PATH") or ""
    val_path = os.getenv("VAL_DATA_PATH") or ""
    if train_path:
        print(f"tag_count:{args.tag}:{count_tag_occurrences(train_path, args.tag)}")
    if val_path:
        print(f"tag_count:{args.tag}:{count_tag_occurrences(val_path, args.tag)}")
