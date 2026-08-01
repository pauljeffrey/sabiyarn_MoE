"""Download training binaries from S3-compatible storage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Optional

import structlog

LOG = structlog.get_logger()


def is_mutable_checkpoint_file(rel_path: str) -> bool:
    """True for the parts of a run_dir that get REWRITTEN IN PLACE on every
    checkpoint save -- trainer_state.json and everything under
    resume_state/ (optimizer/RNG/FSDP shard state) -- as opposed to ckpt_N/,
    a brand new, immutable folder created fresh each save.

    These live at the same S3 key / local path across every save, so plain
    skip-if-exists (override=False / "already cached") silently freezes
    them at whatever content existed the FIRST time this run_dir was
    pushed/downloaded -- even as local training progresses far beyond that
    point and newer ckpt_N folders keep getting added correctly (since
    those have never-before-seen keys). Callers should force these to
    always be (re-)transferred regardless of the override/cache flag.

    ckpt_best/ and resume_state_best/ are the same kind of fixed-name,
    rewritten-in-place slot as resume_state/ -- unlike ckpt_N/, which is a
    brand new immutable folder each save -- so they need the same treatment.
    """
    return (
        rel_path == "trainer_state.json"
        or rel_path.startswith("resume_state/")
        or rel_path.startswith("resume_state_best/")
        or rel_path.startswith("ckpt_best/")
    )


def _s3_client(endpoint: str, access_key: str, secret_key: str):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _object_exists(client, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return False
        raise


def read_remote_json(
    key: str,
    *,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
) -> Optional[dict]:
    """Read and parse a single small JSON object from S3, without
    downloading anything else -- used to cheaply compare recency (e.g.
    iter_num) before committing to downloading a whole checkpoint. Returns
    None if the object doesn't exist."""
    import json

    client = _s3_client(endpoint, access_key, secret_key)
    if not _object_exists(client, bucket, key):
        return None
    obj = client.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def write_remote_json(
    key: str,
    data: dict,
    *,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
) -> None:
    """Overwrite a single small JSON object on S3 directly, unconditionally
    -- e.g. for manually correcting a run_dir's trainer_state.json (to point
    at a specific checkpoint/iter_num) without a full local round-trip."""
    import json

    client = _s3_client(endpoint, access_key, secret_key)
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(data).encode())


def delete_prefix(
    remote_prefix: str,
    *,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    prefix: str = "",
) -> list[str]:
    """Delete every object under remote_prefix on S3. Used to remove a
    run_dir's resume_state/ (optimizer/RNG state) when it no longer matches
    the checkpoint being resumed from -- e.g. warm-starting weights from a
    checkpoint whose own resume_state was never pushed, where reusing a
    different checkpoint's mismatched optimizer momentum under a new
    iter_num label would be worse than just starting the optimizer fresh.
    Returns every deleted key. Irreversible -- callers should be sure.
    """
    client = _s3_client(endpoint, access_key, secret_key)
    full_prefix = f"{prefix.rstrip('/')}/{remote_prefix.strip('/')}/" if prefix else f"{remote_prefix.strip('/')}/"

    deleted = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        keys = [obj["Key"] for obj in page.get("Contents", []) or []]
        if not keys:
            continue
        client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in keys]})
        for k in keys:
            LOG.info("deleted_s3_object", bucket=bucket, key=k)
        deleted.extend(keys)
    return deleted


def upload_if_absent(
    local_path: str,
    remote_key: str,
    *,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    prefix: str = "",
    override: bool = False,
) -> str:
    """Upload a local file to S3-compatible storage.

    Skips the upload if the remote object already exists and `override` is False,
    so re-running data prep doesn't clobber previously-pushed datasets unless asked.
    """
    key = f"{prefix.rstrip('/')}/{remote_key.lstrip('/')}" if prefix else remote_key.lstrip("/")
    client = _s3_client(endpoint, access_key, secret_key)

    if not override and _object_exists(client, bucket, key):
        LOG.info("s3_upload_skipped_exists", bucket=bucket, key=key)
        return key

    LOG.info("uploading_s3_object", bucket=bucket, key=key, src=local_path)
    client.upload_file(local_path, bucket, key)
    return key


def upload_folder(
    local_dir: str,
    remote_prefix: str,
    *,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    prefix: str = "",
    override: bool = False,
    force_override_paths: Optional[Callable[[str], bool]] = None,
) -> list[str]:
    """Recursively upload every file under local_dir to S3, mirroring its
    internal structure under remote_prefix.

    Skips a file if its remote object already exists and override is False,
    so re-pushing a checkpoint (e.g. one still being written to) doesn't
    re-upload files that already made it up. force_override_paths, if given,
    is checked (against the path relative to local_dir) before that skip --
    when it returns True the file is always re-uploaded regardless of
    override, for files like trainer_state.json that get rewritten in place
    (see is_mutable_checkpoint_file). Returns every resulting key (uploaded
    or already-present).
    """
    client = _s3_client(endpoint, access_key, secret_key)
    base = Path(local_dir)
    uploaded = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        remote_key = f"{remote_prefix.rstrip('/')}/{rel}"
        key = f"{prefix.rstrip('/')}/{remote_key.lstrip('/')}" if prefix else remote_key.lstrip("/")

        force = force_override_paths(rel) if force_override_paths else False
        if not override and not force and _object_exists(client, bucket, key):
            LOG.info("s3_upload_skipped_exists", bucket=bucket, key=key)
            uploaded.append(key)
            continue

        LOG.info("uploading_s3_object", bucket=bucket, key=key, src=str(path))
        client.upload_file(str(path), bucket, key)
        uploaded.append(key)
    return uploaded


def find_latest_remote_run_dir(
    remote_root: str,
    mode: str,
    *,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    prefix: str = "",
) -> Optional[str]:
    """S3 counterpart to training/new_train.py's local _find_latest_run_dir:
    lists "directories" (common prefixes) directly under remote_root, keeps
    only those ending in `_{mode}` that actually contain a trainer_state.json
    object, and returns the lexicographically-latest one's full remote key
    (run dir names are `{timestamp}_{mode}`, so this sorts chronologically).
    Returns None if none exist.
    """
    client = _s3_client(endpoint, access_key, secret_key)
    root_key = f"{prefix.rstrip('/')}/{remote_root.strip('/')}/" if prefix else f"{remote_root.strip('/')}/"

    candidates = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=root_key, Delimiter="/"):
        for common in page.get("CommonPrefixes", []) or []:
            run_key = common["Prefix"]
            name = run_key.rstrip("/").rsplit("/", 1)[-1]
            if not name.endswith(f"_{mode}"):
                continue
            if _object_exists(client, bucket, f"{run_key}trainer_state.json"):
                candidates.append(run_key)

    if not candidates:
        return None
    return sorted(candidates)[-1]


def list_immediate_subfolders(
    remote_root: str,
    *,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    prefix: str = "",
) -> list[str]:
    """List "directories" (common prefixes) directly under remote_root --
    e.g. every ckpt_N/ folder inside a run_dir. Returns full keys (each
    ending in a trailing slash)."""
    client = _s3_client(endpoint, access_key, secret_key)
    root_key = f"{prefix.rstrip('/')}/{remote_root.strip('/')}/" if prefix else f"{remote_root.strip('/')}/"

    out = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=root_key, Delimiter="/"):
        for common in page.get("CommonPrefixes", []) or []:
            out.append(common["Prefix"])
    return out


def download_folder(
    remote_prefix: str,
    local_dir: str,
    *,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    prefix: str = "",
    force_redownload_paths: Optional[Callable[[str], bool]] = None,
) -> list[str]:
    """Recursively download every object under remote_prefix into local_dir,
    mirroring its structure -- the download-side counterpart to
    upload_folder. Skips a file that already exists locally with nonzero
    size (same "cached" convention as download_if_missing).
    force_redownload_paths, if given, is checked (against the path relative
    to remote_prefix) before that skip -- when it returns True the file is
    always re-downloaded even if a local copy already exists, for files
    like trainer_state.json that get rewritten in place remotely (see
    is_mutable_checkpoint_file) and so can't be trusted to still match a
    previously-cached local copy.

    remote_prefix may already be a full key (e.g. as returned by
    find_latest_remote_run_dir, which already folds in `prefix`) -- pass
    prefix="" in that case to avoid applying it twice.
    """
    client = _s3_client(endpoint, access_key, secret_key)
    full_prefix = f"{prefix.rstrip('/')}/{remote_prefix.strip('/')}/" if prefix else f"{remote_prefix.strip('/')}/"

    downloaded = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            rel = key[len(full_prefix):]
            if not rel:
                continue
            local_path = os.path.join(local_dir, *rel.split("/"))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            force = force_redownload_paths(rel) if force_redownload_paths else False
            if not force and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                LOG.info("s3_download_skipped_cached", bucket=bucket, key=key)
                downloaded.append(local_path)
                continue

            LOG.info("downloading_s3_object", bucket=bucket, key=key, dest=local_path)
            client.download_file(bucket, key, local_path)
            downloaded.append(local_path)
    return downloaded


def download_if_missing(
    remote_key: str,
    local_path: str,
    *,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    prefix: str = "",
) -> str:
    """Download an S3 object to local_path if it does not exist."""
    local = Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size > 0:
        LOG.info("using_cached_file", path=str(local))
        return str(local)

    key = f"{prefix.rstrip('/')}/{remote_key.lstrip('/')}" if prefix else remote_key.lstrip("/")
    LOG.info("downloading_s3_object", bucket=bucket, key=key, dest=str(local))
    client = _s3_client(endpoint, access_key, secret_key)
    client.download_file(bucket, key, str(local))
    return str(local)


def sync_training_files(
    remote_paths: Iterable[str],
    local_dir: str,
    *,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    prefix: str = "",
    eval_path: Optional[str] = None,
) -> tuple[list[str], str]:
    """Download train + eval bins; return resolved local paths."""
    train_local = []
    for remote in remote_paths:
        name = os.path.basename(remote)
        local = os.path.join(local_dir, name)
        train_local.append(
            download_if_missing(
                remote,
                local,
                bucket=bucket,
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                prefix=prefix,
            )
        )

    eval_local = eval_path or "data/val.bin"
    eval_name = os.path.basename(eval_local)
    eval_resolved = download_if_missing(
        eval_local,
        os.path.join(local_dir, eval_name),
        bucket=bucket,
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        prefix=prefix,
    )
    return train_local, eval_resolved
