"""Download training binaries from S3-compatible storage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import structlog

LOG = structlog.get_logger()


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
) -> list[str]:
    """Recursively upload every file under local_dir to S3, mirroring its
    internal structure under remote_prefix.

    Skips a file if its remote object already exists and override is False,
    so re-pushing a checkpoint (e.g. one still being written to) doesn't
    re-upload files that already made it up. Returns every resulting key
    (uploaded or already-present).
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

        if not override and _object_exists(client, bucket, key):
            LOG.info("s3_upload_skipped_exists", bucket=bucket, key=key)
            uploaded.append(key)
            continue

        LOG.info("uploading_s3_object", bucket=bucket, key=key, src=str(path))
        client.upload_file(str(path), bucket, key)
        uploaded.append(key)
    return uploaded


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
