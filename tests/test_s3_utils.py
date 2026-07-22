import os

import pytest
from botocore.exceptions import ClientError

from training import s3_utils


class _FakeS3Client:
    def __init__(self, existing_keys=None):
        self.existing_keys = set(existing_keys or [])
        self.uploaded = []

    def head_object(self, Bucket, Key):
        if Key in self.existing_keys:
            return {}
        raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def upload_file(self, local_path, bucket, key):
        self.uploaded.append((local_path, bucket, key))
        self.existing_keys.add(key)


def _make_tree(tmp_path):
    (tmp_path / "ckpt_100").mkdir()
    (tmp_path / "ckpt_100" / "config.json").write_text("{}")
    (tmp_path / "ckpt_100" / "model.safetensors").write_text("weights")
    (tmp_path / "ckpt_100" / "nested").mkdir()
    (tmp_path / "ckpt_100" / "nested" / "extra.txt").write_text("x")
    return tmp_path / "ckpt_100"


def test_upload_folder_uploads_every_file_with_mirrored_keys(tmp_path, monkeypatch):
    local_dir = _make_tree(tmp_path)
    fake = _FakeS3Client()
    monkeypatch.setattr(s3_utils, "_s3_client", lambda *a, **k: fake)

    uploaded = s3_utils.upload_folder(
        str(local_dir), "checkpoints/run1/ckpt_100",
        bucket="b", endpoint="e", access_key="a", secret_key="s",
    )

    assert sorted(uploaded) == sorted([
        "checkpoints/run1/ckpt_100/config.json",
        "checkpoints/run1/ckpt_100/model.safetensors",
        "checkpoints/run1/ckpt_100/nested/extra.txt",
    ])
    assert len(fake.uploaded) == 3


def test_upload_folder_skips_existing_unless_override(tmp_path, monkeypatch):
    local_dir = _make_tree(tmp_path)
    fake = _FakeS3Client(existing_keys={"checkpoints/run1/ckpt_100/config.json"})
    monkeypatch.setattr(s3_utils, "_s3_client", lambda *a, **k: fake)

    s3_utils.upload_folder(
        str(local_dir), "checkpoints/run1/ckpt_100",
        bucket="b", endpoint="e", access_key="a", secret_key="s",
    )
    # config.json already existed -- should not have been re-uploaded.
    uploaded_keys = [key for (_, _, key) in fake.uploaded]
    assert "checkpoints/run1/ckpt_100/config.json" not in uploaded_keys
    assert "checkpoints/run1/ckpt_100/model.safetensors" in uploaded_keys


def test_upload_folder_override_reuploads_existing(tmp_path, monkeypatch):
    local_dir = _make_tree(tmp_path)
    fake = _FakeS3Client(existing_keys={"checkpoints/run1/ckpt_100/config.json"})
    monkeypatch.setattr(s3_utils, "_s3_client", lambda *a, **k: fake)

    s3_utils.upload_folder(
        str(local_dir), "checkpoints/run1/ckpt_100",
        bucket="b", endpoint="e", access_key="a", secret_key="s",
        override=True,
    )
    uploaded_keys = [key for (_, _, key) in fake.uploaded]
    assert "checkpoints/run1/ckpt_100/config.json" in uploaded_keys


def test_upload_folder_applies_bucket_prefix(tmp_path, monkeypatch):
    local_dir = _make_tree(tmp_path)
    fake = _FakeS3Client()
    monkeypatch.setattr(s3_utils, "_s3_client", lambda *a, **k: fake)

    uploaded = s3_utils.upload_folder(
        str(local_dir), "checkpoints/run1/ckpt_100",
        bucket="b", endpoint="e", access_key="a", secret_key="s",
        prefix="myprefix",
    )
    assert all(key.startswith("myprefix/checkpoints/run1/ckpt_100/") for key in uploaded)
