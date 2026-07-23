import os

import pytest
from botocore.exceptions import ClientError

from training import s3_utils


class _FakePaginator:
    def __init__(self, keys):
        self.keys = keys

    def paginate(self, Bucket, Prefix="", Delimiter=""):
        matching = [k for k in self.keys if k.startswith(Prefix)]
        if Delimiter:
            seen_prefixes = set()
            contents = []
            for k in matching:
                rest = k[len(Prefix):]
                if Delimiter in rest:
                    seen_prefixes.add(Prefix + rest.split(Delimiter, 1)[0] + Delimiter)
                else:
                    contents.append({"Key": k})
            page = {"CommonPrefixes": [{"Prefix": p} for p in sorted(seen_prefixes)]}
            if contents:
                page["Contents"] = contents
            yield page
        else:
            yield {"Contents": [{"Key": k} for k in matching]}


class _FakeS3Client:
    def __init__(self, existing_keys=None, objects=None):
        self.existing_keys = set(existing_keys or [])
        self.uploaded = []
        self.downloaded = []
        self.objects = objects or {}  # key -> content, drives listing/download

    def head_object(self, Bucket, Key):
        if Key in self.existing_keys:
            return {}
        raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def upload_file(self, local_path, bucket, key):
        self.uploaded.append((local_path, bucket, key))
        self.existing_keys.add(key)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator(list(self.objects.keys()))

    def download_file(self, bucket, key, local_path):
        self.downloaded.append((bucket, key, local_path))
        content = self.objects.get(key, "data")
        with open(local_path, "w") as f:
            f.write(content)

    def get_object(self, Bucket, Key):
        content = self.objects.get(Key, "{}")
        return {"Body": _FakeBody(content.encode())}


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


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


def test_find_latest_remote_run_dir_picks_latest_matching_mode(monkeypatch):
    objects = {
        "checkpoints/out_280M/20260101_000000_pretrain/trainer_state.json": "{}",
        "checkpoints/out_280M/20260101_000000_pretrain/ckpt_100/config.json": "{}",
        "checkpoints/out_280M/20260722_000000_pretrain/trainer_state.json": "{}",
        "checkpoints/out_280M/20260722_000000_pretrain/ckpt_400/config.json": "{}",
        "checkpoints/out_280M/20260722_120000_sft/trainer_state.json": "{}",  # different mode
    }
    fake = _FakeS3Client(
        existing_keys={
            "checkpoints/out_280M/20260101_000000_pretrain/trainer_state.json",
            "checkpoints/out_280M/20260722_000000_pretrain/trainer_state.json",
            "checkpoints/out_280M/20260722_120000_sft/trainer_state.json",
        },
        objects=objects,
    )
    monkeypatch.setattr(s3_utils, "_s3_client", lambda *a, **k: fake)

    found = s3_utils.find_latest_remote_run_dir(
        "checkpoints/out_280M", "pretrain",
        bucket="b", endpoint="e", access_key="a", secret_key="s",
    )
    assert found == "checkpoints/out_280M/20260722_000000_pretrain/"


def test_find_latest_remote_run_dir_ignores_dirs_without_trainer_state(monkeypatch):
    objects = {
        "checkpoints/out_280M/20270101_000000_pretrain/ckpt_1/config.json": "{}",  # no trainer_state.json
        "checkpoints/out_280M/20260101_000000_pretrain/trainer_state.json": "{}",
    }
    fake = _FakeS3Client(
        existing_keys={"checkpoints/out_280M/20260101_000000_pretrain/trainer_state.json"},
        objects=objects,
    )
    monkeypatch.setattr(s3_utils, "_s3_client", lambda *a, **k: fake)

    found = s3_utils.find_latest_remote_run_dir(
        "checkpoints/out_280M", "pretrain",
        bucket="b", endpoint="e", access_key="a", secret_key="s",
    )
    assert found == "checkpoints/out_280M/20260101_000000_pretrain/"


def test_find_latest_remote_run_dir_returns_none_when_absent(monkeypatch):
    fake = _FakeS3Client(objects={})
    monkeypatch.setattr(s3_utils, "_s3_client", lambda *a, **k: fake)

    found = s3_utils.find_latest_remote_run_dir(
        "checkpoints/out_280M", "pretrain",
        bucket="b", endpoint="e", access_key="a", secret_key="s",
    )
    assert found is None


def test_download_folder_downloads_all_objects_mirroring_structure(tmp_path, monkeypatch):
    objects = {
        "checkpoints/run1/ckpt_100/config.json": '{"a": 1}',
        "checkpoints/run1/ckpt_100/nested/extra.txt": "x",
    }
    fake = _FakeS3Client(objects=objects)
    monkeypatch.setattr(s3_utils, "_s3_client", lambda *a, **k: fake)

    local_dir = tmp_path / "dest"
    downloaded = s3_utils.download_folder(
        "checkpoints/run1/ckpt_100", str(local_dir),
        bucket="b", endpoint="e", access_key="a", secret_key="s",
    )

    assert len(downloaded) == 2
    assert (local_dir / "config.json").read_text() == '{"a": 1}'
    assert (local_dir / "nested" / "extra.txt").read_text() == "x"


def test_download_folder_skips_existing_local_files(tmp_path, monkeypatch):
    objects = {"checkpoints/run1/ckpt_100/config.json": '{"a": 1}'}
    fake = _FakeS3Client(objects=objects)
    monkeypatch.setattr(s3_utils, "_s3_client", lambda *a, **k: fake)

    local_dir = tmp_path / "dest"
    local_dir.mkdir()
    (local_dir / "config.json").write_text("already here")

    s3_utils.download_folder(
        "checkpoints/run1/ckpt_100", str(local_dir),
        bucket="b", endpoint="e", access_key="a", secret_key="s",
    )

    assert fake.downloaded == []
    assert (local_dir / "config.json").read_text() == "already here"


def test_read_remote_json_returns_parsed_object(monkeypatch):
    objects = {"checkpoints/run1/trainer_state.json": '{"iter_num": 420, "best_val_loss": 3.2}'}
    fake = _FakeS3Client(existing_keys=set(objects.keys()), objects=objects)
    monkeypatch.setattr(s3_utils, "_s3_client", lambda *a, **k: fake)

    result = s3_utils.read_remote_json(
        "checkpoints/run1/trainer_state.json",
        bucket="b", endpoint="e", access_key="a", secret_key="s",
    )
    assert result == {"iter_num": 420, "best_val_loss": 3.2}


def test_read_remote_json_returns_none_when_missing(monkeypatch):
    fake = _FakeS3Client()
    monkeypatch.setattr(s3_utils, "_s3_client", lambda *a, **k: fake)

    result = s3_utils.read_remote_json(
        "checkpoints/run1/trainer_state.json",
        bucket="b", endpoint="e", access_key="a", secret_key="s",
    )
    assert result is None
