import json
import os

import pytest

from training import push_checkpoint_to_s3_modal as m


def _make_run_dir(out_dir, name, latest_ckpt_name=None, write_state=True):
    run_dir = out_dir / name
    run_dir.mkdir(parents=True)
    if write_state:
        ckpt = str(run_dir / latest_ckpt_name) if latest_ckpt_name else ""
        if latest_ckpt_name:
            (run_dir / latest_ckpt_name).mkdir()
        (run_dir / "trainer_state.json").write_text(
            json.dumps({"iter_num": 100, "best_val_loss": 1.0, "latest_ckpt": ckpt})
        )
    return run_dir


def test_find_latest_run_dir_picks_latest_matching_mode(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _make_run_dir(out_dir, "20260101_000000_pretrain", "ckpt_1")
    latest = _make_run_dir(out_dir, "20260722_000000_pretrain", "ckpt_2")
    _make_run_dir(out_dir, "20260722_120000_sft", "ckpt_3")  # different mode, ignored

    found = m._find_latest_run_dir(str(out_dir), "pretrain")
    assert found == str(latest)


def test_find_latest_run_dir_ignores_dirs_without_trainer_state(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "20270101_000000_pretrain").mkdir()  # no trainer_state.json
    valid = _make_run_dir(out_dir, "20260101_000000_pretrain", "ckpt_1")

    found = m._find_latest_run_dir(str(out_dir), "pretrain")
    assert found == str(valid)


def test_find_latest_run_dir_returns_none_when_absent(tmp_path):
    assert m._find_latest_run_dir(str(tmp_path / "nope"), "pretrain") is None


def test_resolve_checkpoint_dir_explicit_relative_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(m, "_load_yaml_cfg", lambda: {"training": {"out_dir": "out_280M"}})
    ckpt = tmp_path / "out_280M" / "run1" / "ckpt_5"
    ckpt.mkdir(parents=True)

    resolved = m._resolve_checkpoint_dir("pretrain", "run1/ckpt_5")
    assert os.path.normpath(resolved) == os.path.normpath(str(ckpt))


def test_resolve_checkpoint_dir_explicit_absolute_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(m, "_load_yaml_cfg", lambda: {"training": {"out_dir": "out_280M"}})
    somewhere = tmp_path / "elsewhere"
    somewhere.mkdir()

    resolved = m._resolve_checkpoint_dir("pretrain", str(somewhere))
    assert resolved == str(somewhere)


def test_resolve_checkpoint_dir_missing_folder_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(m, "_load_yaml_cfg", lambda: {"training": {"out_dir": "out_280M"}})

    with pytest.raises(FileNotFoundError):
        m._resolve_checkpoint_dir("pretrain", "does/not/exist")


def test_resolve_checkpoint_dir_auto_discovers_latest_ckpt(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(m, "_load_yaml_cfg", lambda: {"training": {"out_dir": "out_280M"}})
    out_dir = tmp_path / "out_280M"
    out_dir.mkdir()
    run_dir = _make_run_dir(out_dir, "20260722_000000_pretrain", "ckpt_800")

    resolved = m._resolve_checkpoint_dir("pretrain", "")
    assert resolved == str(run_dir / "ckpt_800")


def test_resolve_checkpoint_dir_no_run_dir_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(m, "_load_yaml_cfg", lambda: {"training": {"out_dir": "out_280M"}})

    with pytest.raises(FileNotFoundError):
        m._resolve_checkpoint_dir("pretrain", "")


def test_resolve_checkpoint_dir_invalid_latest_ckpt_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(m, "_load_yaml_cfg", lambda: {"training": {"out_dir": "out_280M"}})
    out_dir = tmp_path / "out_280M"
    out_dir.mkdir()
    _make_run_dir(out_dir, "20260722_000000_pretrain", latest_ckpt_name=None)

    with pytest.raises(FileNotFoundError):
        m._resolve_checkpoint_dir("pretrain", "")
