import json
import math
from pathlib import Path

import pytest

from training.tracking import LN2, MlflowTracker, bits_per_byte, build_token_byte_lengths


def test_byte_table_special_tokens_are_zero_and_padding_slot_exists():
    vocab = {"a": 0, "Ġthe": 1, "<|user|>": 2, "hello": 3}
    table = build_token_byte_lengths(vocab, special_ids={2})
    assert table == [1, 4, 0, 5, 0]  # trailing slot: ids past the vocab resolve to 0 bytes


def test_bits_per_byte_matches_hand_calculation():
    # 1 nat/token over 10 tokens covering 40 bytes -> 10 nats / (ln2 * 40) bits/byte
    assert bits_per_byte(1.0, 10, 40) == pytest.approx(10 / (LN2 * 40))
    assert bits_per_byte(1.0, 10, 0) is None


def test_real_tokenizer_vocab_gives_sane_bytes_per_token():
    cache = Path.home() / ".cache/huggingface/hub/models--BeardedMonster--SabiYarn-32k/snapshots"
    files = list(cache.glob("*/tokenizer.json"))
    if not files:
        pytest.skip("tokenizer not cached locally")
    tok = json.loads(files[0].read_text(encoding="utf-8"))
    vocab = tok["model"]["vocab"]
    special = {t["id"] for t in tok["added_tokens"]}
    table = build_token_byte_lengths(vocab, special)
    regular = [table[i] for i in vocab.values() if i not in special]
    assert all(b >= 1 for b in regular)
    assert 1.0 < sum(regular) / len(regular) < 15.0
    assert all(table[min(i, len(table) - 1)] == 0 for i in special)  # ids past the table clamp to the 0 slot


def test_tracker_is_noop_when_not_started():
    t = MlflowTracker()
    t.log_metrics({"x": 1.0}, step=0)  # must not raise
    t.end()


def test_tracker_logs_to_file_store(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = f"file:{tmp_path / 'mlruns'}"
    t = MlflowTracker()
    assert t.start(
        tracking_uri=uri, experiment_name="test", run_name="r", run_id=None,
        params={"a": 1, "nested": {"b": 2}},
    )
    t.log_metrics({"train/loss": 2.5, "train/bpb": 1.1, "bad": float("nan"), "none": None}, step=3)
    t.end()

    client = mlflow.tracking.MlflowClient(uri)
    run = client.search_runs([client.get_experiment_by_name("test").experiment_id])[0]
    assert run.data.params == {"a": "1", "nested.b": "2"}
    assert run.data.metrics == {"train/loss": 2.5, "train/bpb": 1.1}
    assert math.isclose(client.get_metric_history(run.info.run_id, "train/loss")[0].step, 3)


def test_tracker_survives_bad_tracking_uri(monkeypatch):
    pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")  # nothing listens here
    t = MlflowTracker()
    t.start(tracking_uri=None, experiment_name="x", run_name="r", run_id=None, params={})
    t.log_metrics({"a": 1.0}, step=0)  # must not raise whether or not start() succeeded
    t.end()


def test_start_ui_serves_and_stops(tmp_path, monkeypatch):
    import socket
    import time
    import urllib.request

    pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    t = MlflowTracker()
    assert t.start(tracking_uri=f"file:{tmp_path / 'mlruns'}", experiment_name="ui", run_name="r", run_id=None, params={})
    assert t.start_ui("127.0.0.1", port)
    try:
        for _ in range(60):
            try:
                assert urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1).status == 200
                break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.fail("mlflow ui never came up")
    finally:
        t.end()
    assert t._ui_proc is None


def test_start_ui_skipped_for_remote_server():
    t = MlflowTracker()
    t.enabled, t.uri = True, "http://example.com:5000"
    assert t.start_ui() is False
