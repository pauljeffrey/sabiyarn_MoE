"""OpenRouter's batch API is NOT OpenAI-shaped. These pin the differences that cost real debugging time.

No network: the HTTP layer is stubbed, so this asserts the request we build, not the service.
"""
import json

import pytest

from providers.base import Request, _parse_batch_line
from providers.openrouter import OpenRouterProvider


@pytest.fixture()
def prov(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return OpenRouterProvider("openai/gpt-oss-120b:batch")


def _capture(prov, monkeypatch):
    seen = {}

    def fake(method, path, payload=None):
        seen["method"], seen["path"], seen["payload"] = method, path, payload
        return {"id": "batch-x", "status": "validating", "request_counts": {"total": 1}}

    monkeypatch.setattr(prov, "_http", fake)
    return seen


def test_batch_payload_is_inline_and_key_order_matters(prov, monkeypatch, tmp_path):
    """Their parser rejects the body unless `endpoint` and `model` appear BEFORE `requests`:
    "`endpoint` and `model` are required and must appear before `requests` in the request body"."""
    seen = _capture(prov, monkeypatch)
    prov.submit_batch([Request("a", [{"role": "user", "content": "hi"}], metadata={"lang": "yor"})], tmp_path)
    assert seen["method"] == "POST" and seen["path"] == "/batches"
    keys = list(seen["payload"])
    assert keys.index("endpoint") < keys.index("requests")
    assert keys.index("model") < keys.index("requests")
    assert seen["payload"]["endpoint"] == "/v1/chat/completions"
    # inline, not a file upload
    assert isinstance(seen["payload"]["requests"], list)


def test_model_is_batch_level_not_per_request(prov, monkeypatch, tmp_path):
    seen = _capture(prov, monkeypatch)
    prov.submit_batch([Request("a", [{"role": "user", "content": "hi"}])], tmp_path)
    assert seen["payload"]["model"] == "openai/gpt-oss-120b:batch"
    body = seen["payload"]["requests"][0]["body"]
    assert "model" not in body, "model must not be repeated in each request body"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_metadata_is_persisted_for_the_fetch(prov, monkeypatch, tmp_path):
    """Results come back hours later in a different process, so metadata must survive on disk."""
    _capture(prov, monkeypatch)
    prov.submit_batch([Request("a", [{"role": "user", "content": "hi"}], metadata={"lang": "hau"})], tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["a"]["lang"] == "hau"
    assert json.loads((tmp_path / "batch.json").read_text())["batch_id"] == "batch-x"


def test_oversized_batch_is_refused_with_a_way_forward(prov, monkeypatch, tmp_path):
    _capture(prov, monkeypatch)
    too_many = [Request(f"r{i}", [{"role": "user", "content": "x"}])
                for i in range(prov.MAX_REQUESTS_PER_BATCH + 1)]
    with pytest.raises(SystemExit, match="DATA_GEN_SHARDS"):
        prov.submit_batch(too_many, tmp_path)


def test_poll_retries_the_404_a_fresh_batch_returns(prov, monkeypatch):
    """A just-created batch 404s for a few seconds; without the retry that looks like a lost job."""
    calls = {"n": 0}

    def flaky(method, path, payload=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("GET /batches/x -> 404: not found")
        return {"id": "x", "status": "in_progress", "request_counts": {"total": 2, "completed": 1}}

    monkeypatch.setattr(prov, "_http", flaky)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    st = prov.poll_batch("x")
    assert st["status"] == "in_progress" and st["completed"] == 1 and calls["n"] == 3


def test_poll_does_not_swallow_a_real_error(prov, monkeypatch):
    monkeypatch.setattr(prov, "_http", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("-> 500: boom")))
    with pytest.raises(RuntimeError, match="500"):
        prov.poll_batch("x")


def test_results_parse_back_into_responses_with_metadata():
    row = {"custom_id": "a", "response": {"body": {
        "choices": [{"message": {"role": "assistant", "content": "Ghana"}}],
        "usage": {"prompt_tokens": 63, "completion_tokens": 46}, "model": "openai/gpt-oss-120b"}}}
    r = _parse_batch_line(row, {"a": {"lang": "pcm"}}, "openai/gpt-oss-120b:batch")
    assert r.ok and r.text == "Ghana" and r.metadata["lang"] == "pcm"
    assert r.prompt_tokens == 63 and r.completion_tokens == 46


def test_gemma_has_no_batch_tier_documented():
    """Recorded because it is the whole reason gemma must run synchronously or self-hosted: the batch
    endpoint answers "Model 'google/gemma-4-31b-it' does not have a :batch endpoint"."""
    src = (__import__("pathlib").Path(__file__).resolve().parents[1] / "providers/openrouter.py").read_text()
    assert "does not have a :batch endpoint" in src
    assert "vllm_gen.py" in src
