"""vLLM path: everything that does not need a GPU -- work ordering, schemas, presets, resumability."""
import os

import pytest

from schemas.output import schema_for
from schemas.seed import Seed
from vllm_gen import build_work, preset_for


def test_work_is_sorted_for_prefix_cache_hits():
    """The shared system prompt must arrive consecutively, or prefix caching buys nothing."""
    reqs, seed = build_work("sft", ["yor", "hau"], limit=400)
    assert reqs and seed is not None
    keys = [(r.metadata["lang"], r.metadata["task"]) for r in reqs]
    assert keys == sorted(keys), "work is not grouped by (lang, task)"
    # consecutive rows in a group share a byte-identical system message
    groups = {}
    for r in reqs:
        groups.setdefault((r.metadata["lang"], r.metadata["task"]), []).append(r.messages[0]["content"])
    reused = [g for g in groups.values() if len(g) > 1]
    assert reused, "no group had more than one row; cannot demonstrate prefix reuse"
    for g in reused:
        assert len(set(g)) == 1, "same (lang, task) produced different system prompts"


def test_work_respects_shard_partitioning(monkeypatch):
    monkeypatch.setenv("DATA_GEN_SHARDS", "4")
    totals = []
    for i in range(4):
        monkeypatch.setenv("DATA_GEN_SHARD_INDEX", str(i))
        reqs, _ = build_work("rl", None, limit=0)
        totals.append(len(reqs))
    monkeypatch.delenv("DATA_GEN_SHARDS")
    monkeypatch.delenv("DATA_GEN_SHARD_INDEX")
    full, _ = build_work("rl", None, limit=0)
    assert sum(totals) == len(full)
    assert max(totals) - min(totals) <= 1


def test_language_filter_is_honoured():
    reqs, _ = build_work("pretrain", ["fon"], limit=50)
    assert reqs and {r.metadata["lang"] for r in reqs} == {"fon"}


@pytest.mark.parametrize("kind", ["pretrain", "sft", "rl", "judge"])
def test_output_schemas_are_shallow_and_complete(kind):
    """Guided decoding compiles these; deep nesting blows up compile time for no gain."""
    s = schema_for(kind)
    assert s["type"] == "object" and s["required"]
    if kind != "judge":
        assert "confidence" in s["properties"]

    def depth(node, d=0):
        if not isinstance(node, dict):
            return d
        kids = [v for k, v in node.items() if k in ("properties", "items")]
        inner = []
        for k in kids:
            inner += list(k.values()) if isinstance(k, dict) and "type" not in k else [k]
        return max([depth(c, d + 1) for c in inner], default=d)

    assert depth(s) <= 6, f"{kind} schema is too deeply nested for grammar compilation"


def test_schema_accepts_a_realistic_sample():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate({
        "messages": [{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": "<|input_lang|><yor><think>ok</think>",
                      "tool_calls": [{"function": {"name": "search_internet", "arguments": {"query": "x"}}}]},
                     {"role": "tool", "name": "search_internet", "content": "result"},
                     {"role": "assistant", "content": "<response>answer"}],
        "tasks": ["tool_search_answer"], "tags": ["tool-calling"], "confidence": 0.8,
    }, schema_for("sft"))


def test_model_presets_cover_both_named_models():
    for m in ("openai/gpt-oss-120b", "google/gemma-3-27b-it"):
        p = preset_for(m)
        assert p["min_gpus_80gb"] >= 1 and p["max_model_len"] >= 4096 and p["notes"]
    # unknown models fall back rather than crash
    assert preset_for("some/unreleased-model")["max_model_len"] >= 4096
