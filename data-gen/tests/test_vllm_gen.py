"""vLLM path: everything that does not need a GPU -- work ordering, schemas, presets, resumability."""
import os

import pytest

from schemas.output import schema_for
from schemas.seed import Seed
from vllm_gen import build_work, preset_for


def test_work_is_sorted_for_prefix_cache_hits():
    """The shared system prompt must arrive consecutively, or prefix caching buys nothing.

    build_work returns PLAN ROWS, not requests: the requests cannot be built until stage 1 has produced the
    documents that the long-document and RAG rows carry, and those run on the same engine."""
    from prompts import build_request
    rows, seed = build_work("sft", ["yor", "hau"], limit=400)
    assert rows and seed is not None
    reqs = [build_request(seed, r) for r in rows]
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
    rows, _ = build_work("pretrain", ["fon"], limit=50)
    assert rows and {r["lang"] for r in rows} == {"fon"}


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


def test_model_presets_cover_the_named_models():
    for m in ("openai/gpt-oss-120b", "google/gemma-3-27b-it", "google/gemma-4-31b-it"):
        p = preset_for(m)
        assert p["min_gpus_80gb"] >= 1 and p["notes"]
    # unknown models fall back rather than crash
    assert "kv_bytes_per_token" in preset_for("some/unreleased-model")


def test_gemma4_kv_geometry_matches_the_published_config():
    """These numbers decide how many sequences fit, which is the difference between saturating a rental and
    OOMing twenty minutes in. From google/gemma-4-31b-it's own config: 60 layers as 50 sliding_attention
    (window 1024) + 10 full_attention, num_global_key_value_heads 4, global_head_dim 512, and
    attention_k_eq_v so K and V share one tensor."""
    p = preset_for("google/gemma-4-31b-it")
    assert p["kv_bytes_per_token"] == 10 * 4 * 512 * 2          # full-attention layers only
    assert p["kv_bytes_fixed"] == 50 * 16 * 256 * 1024 * 2      # sliding layers, capped at the window
    assert 55 <= p["weight_gb_bf16"] <= 68


# --------------------------------------------------------------------------- hardware-agnostic loading


BOXES = [("RTX 5090", 32, 12.0), ("L40S 48GB", 48, 8.9), ("A100 40GB", 40, 8.0),
         ("GB10 119GB", 119, 12.0), ("A800 80GB", 80, 8.0), ("A100 80GB", 80, 8.0)]


def _gpus(gb, cap, n=1):
    return [{"index": i, "name": "x", "total_gb": float(gb), "capability": cap} for i in range(n)]


@pytest.mark.parametrize("name,gb,cap", BOXES)
def test_every_listed_box_gets_a_loadable_choice(name, gb, cap, capsys):
    """The point of `auto`: one command works on a 32 GB consumer card and on an 80 GB datacentre card with
    nothing edited. Anything that cannot load must raise with an instruction, never return silently."""
    from vllm_gen import choose_quantization, _fits
    q = choose_quantization("google/gemma-4-31b-it", None, _gpus(gb, cap), 1, 32_768)
    assert _fits("google/gemma-4-31b-it", q, float(gb)), (name, q)


def test_fp8_is_never_chosen_on_ampere():
    """Ampere (A100, A800, A40) has no fp8 kernels -- capability 8.0 < 8.9. Choosing it there is a load
    failure minutes into a paid rental."""
    from vllm_gen import choose_quantization
    for gb in (40, 80):
        for req in (None, "throughput"):
            assert choose_quantization("google/gemma-4-31b-it", req, _gpus(gb, 8.0), 1, 32_768) != "fp8"


def test_auto_is_quality_first_and_throughput_is_opt_in():
    """On an 80 GB Ampere card bf16 fits but leaves room for ~5 sequences, and 4-bit leaves room for ~40. That
    is a quality/throughput trade, so `auto` takes the precision and names the alternative rather than
    silently dropping to 4-bit on exactly the low-resource languages this corpus exists for."""
    from vllm_gen import choose_quantization
    g = _gpus(80, 8.0)
    assert choose_quantization("google/gemma-4-31b-it", None, g, 1, 32_768) is None
    assert choose_quantization("google/gemma-4-31b-it", "throughput", g, 1, 32_768) == "bitsandbytes"


def test_explicit_quantization_always_wins():
    from vllm_gen import choose_quantization
    g = _gpus(80, 8.0)
    assert choose_quantization("google/gemma-4-31b-it", "fp8", g, 1) == "fp8"
    assert choose_quantization("google/gemma-4-31b-it", "none", g, 1) is None
    assert choose_quantization("google/gemma-4-31b-it", "awq", g, 1) == "awq"


def test_a_prequantized_checkpoint_is_left_to_vllm():
    """AWQ and GPTQ cannot be applied on the fly, so a pre-quantized repo must not have a second scheme
    stacked on top of it."""
    from vllm_gen import choose_quantization
    for m in ("some/gemma-4-31b-AWQ", "some/gemma-4-31b-GPTQ-Int4", "some/model-FP8-dynamic"):
        assert choose_quantization(m, None, _gpus(80, 9.0), 1) is None


def test_a_model_too_big_for_the_box_fails_with_an_instruction():
    from vllm_gen import choose_quantization
    with pytest.raises(SystemExit, match="does not fit"):
        choose_quantization("google/gemma-4-31b-it", None, _gpus(8, 8.0), 1, 32_768)


def test_concurrency_falls_as_the_context_grows():
    from vllm_gen import max_sequences
    g = _gpus(80, 8.0)
    at16 = max_sequences("google/gemma-4-31b-it", g, 1, "bitsandbytes", 0.0, 16_384)
    at32 = max_sequences("google/gemma-4-31b-it", g, 1, "bitsandbytes", 0.0, 32_768)
    assert at16 > at32 >= 1


def test_the_per_sequence_sliding_window_cost_is_counted():
    """gemma-4 has 50 sliding-window layers whose KV is 419 MB per SEQUENCE regardless of length, on top of the
    per-token cost of its 10 full-attention layers. Ignoring it overstates concurrency by ~30% at 32k, which is
    an OOM twenty minutes into a paid rental rather than a rounding error."""
    from vllm_gen import kv_free_gib, max_sequences, preset_for
    g = _gpus(80, 8.0)
    free = kv_free_gib("google/gemma-4-31b-it", g, 1, "bitsandbytes", 0.0)
    p = preset_for("google/gemma-4-31b-it")
    naive = int(free * 2**30 / p["kv_bytes_per_token"]) // 32_768
    real = max_sequences("google/gemma-4-31b-it", g, 1, "bitsandbytes", 0.0, 32_768)
    assert real < naive, "the fixed per-sequence term is not being charged"
    # and the arithmetic is exactly free / (per_token * context + fixed)
    assert real == int(free * 2**30 // (p["kv_bytes_per_token"] * 32_768 + p["kv_bytes_fixed"]))


# --------------------------------------------------------------------------- resilience


def test_an_oom_splits_the_batch_instead_of_ending_the_run():
    """A generation run is hours long on a paid box. The failures that actually happen are a CUDA OOM when
    several long sequences land in one step, and one malformed conversation upsetting the batch. Neither may
    cost more than the samples involved."""
    from vllm_gen import _chat_resilient

    class Flaky:
        def __init__(self, limit):
            self.limit = limit
            self.calls = 0

        def chat(self, convos, params, use_tqdm=False):
            self.calls += 1
            if len(convos) > self.limit:
                raise RuntimeError("CUDA out of memory: tried to allocate 40.00 GiB")
            return [f"ok{i}" for i in range(len(convos))]

    eng = Flaky(limit=2)
    out = _chat_resilient(eng, [[{"role": "user", "content": "x"}]] * 8, None)
    assert len(out) == 8 and all(o is not None for o in out)
    assert eng.calls > 1, "the batch was never split"


def test_a_single_request_that_always_fails_is_dropped_not_raised():
    from vllm_gen import _chat_resilient

    class Broken:
        def chat(self, convos, params, use_tqdm=False):
            raise ValueError("malformed conversation")

    out = _chat_resilient(Broken(), [[{"role": "user", "content": "x"}]] * 4, None)
    assert out == [None, None, None, None]


# --------------------------------------------------------------------------- phases


@pytest.mark.parametrize("kind", ["pretrain", "sft", "rl"])
@pytest.mark.parametrize("context", [16_384, 32_768])
def test_every_phase_has_a_coherent_budget(kind, context):
    from budgets import budget_for
    b = budget_for(kind, context)
    assert b.context == context
    assert b.max_response_tokens < context
    lo, hi = b.doc_token_range
    assert 0 < lo <= hi < context
    assert b.max_output_tokens <= context
    assert b.rag_context_tokens == (2048, 8192)


# --------------------------------------------------------------------------- 8-bit policy


def test_auto_takes_fp8_over_bf16_when_it_buys_throughput():
    """On a GB10 (119 GiB, Blackwell) bf16 fits, but fp8 gives ~1.7x the sequences AND halves the bytes read
    per decode step, so ~3.5x the throughput. fp8's per-tensor W8A8 degradation is small and well
    characterised, which makes it worth taking automatically where 4-bit is not."""
    from vllm_gen import choose_quantization
    gb10 = _gpus(119, 12.0)
    assert choose_quantization("google/gemma-4-31b-it", None, gb10, 1, 8_192) == "fp8"
    assert choose_quantization("google/gemma-4-31b-it", None, gb10, 1, 32_768) == "fp8"
    # and it remains overridable
    assert choose_quantization("google/gemma-4-31b-it", "none", gb10, 1, 8_192) is None


def test_auto_never_takes_4bit_while_a_higher_precision_fits():
    """4-bit fluency on Fon or Efik is unmeasured here, so it is the operator's call, not a default."""
    from vllm_gen import choose_quantization
    for gpus in (_gpus(119, 12.0), _gpus(80, 8.0), _gpus(48, 8.9)):
        assert choose_quantization("google/gemma-4-31b-it", None, gpus, 1, 32_768) != "bitsandbytes"


def test_int8_and_8bit_are_accepted_as_fp8_aliases():
    """vLLM's only on-the-fly 8-bit path is fp8; W8A8-int8 needs a pre-quantized compressed-tensors
    checkpoint, so asking for `int8` and silently getting bf16 would be the wrong surprise."""
    from vllm_gen import choose_quantization
    g = _gpus(119, 12.0)
    for alias in ("int8", "8bit", "w8a8"):
        assert choose_quantization("google/gemma-4-31b-it", alias, g, 1) == "fp8"
    for alias in ("4bit", "nf4"):
        assert choose_quantization("google/gemma-4-31b-it", alias, g, 1) == "bitsandbytes"


def test_pretrain_gets_a_small_engine_window():
    """vLLM reserves KV per sequence at max_model_len, so the window must come from what a request needs --
    prompt plus completion -- not from the model's context. Every token saved is concurrency."""
    from budgets import budget_for
    from vllm_gen import max_sequences
    g = _gpus(119, 12.0)
    pre, sft = budget_for("pretrain"), budget_for("sft")
    assert pre.engine_len < 8_192 and sft.engine_len == 32_768
    # The window must fit the completion AND the ~1,100-token prompt; a 1,000-token document asking for a
    # 1,000-token window would not fit its own prompt.
    assert pre.engine_len >= pre.max_output_tokens + 1_500
    wide = max_sequences("google/gemma-4-31b-it", g, 1, None, 0.0, 32_768)
    narrow = max_sequences("google/gemma-4-31b-it", g, 1, None, 0.0, pre.engine_len)
    assert narrow >= wide * 2


def test_pretrain_context_sets_the_document_size():
    """`context` for pretraining IS the document's target size, because a pretraining sample is one document.
    Setting it must actually change the words asked for, per language."""
    from budgets import budget_for
    from prompts import build_request
    from schemas.seed import Seed
    seed = Seed.load("pretrain")

    def words(lang, ctx):
        import budgets
        saved = budgets._DEFAULT_BY_KIND["pretrain"]
        budgets._DEFAULT_BY_KIND["pretrain"] = ctx
        try:
            row = {"custom_id": f"pretrain__{lang}__doc__000001", "lang": lang, "task": "doc", "index": 1}
            return build_request(seed, row).metadata["target_words"]
        finally:
            budgets._DEFAULT_BY_KIND["pretrain"] = saved

    assert max(words("yor", 1_000)) < max(words("yor", 2_000))
    assert budget_for("pretrain", 1_000).pretrain_tokens == 1_000

    # Per-language conversion. Averaged over several rows, because each row also draws a length-variation
    # bucket seeded from its custom_id, and that spread (0.70-1.06) dominates any single comparison -- a single
    # draw made Yoruba look longer than Pidgin despite costing more tokens per word.
    def mean_words(lang, ctx, n=24):
        import budgets
        saved = budgets._DEFAULT_BY_KIND["pretrain"]
        budgets._DEFAULT_BY_KIND["pretrain"] = ctx
        try:
            tot = 0
            for i in range(n):
                row = {"custom_id": f"pretrain__{lang}__doc__{i:06d}", "lang": lang, "task": "doc", "index": i}
                lo, hi = build_request(seed, row).metadata["target_words"]
                tot += (lo + hi) / 2
            return tot / n
        finally:
            budgets._DEFAULT_BY_KIND["pretrain"] = saved

    # Fon costs 2.63 tokens/word against Pidgin's 1.24 AND is low-tier, so it gets far fewer words.
    assert mean_words("fon", 1_000) < mean_words("pcm", 1_000)
    assert mean_words("ewe", 1_000) < mean_words("hau", 1_000)


def test_a_pretrain_document_longer_than_block_size_is_clamped():
    """Pretraining trains at block_size 4,096, so a longer document is split across windows anyway. The
    reported band must never be a size the completion cap cannot deliver."""
    from budgets import budget_for
    b = budget_for("pretrain", 32_768)
    assert b.pretrain_tokens < 4_096
    assert b.doc_token_range[1] == b.pretrain_tokens
    assert "clamped" in b.notes


# --------------------------------------------------------------------------- sample inspection


def test_a_record_is_printable_without_dumping_a_20k_token_document():
    """One random kept record per chunk makes a multi-hour run watchable; printing a whole long-document sample
    would fill the terminal instead."""
    from inspect_sample import format_record
    rec = {"id": "sft__yor__long_document_summarization__000001", "lang": "yor",
           "io_direction": "english_to_native", "confidence": 0.95, "doc_words": 4300,
           "tasks": ["long_document_summarization"],
           "messages": [{"role": "user", "content": "Summarise: " + ("word " * 5000)},
                        {"role": "assistant", "content": "<|input_lang|><eng><response>Ìwé yìí..."}]}
    out = format_record(rec)
    assert "sft__yor__long_document_summarization__000001" in out
    assert "doc_words=4300" in out
    assert "Ìwé yìí" in out
    assert len(out) < 4_000, "a long document was dumped in full"
    assert "+" in out and "chars]" in out, "the elision is not reported"


def test_a_tool_call_and_its_result_are_both_shown():
    from inspect_sample import format_record
    rec = {"id": "x", "lang": "pcm", "messages": [
        {"role": "assistant", "content": "<|input_lang|><pcm>",
         "tool_calls": [{"function": {"name": "search_documents", "arguments": {"query": "cold chain"}}}]},
        {"role": "tool", "name": "search_documents", "content": "Passage 1: the fridge failed."},
        {"role": "assistant", "content": "<response>Di fridge spoil."}]}
    out = format_record(rec)
    assert "search_documents" in out and "cold chain" in out and "fridge failed" in out


def test_pretrain_records_print_their_prose():
    from inspect_sample import format_record
    out = format_record({"id": "p", "lang": "hau", "title": "Yadda ake noma", "text": "word " * 400},
                        kind="pretrain")
    assert "Yadda ake noma" in out and "400 words" in out


# --------------------------------------------------------------------------- --limit


def test_limit_keeps_every_language_represented():
    """A plain rows[:limit] is wrong on THIS path: it sorts by (lang, task) so identical prefixes arrive
    together, and truncating a language-sorted list takes only the first language. Measured: --limit 500 across
    eight languages returned 500 rows of Ewe and nothing else, so a pilot measured one language and the
    preflight had nothing to compare."""
    langs = ["pcm", "yor", "hau", "ibo", "twi", "ewe", "ful", "fuv"]
    for limit in (8, 56, 500):
        rows, _ = build_work("pretrain", langs, limit)
        assert len(rows) == limit
        got = {r["lang"] for r in rows}
        assert got == set(langs), (limit, sorted(set(langs) - got))


def test_limit_preserves_the_seeds_proportions():
    """The mix a pilot sees should be the mix the full run produces, or the pilot measures the wrong corpus."""
    import collections
    langs = ["pcm", "twi", "ewe"]
    rows, seed = build_work("pretrain", langs, 400)
    got = collections.Counter(r["lang"] for r in rows)
    planned = {l.code: l.samples for l in seed.languages if l.code in langs}
    total = sum(planned.values())
    for lg in langs:
        assert abs(got[lg] / 400 - planned[lg] / total) < 0.06, (lg, got[lg])


def test_limit_output_is_still_grouped_for_prefix_caching():
    rows, _ = build_work("pretrain", ["pcm", "yor", "hau"], 120)
    keys = [(r["lang"], r["task"]) for r in rows]
    assert keys == sorted(keys)


def test_context_is_free_within_the_models_own_limit():
    """Any value up to the model's block_size, because its learned absolute position embedding has no rows past
    32,768 -- a larger context would be untrainable rather than merely expensive."""
    import pytest as _pt
    from budgets import MAX_CONTEXT, budget_for
    assert MAX_CONTEXT == 32_768
    for ctx in (1_000, 1_024, 4_096, 32_768):
        assert budget_for("pretrain", ctx).context == ctx
    with _pt.raises(SystemExit, match="outside"):
        budget_for("pretrain", 40_000)
    with _pt.raises(SystemExit, match="outside"):
        budget_for("pretrain", 100)


def test_a_conversation_phase_refuses_a_context_too_small_to_hold_one():
    """A conversation carries a ~1,200-token tool catalogue plus several tool results before any answer, so a
    1,000-token context would produce nothing usable -- and silently, because every sample would simply fail
    assembly."""
    import pytest as _pt
    from budgets import budget_for
    for kind in ("sft", "rl"):
        with _pt.raises(SystemExit, match="too small"):
            budget_for(kind, 1_000)


# --------------------------------------------------------------------------- weights download


def test_a_full_disk_is_caught_before_the_engine_starts(tmp_path, monkeypatch):
    """vLLM downloads weights inside engine startup, so a disk problem surfaces as ~200 lines of engine-core
    traceback with the real cause on the last line:

        RuntimeError: File reconstruction error: Internal Writer Error: Background writer channel closed

    which is huggingface_hub's Xet backend reporting that the writer could not write. HF_HOME defaults to
    ~/.cache/huggingface -- the small root filesystem on a rented box -- so 62 GB of weights has nowhere to go.
    Checking first turns forty minutes of downloading into an instant, readable error.
    """
    from vllm_gen import ensure_weights
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    with pytest.raises(SystemExit) as exc:
        ensure_weights("google/gemma-4-31b-it", expected_gb=10_000_000)
    msg = str(exc.value)
    assert "Background writer channel closed" in msg      # names the symptom it prevents
    assert "export HF_HOME=" in msg                       # and the fix
    assert "Nothing has been spent on GPU time yet" in msg


def test_the_disk_check_passes_when_there_is_room(tmp_path, monkeypatch):
    """It must not block a box that is actually fine; the download itself is then attempted separately."""
    from vllm_gen import ensure_weights
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "huggingface_hub", None)   # skip the real download
    try:
        ensure_weights("google/gemma-4-31b-it", expected_gb=0.001)
    except SystemExit as exc:                                    # pragma: no cover
        pytest.fail(f"blocked a box with room: {exc}")


def test_the_suggested_mount_is_the_one_with_the_most_space():
    from vllm_gen import _biggest_writable_mount
    assert _biggest_writable_mount().startswith("/")
