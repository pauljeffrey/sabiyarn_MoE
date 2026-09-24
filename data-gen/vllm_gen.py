#!/usr/bin/env python3
"""Generate the corpus yourself with vLLM on a rented GPU, instead of paying per token.

    python vllm_gen.py --kind sft --model openai/gpt-oss-120b --limit 200        # pilot
    python vllm_gen.py --kind pretrain --model google/gemma-3-27b-it --tp 2
    python vllm_gen.py --kind rl --model openai/gpt-oss-120b --push
    python vllm_gen.py --kind judge --model google/gemma-3-27b-it --push         # judge the RL candidates
    python vllm_gen.py --kind sft --plan-only                                    # cost model, no GPU needed

WHY THIS IS CHEAPER HERE, SPECIFICALLY
--------------------------------------
42-77% of every prompt in this pipeline is a byte-identical prefix: the seed brief, the special-token format
rules and the tool catalogue are the same for every row of the same (kind, language). Measured:

    pretrain   852 of 1106 prompt tokens shared   (77%)
    sft       1208 of 2587                        (47%)
    rl        1186 of 2796                        (42%)

On a per-token API you pay for that prefix on every single request -- ~592M input tokens for SFT alone. vLLM
computes it ONCE per group and reuses the KV cache, so this script:
  * turns on `enable_prefix_caching`, and
  * SORTS the work by (kind, language, task) so identical prefixes arrive consecutively and actually hit the
    cache instead of being evicted between rows.
That is the whole reason self-hosting wins on this workload rather than being a wash.

The second win is grammar-constrained decoding. `--guided` (default on) compiles the output JSON schema into
a decoding constraint, so malformed JSON is structurally impossible rather than merely discouraged. On the API
path `json_invalid` is the largest drop reason; here it goes to zero, which raises effective yield and means
you generate fewer wasted samples.

WHAT IT SHARES WITH THE API PATH
--------------------------------
Everything except the transport: the same seeds, the same `prompts.build_request`, the same
`postprocess_gen.to_record` validation, the same shard files, the same `custom_id` resumability, the same
`DATA_GEN_SHARDS` partitioning and the same `hub.push_shards`. A corpus half-generated on Together and half
on your own GPU is one corpus, and rows are interchangeable.

COST
----
The script measures its own throughput and prints $/1M output tokens as it goes, so you can compare against
Together's $0.300/1M batch rate with real numbers instead of estimates. `--gpu-cost` is your hourly rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate import OUT_ROOT, ShardWriter, done_ids, plan_rows  # noqa: E402
from prompts import build_request                                # noqa: E402
from providers.base import Request, Response                     # noqa: E402
from schemas.output import schema_for                            # noqa: E402
from schemas.seed import Seed                                    # noqa: E402

# Recommended engine settings per model. `min_gpus_80gb` is what the weights need before any KV cache, so
# treat it as a floor, not a target -- throughput improves a lot with room for a big cache.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-120b": {
        # ~117B total but only ~5B active per token (MoE), and the release weights are MXFP4, so it is far
        # lighter and faster than the parameter count suggests: ~60GB, fits one 80GB card.
        "min_gpus_80gb": 1, "max_model_len": 8192, "gpu_memory_utilization": 0.92,
        "notes": "MoE, ~5B active params. Needs a recent vLLM (>=0.10) for the MXFP4 checkpoint.",
    },
    "google/gemma-3-27b-it": {
        # Dense 27B: ~54GB in bf16. One 80GB card, or two smaller ones with --tp 2.
        "min_gpus_80gb": 1, "max_model_len": 8192, "gpu_memory_utilization": 0.90,
        "notes": "Dense. Strong in several West African languages, which is why it is worth A/B-ing here.",
    },
}
DEFAULT_PRESET = {"min_gpus_80gb": 1, "max_model_len": 8192, "gpu_memory_utilization": 0.90, "notes": ""}


def preset_for(model: str) -> dict[str, Any]:
    for key, val in MODEL_PRESETS.items():
        if key.lower() in model.lower() or model.lower() in key.lower():
            return val
    return DEFAULT_PRESET


# --------------------------------------------------------------------------- work list


def _judge_requests(langs: Optional[list[str]], limit: int) -> list[Request]:
    from judge_gen import JUDGED_KIND, build_judge_request, _iter_rl_records

    seed = Seed.load("rl")
    already = done_ids(JUDGED_KIND, langs)
    out = []
    for rec in _iter_rl_records(langs):
        if rec["id"] in already:
            continue
        r = build_judge_request(rec, seed)
        if r:
            out.append(r)
        if limit and len(out) >= limit:
            break
    return out


def build_work(kind: str, langs: Optional[list[str]], limit: int) -> tuple[list[Request], Optional[Seed]]:
    """Requests still outstanding, SORTED so identical prompt prefixes are adjacent (prefix-cache hits)."""
    if kind == "judge":
        reqs = _judge_requests(langs, limit)
        # judge prompts share the criteria block; group by language so the rest lines up too
        reqs.sort(key=lambda r: (r.metadata.get("record", {}).get("lang", ""), r.custom_id))
        return reqs, None

    seed = Seed.load(kind)
    rows = list(plan_rows(seed, langs))
    already = done_ids(kind, langs)
    todo = [r for r in rows if r["custom_id"] not in already]

    n_shards = max(1, int(os.environ.get("DATA_GEN_SHARDS", "1")))
    shard_index = int(os.environ.get("DATA_GEN_SHARD_INDEX", "0"))
    if n_shards > 1:
        todo = todo[shard_index::n_shards]

    # THE important line: group by (lang, task) so the shared system prompt is computed once per group.
    # Do NOT shuffle here -- shuffling is what destroys prefix-cache locality and it is the default on the
    # API path only because there the prefix costs the same either way.
    todo.sort(key=lambda r: (r["lang"], r["task"], r["index"]))
    if limit:
        todo = todo[:limit]
    return [build_request(seed, r) for r in todo], seed


# --------------------------------------------------------------------------- engine


def load_engine(model: str, *, tp: int, max_model_len: int, gpu_mem: float, quantization: Optional[str],
                seed: int = 0):
    try:
        from vllm import LLM
    except ImportError:
        raise SystemExit(
            "vLLM is not installed. On a GPU box:\n"
            "    pip install vllm\n"
            "Nothing else in data-gen needs it -- the API path (generate.py) works without it, and\n"
            "`--plan-only` here works without a GPU.")
    p = preset_for(model)
    kw: dict[str, Any] = {
        "model": model,
        "tensor_parallel_size": tp,
        "max_model_len": max_model_len or p["max_model_len"],
        "gpu_memory_utilization": gpu_mem or p["gpu_memory_utilization"],
        # The reason this is cheaper than an API for this workload. Without it, the shared brief is
        # recomputed for every row.
        "enable_prefix_caching": True,
        "trust_remote_code": True,
        "seed": seed,
    }
    if quantization:
        kw["quantization"] = quantization
    if p["notes"]:
        print(f"[vllm] {model}: {p['notes']}")
    print(f"[vllm] loading with {json.dumps({k: v for k, v in kw.items() if k != 'model'})}")
    return LLM(**kw)


def sampling_params(kind: str, *, temperature: float, top_p: float, max_tokens: int, guided: bool):
    from vllm import SamplingParams

    kw: dict[str, Any] = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    if guided:
        schema = schema_for(kind)
        # vLLM moved this between releases: newer takes structured_outputs=, older guided_decoding=.
        try:
            from vllm.sampling_params import StructuredOutputsParams

            kw["structured_outputs"] = StructuredOutputsParams(json=schema)
        except ImportError:
            try:
                from vllm.sampling_params import GuidedDecodingParams

                kw["guided_decoding"] = GuidedDecodingParams(json=schema)
            except ImportError:
                print("[vllm] this build exposes no structured-output API; falling back to free-form JSON. "
                      "Expect a higher json_invalid drop rate (see postprocess_gen.summary()).")
    return SamplingParams(**kw)


# --------------------------------------------------------------------------- run


def run(kind: str, model: str, *, langs: Optional[list[str]], limit: int, tp: int, max_model_len: int,
        gpu_mem: float, quantization: Optional[str], chunk: int, temperature: float, top_p: float,
        max_tokens: int, guided: bool, push: bool, repo_id: str, gpu_cost: float,
        plan_only: bool) -> int:
    reqs, seed = build_work(kind, langs, limit)
    print(f"[{kind}] {len(reqs):,} requests outstanding for this worker")
    if not reqs:
        print("nothing to do")
        return 0

    in_tok_est = sum(len(m["content"]) for m in reqs[0].messages) / 4
    if plan_only:
        _plan_report(kind, reqs, in_tok_est, max_tokens, gpu_cost, model)
        return 0

    engine = load_engine(model, tp=tp, max_model_len=max_model_len, gpu_mem=gpu_mem,
                         quantization=quantization)
    params = sampling_params(kind, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                             guided=guided)

    from postprocess_gen import STATS, summary, to_record
    if kind == "judge":
        from judge_gen import JUDGED_KIND, apply_verdict
        writer = ShardWriter(JUDGED_KIND, _shard_tag())
    else:
        writer = ShardWriter(kind, _shard_tag())

    kept = failed = 0
    out_tokens = 0
    t0 = time.time()

    for start in range(0, len(reqs), chunk):
        batch = reqs[start:start + chunk]
        convos = [r.messages for r in batch]
        # One engine call per chunk: vLLM does continuous batching internally, so a big chunk is what
        # actually saturates the GPU. Chunking exists only so shards flush and progress is visible.
        outs = engine.chat(convos, params, use_tqdm=True)
        for req, out in zip(batch, outs):
            comp = out.outputs[0]
            out_tokens += len(comp.token_ids)
            resp = Response(req.custom_id, comp.text, True, None,
                            len(out.prompt_token_ids), len(comp.token_ids), model,
                            metadata=req.metadata)
            if kind == "judge":
                rec = apply_verdict(resp)
            else:
                rec = to_record(seed, resp)
            if rec is None:
                failed += 1
                continue
            writer.write(rec["lang"], rec)
            kept += 1
        dt = time.time() - t0
        tps = out_tokens / max(dt, 1e-9)
        done = start + len(batch)
        cost = gpu_cost * dt / 3600
        print(f"  [{kind}] {done:,}/{len(reqs):,}  kept {kept:,}  dropped {failed:,}  "
              f"{tps:,.0f} out-tok/s  {dt/60:.1f}m  ${cost:.2f} so far"
              + (f"  (${cost / (out_tokens / 1e6):.3f}/1M out tok)" if out_tokens > 1e5 else ""), flush=True)
        if STATS:
            print("   ", summary(), flush=True)

    paths = writer.close()
    dt = time.time() - t0
    total_cost = gpu_cost * dt / 3600
    yield_rate = kept / max(kept + failed, 1)
    print(f"\n[{kind}] kept {kept:,}  dropped {failed:,}  (yield {yield_rate:.1%})  in {dt/60:.1f}m")
    print(f"  {out_tokens:,} output tokens at {out_tokens/max(dt,1e-9):,.0f} tok/s")
    if out_tokens:
        per_m = total_cost / (out_tokens / 1e6)
        print(f"  GPU cost ${total_cost:.2f} = ${per_m:.3f} per 1M output tokens "
              f"(Together batch is $0.300, standard $0.600)")
        verdict = "CHEAPER than Together batch" if per_m < 0.300 else "more expensive than Together batch"
        print(f"  -> self-hosting is {verdict} at this throughput and GPU price")
    print(" ", summary())
    for p in paths:
        print(f"  wrote {p}")
    if push and paths:
        from hub import push_shards
        push_shards(JUDGED_KIND if kind == "judge" else kind, paths, repo_id=repo_id)
    return 0


def _shard_tag() -> Optional[str]:
    if os.environ.get("DATA_GEN_SHARDS", "1") == "1":
        return None
    import uuid
    return f"vllm-w{os.environ.get('DATA_GEN_SHARD_INDEX', '0')}-{uuid.uuid4().hex[:6]}"


def _plan_report(kind: str, reqs: list[Request], in_tok: float, max_tokens: int, gpu_cost: float,
                 model: str) -> None:
    """What this would cost at a few plausible throughputs. No GPU required."""
    n = len(reqs)
    # Shared-prefix share, measured from the actual first request.
    sysc = len(reqs[0].messages[0]["content"])
    totc = sum(len(m["content"]) for m in reqs[0].messages)
    print(f"\n=== {kind}: {n:,} requests, ~{in_tok:,.0f} input tokens each "
          f"({sysc/totc:.0%} of it a shared prefix, computed once per (lang, task) group)")
    print(f"    model {model}; preset: {preset_for(model)['notes'] or '(none)'}")
    # Apples to apples: the API bills INPUT too (and here input is huge), while the GPU bills wall-clock
    # regardless of token split. So compare total job cost, and report the throughput at which they cross.
    from estimate_gen import BATCH, STANDARD

    print(f"\n    {'out tok/sample':>14}{'total out':>12}{'API batch':>11}{'API std':>10}"
          f"{'  GPU @1k/2.5k/5k tok/s':>26}{'  break-even':>13}")
    for out_tok in (600, 1200, 2000, 2800):
        total_out = n * out_tok
        api_b = (n * in_tok / 1e6 * BATCH["input"]) + (total_out / 1e6 * BATCH["output"])
        api_s = (n * in_tok / 1e6 * STANDARD["input"]) + (total_out / 1e6 * STANDARD["output"])
        gpus = "/".join(f"${total_out / tps / 3600 * gpu_cost:,.0f}" for tps in (1000, 2500, 5000))
        # tok/s at which GPU cost equals the API batch price for the same work
        breakeven = total_out * gpu_cost / (3600 * api_b) if api_b else 0
        star = "  <-" if out_tok == 2000 else ""
        print(f"    {out_tok:>14,}{total_out/1e6:>11,.0f}M{'$' + format(api_b, ',.0f'):>11}"
              f"{'$' + format(api_s, ',.0f'):>10}{gpus:>26}{breakeven:>10,.0f}/s{star}")
    print(f"\n    Read the last column as: above that many output tokens/sec, your own GPU at ${gpu_cost:.2f}/hr")
    print(f"    beats Together's batch price for this job. Below it, the API is cheaper.")
    print(f"    Your $/1M output depends only on tok/s and $/hr. The break-even RISES with sample length")
    print(f"    because the API's per-request input cost amortises over more output, so self-hosting wins")
    print(f"    most clearly on SHORT outputs and on the prompt-heavy kinds.")
    print(f"\n    Throughput is the whole question and depends on your GPU, the model and batch size. Run a"
          f"\n    real --limit 200 first; the script prints measured tok/s and $/1M so you can decide with"
          f"\n    numbers rather than this table.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["pretrain", "sft", "rl", "judge"])
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--langs", default=None, help="comma list, e.g. yor,hau")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size = number of GPUs")
    ap.add_argument("--max-model-len", type=int, default=0, help="0 = the model preset")
    ap.add_argument("--gpu-mem", type=float, default=0.0, help="0 = the model preset")
    ap.add_argument("--quantization", default=None, help="e.g. fp8, awq, bitsandbytes")
    ap.add_argument("--chunk", type=int, default=2048,
                    help="requests per engine call; only affects flush cadence, not batching")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--no-guided", dest="guided", action="store_false",
                    help="disable grammar-constrained JSON (expect more json_invalid drops)")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo-id", default="BeardedMonster/data-gen")
    ap.add_argument("--gpu-cost", type=float, default=2.0, help="USD per hour for the whole box")
    ap.add_argument("--plan-only", action="store_true", help="cost model only; no GPU, no generation")
    a = ap.parse_args()
    if a.kind == "judge":
        a.temperature = 0.0  # a ranking should be reproducible
        a.max_tokens = min(a.max_tokens, 700)
    langs = [s.strip() for s in a.langs.split(",")] if a.langs else None
    return run(a.kind, a.model, langs=langs, limit=a.limit, tp=a.tp, max_model_len=a.max_model_len,
               gpu_mem=a.gpu_mem, quantization=a.quantization, chunk=a.chunk,
               temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, guided=a.guided,
               push=a.push, repo_id=a.repo_id, gpu_cost=a.gpu_cost, plan_only=a.plan_only)


if __name__ == "__main__":
    raise SystemExit(main())
