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
import random
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
    "google/gemma-4-31b-it": {
        # Dense 30.8B (60 layers, hidden 5376, intermediate 21504, vocab 262144, tied embeddings), read from
        # the published config. bf16 weights are ~61.6 GB, so unquantized it needs an 80 GB card.
        #
        # Its KV cache is unusually cheap for long context, which is why 32k generation is affordable here:
        # layer_types is 50 sliding_attention (window 1024) + 10 full_attention, and attention_k_eq_v means K
        # and V share one tensor. So per sequence:
        #   full layers    10 x (4 kv heads x 512) x 2 B      = 40,960 B per TOKEN
        #   sliding layers 50 x (16 x 256) x 1024 x 2 B       = 419 MB FIXED, regardless of length
        # A 32,768-token sequence is therefore ~1.76 GB, not the ~30 GB a naive all-global estimate gives.
        "min_gpus_80gb": 1, "max_model_len": 0, "gpu_memory_utilization": 0.90,
        "kv_bytes_per_token": 40_960, "kv_bytes_fixed": 419_430_400, "weight_gb_bf16": 61.6,
        "notes": "dense 30.8B; 50/60 layers are sliding-window 1024, so long-context KV is ~1.8 GB at 32k",
    },
    "google/gemma-3-27b-it": {
        "min_gpus_80gb": 1, "max_model_len": 0, "gpu_memory_utilization": 0.90,
        "kv_bytes_per_token": 32_768, "kv_bytes_fixed": 419_430_400, "weight_gb_bf16": 54.0,
        "notes": ("dense 27B, also sliding-window. THE MODEL FOR LOW-RESOURCE PRETRAINING: 89% clean against "
                  "gemma-4-31b's 6% on fon/ewe/efi/urh/ful/fuv (measured 2026-09-29, unpacked, n=18 each)."),
    },
    "openai/gpt-oss-120b": {
        # ~117B total but only ~5B active per token (MoE), and the release weights are MXFP4, so it is far
        # lighter and faster than the parameter count suggests: ~60GB, fits one 80GB card.
        "min_gpus_80gb": 1, "max_model_len": 8192, "gpu_memory_utilization": 0.92,
        "notes": "MoE, ~5B active params. Needs a recent vLLM (>=0.10) for the MXFP4 checkpoint.",
    },
}
DEFAULT_PRESET = {"min_gpus_80gb": 1, "max_model_len": 0, "gpu_memory_utilization": 0.90,
                  "kv_bytes_per_token": 131_072, "kv_bytes_fixed": 0, "weight_gb_bf16": 0.0, "notes": ""}


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
        todo = _balanced_slice(todo, limit)
    return todo, seed


def _balanced_slice(rows: list[dict], limit: int) -> list[dict]:
    """Take `limit` rows while keeping every language represented, in the seed's proportions.

    A plain rows[:limit] is wrong HERE specifically. This path sorts by (lang, task) so identical prompt
    prefixes arrive together -- that is what makes prefix caching pay -- and truncating a language-sorted list
    takes only the first language: `--limit 500` across eight languages returned 500 rows of Ewe and nothing
    else, so a pilot measured one language and the preflight had nothing to compare.

    Proportional rather than equal, so the mix a pilot sees is the mix the full run would produce, with a floor
    of one row per language so nothing silently vanishes at a small limit. The result is re-sorted, so
    prefix-cache locality holds within the selection.
    """
    if limit <= 0 or len(rows) <= limit:
        return rows
    by_lang: dict[str, list[dict]] = {}
    for r in rows:
        by_lang.setdefault(r["lang"], []).append(r)
    total = len(rows)
    quota = {lg: max(1, round(limit * len(rs) / total)) for lg, rs in by_lang.items()}
    picked: list[dict] = []
    for lg, rs in by_lang.items():
        picked += rs[:quota[lg]]
    # Rounding and the per-language floor can overshoot or undershoot; settle it by taking from the largest
    # languages, which is where a row matters least.
    if len(picked) > limit:
        order = sorted(by_lang, key=lambda lg: -len(by_lang[lg]))
        keep = {lg: quota[lg] for lg in quota}
        i = 0
        while sum(keep.values()) > limit:
            lg = order[i % len(order)]
            if keep[lg] > 1:
                keep[lg] -= 1
            i += 1
        picked = [r for lg in by_lang for r in by_lang[lg][:keep[lg]]]
    elif len(picked) < limit:
        seen = {id(r) for r in picked}
        for r in rows:
            if len(picked) >= limit:
                break
            if id(r) not in seen:
                picked.append(r)
    picked.sort(key=lambda r: (r["lang"], r["task"], r["index"]))
    return picked


class EngineProvider:
    """A `Provider`-shaped wrapper around a vLLM engine.

    It exists so the document stages (longdocs.generate_documents) run on the rented GPU exactly as they run
    against an API. Without it, vllm_gen skipped stages 0 and 1 entirely and every long-document and RAG row
    silently fell back to asking one request for both the document and the conversation -- the behaviour that
    measured 635 words against a 4,000-word target.
    """

    def __init__(self, engine: Any, model: str, params_for: Any):
        self.engine = engine
        self.model = model
        self._params_for = params_for
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete_many(self, requests, *, on_result=None, progress: bool = True):
        reqs = list(requests)
        if not reqs:
            return
        for start in range(0, len(reqs), 256):
            batch = reqs[start:start + 256]
            outs = _chat_resilient(self.engine, [r.messages for r in batch],
                                   self._params_for(max(r.max_tokens for r in batch)))
            for req, out in zip(batch, outs):
                if out is None:
                    resp = Response(req.custom_id, "", False, "engine returned nothing",
                                    metadata=req.metadata)
                else:
                    comp = out.outputs[0]
                    self.prompt_tokens += len(out.prompt_token_ids)
                    self.completion_tokens += len(comp.token_ids)
                    resp = Response(req.custom_id, comp.text, True, None, len(out.prompt_token_ids),
                                    len(comp.token_ids), self.model, metadata=req.metadata)
                if on_result:
                    on_result(resp)
                yield resp
            if progress:
                print(f"    [stage] {min(start + len(batch), len(reqs)):,}/{len(reqs):,}", flush=True)

    def usage_line(self) -> str:
        return f"vllm/{self.model}: {self.prompt_tokens:,} in + {self.completion_tokens:,} out"


def stage_one(seed: Seed, todo: list[dict], provider: Optional[EngineProvider]) -> dict[str, dict]:
    """Documents for every long-document and RAG row, on this GPU. Mirrors generate.stage_one_documents."""
    from generate import stage_one_documents
    return stage_one_documents(seed, todo, provider, dry_run=provider is None)


# --------------------------------------------------------------------------- hardware


def gpu_report() -> list[dict[str, Any]]:
    """What is actually in this box. Empty list when torch/CUDA is absent, so --plan-only still works."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    cuda = getattr(torch.version, "cuda", None) or "0.0"
    try:
        cuda_ver = float(".".join(cuda.split(".")[:2]))
    except ValueError:
        cuda_ver = 0.0
    out = []
    for i in range(torch.cuda.device_count()):
        pr = torch.cuda.get_device_properties(i)
        out.append({"index": i, "name": pr.name, "total_gb": pr.total_memory / 2**30,
                    "capability": float(f"{pr.major}.{pr.minor}"), "cuda": cuda_ver,
                    "host_ram_gb": _host_ram_gb()})
    return out


def _host_ram_gb() -> float:
    """Total host RAM. On a unified-memory box (GB10 and the Grace-Blackwell family) this is the SAME pool the
    GPU allocates from, which is why gpu_memory_utilization there also decides how much page cache is left to
    stream the checkpoint through."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 0.0


# fp8 needs the hardware AND a toolkit that can build for it. Blackwell (SM 12.x) kernels require CUDA >= 12.9;
# with an older toolkit vLLM logs "Failed to get device capability: SM 12.x requires CUDA >= 12.9" and the fp8
# path is unreliable -- which is worse than slow, because it surfaces as wrong output or a crash mid-run rather
# than a refusal at startup.
_SM_MIN_CUDA = ((12.0, 12.9), (10.0, 12.8), (9.0, 12.0))


def cuda_supports_fp8(capability: float, cuda_ver: float) -> tuple[bool, str]:
    if capability < _FP8_MIN_CAPABILITY:
        return False, f"compute capability {capability} < {_FP8_MIN_CAPABILITY} (no fp8 hardware)"
    for sm, need in _SM_MIN_CUDA:
        if capability >= sm:
            if cuda_ver and cuda_ver < need:
                return False, (f"SM {capability} needs CUDA >= {need} for fp8 kernels, but torch was built "
                               f"against CUDA {cuda_ver}")
            break
    return True, ""


# On-the-fly quantization options, in descending quality. AWQ and GPTQ are deliberately NOT here: they need a
# pre-quantized CHECKPOINT, so they cannot be applied to an arbitrary --model on a rented box. Point --model at
# an already-quantized repo instead and vLLM detects it from the checkpoint's own config.
#   none          bf16 weights. Needs weights + KV to fit; for gemma-4-31b that is ~62 GB before any cache.
#   fp8           8-BIT: ~half the weight bytes, quantized at load, with fused W8A8 kernels. Needs compute
#                 capability >= 8.9, so Ada (L40S, RTX 4090/5090), Hopper (H100) and Blackwell (GB10) yes;
#                 Ampere (A100, A800, A40) NO -- there is no fp8 hardware path on Ampere.
#                 Per-tensor scales make its degradation small and well characterised, which is why `auto`
#                 will choose it over bf16 when it buys real concurrency. `int8` and `8bit` are accepted as
#                 aliases: vLLM's on-the-fly 8-bit path IS fp8, and W8A8-int8 requires a pre-quantized
#                 compressed-tensors checkpoint instead.
#   bitsandbytes  ~4-bit, quantized at load, works on ANY CUDA card including Ampere. It dequantizes inside
#                 the matmul with general-purpose kernels, so it buys memory rather than speed -- the fallback
#                 that makes a small card work at all. `auto` never selects it while a higher precision fits;
#                 4-bit fluency on Fon or Efik is unmeasured here and that is the operator's call.
# What --quantization accepts. Ours are the policies and aliases this script implements; the vLLM set is
# passed straight through for anyone who needs a backend we do not special-case. Kept so a typo is caught here
# in milliseconds instead of by vLLM after it has resolved the architecture and read the config.
_OUR_QUANTIZATIONS = {"fp8", "bitsandbytes", "int8", "8bit", "w8a8", "4bit", "nf4"}
_VLLM_QUANTIZATIONS = {
    "awq", "auto_awq", "awq_marlin", "gptq", "auto_gptq", "gptq_marlin", "fp8", "fbgemm_fp8", "fp_quant",
    "modelopt", "modelopt_fp4", "modelopt_mxfp8", "modelopt_mixed", "compressed-tensors", "experts_int8",
    "quark", "moe_wna16", "torchao", "inc", "mxfp4", "gpt_oss_mxfp4", "bitsandbytes", "online",
    "fp8_per_tensor", "fp8_per_block", "fp8_per_channel", "int8_per_channel_weight_only", "nvfp4_per_token",
    "mxfp8", "deepseek_v4_fp8", "humming",
}
_KNOWN_QUANTIZATIONS = _OUR_QUANTIZATIONS | _VLLM_QUANTIZATIONS

_FP8_MIN_CAPABILITY = 8.9
# A batch this small wastes the rental: vLLM's continuous batching is what amortises reading 62 GB of weights
# per decode step, so a box that fits the weights but leaves room for only 5 sequences is slower per dollar
# than the same box running a quantized copy with room for 40. `auto` therefore picks the highest-precision
# option that ALSO clears this floor, rather than the highest-precision option that merely loads.
MIN_CONCURRENCY = 12


def _fits(model: str, quant: Optional[str], vram_gb: float) -> bool:
    w = _weights_gb(model, quant)
    return not w or vram_gb >= w * 1.15


def _weights_gb(model: str, quant: Optional[str]) -> float:
    w = preset_for(model).get("weight_gb_bf16") or 0.0
    if quant == "fp8":
        return w / 2
    if quant == "bitsandbytes":
        return w / 3.6            # ~4-bit, plus norms and embeddings that stay in higher precision
    return w


def choose_quantization(model: str, requested: Optional[str], gpus: list[dict[str, Any]], tp: int,
                        context: int = 32_768, gpu_mem: float = 0.0) -> Optional[str]:
    """Pick a quantization that loads on THIS box. One command has to work on a 32 GB RTX 5090 and on an 80 GB
    A100 with nothing edited, which is what `auto` (the default) is for.

    `auto` is QUALITY-FIRST: the highest precision that fits. It does not quietly drop to 4-bit to win
    throughput, because the entire purpose of this corpus is fluency in languages where the generator is
    already weakest, and 4-bit quality on Fon or Efik is unmeasured here. Where a lower precision would buy a
    lot of concurrency, it says so and names the flag, and the decision stays with the operator.

    `--quantization throughput` opts into that trade: the highest precision that ALSO leaves room for
    MIN_CONCURRENCY sequences. On an 80 GB Ampere card at 32k that is the difference between ~5 concurrent
    sequences and ~40.
    """
    req = (requested or "auto").strip().lower()
    # vLLM's only on-the-fly 8-bit path is fp8; W8A8-int8 needs a pre-quantized checkpoint.
    req = {"int8": "fp8", "8bit": "fp8", "w8a8": "fp8", "4bit": "bitsandbytes",
           "nf4": "bitsandbytes"}.get(req, req)
    if req not in ("auto", "", "throughput"):
        if req in ("none", "bf16", "off"):
            return None
        # Validate HERE rather than letting vLLM reject it. It only finds out after resolving the
        # architecture and reading the config, which on a rented box is minutes of paid time, and its error is
        # a pydantic ValidationError wrapping a 30-item list. The value that provoked this was "aut": a pasted
        # command had wrapped mid-word, so the shell passed "aut" and then tried to run "o" as a command.
        if req not in _KNOWN_QUANTIZATIONS:
            near = [k for k in sorted(_OUR_QUANTIZATIONS) if k.startswith(req[:2]) or req in k]
            raise SystemExit(
                f"[vllm] --quantization {requested!r} is not a quantization this script or vLLM knows.\n"
                + (f"  did you mean: {', '.join(near)}?\n" if near else "")
                + f"  ours:  auto (default) | throughput | none | {' | '.join(sorted(_OUR_QUANTIZATIONS))}\n"
                f"  vLLM's own names are also accepted: {', '.join(sorted(_VLLM_QUANTIZATIONS)[:10])}, ...\n"
                f"  If the value looks truncated, check the command did not wrap mid-word -- a shell that "
                f"splits 'auto' passes 'aut' and then runs 'o' as a command.")
        return req
    low = model.lower()
    if any(k in low for k in ("awq", "gptq", "-fp8", "fp8-", "int4", "w4a16", "mxfp4", "bnb")):
        print(f"[vllm] {model} looks pre-quantized; letting vLLM read the format from its own config")
        return None
    if not gpus:
        return None                                   # --plan-only, or CPU box: nothing to decide against

    vram = sum(g["total_gb"] for g in gpus[:tp])
    cap = min(g["capability"] for g in gpus[:tp])
    cuda_ver = min((g.get("cuda") or 0.0) for g in gpus[:tp])
    candidates: list[Optional[str]] = [None]
    fp8_ok, fp8_why = cuda_supports_fp8(cap, cuda_ver)
    if fp8_ok:
        candidates.append("fp8")
    else:
        print(f"[vllm] fp8 unavailable: {fp8_why}.")
        if cap >= _FP8_MIN_CAPABILITY:
            print(f"[vllm]   the hardware has fp8 but this CUDA build cannot target it. Either use a newer "
                  f"image, or accept bf16/bitsandbytes. Forcing --quantization fp8 anyway risks failing "
                  f"mid-run rather than at startup.")
    candidates.append("bitsandbytes")

    fitting = [q for q in candidates if _fits(model, q, vram)]
    if not fitting:
        raise SystemExit(
            f"[vllm] {model} does not fit in {vram:.0f} GiB even 4-bit. Use --tp with more GPUs, pick a "
            f"smaller model (google/gemma-4-26b-a4b-it is an MoE with ~4B active), or rent a bigger box.")

    def seqs(q: Optional[str]) -> int:
        return max_sequences(model, gpus, tp, q, gpu_mem, context)

    if req == "throughput":
        for q in fitting:
            if seqs(q) >= MIN_CONCURRENCY:
                print(f"[vllm] throughput mode -> {q or 'bf16'}: ~{seqs(q)} concurrent {context:,}-token "
                      f"sequences on {vram:.0f} GiB")
                return q
        q = fitting[-1]
        print(f"[vllm] throughput mode: even {q or 'bf16'} leaves only ~{seqs(q)} sequences at {context:,}. "
              f"Consider --context 16384 or a bigger card.")
        return q

    best = fitting[0]
    # 8-bit is worth taking automatically; 4-bit is not. fp8's per-tensor W8A8 degradation is small and well
    # characterised, and on a bandwidth-poor card halving the bytes read per decode step is most of the
    # throughput. bitsandbytes 4-bit is a different proposition -- unmeasured on these languages, and its
    # kernels give back much of the saving -- so it stays an explicit choice.
    # Compare THROUGHPUT, not concurrency. A decode step reads every weight once and emits one token per
    # sequence, so tokens/s is proportional to sequences / weight_bytes -- fp8 wins on both terms at once, and
    # judging it on concurrency alone understated it by 2x. (This assumes the decode is bandwidth-bound, which
    # holds for a 31B dense model at these batch sizes; it would not at very large batch.)
    def tput(q: Optional[str]) -> float:
        w = _weights_gb(model, q)
        return seqs(q) / w if w else 0.0

    if best is None and "fp8" in fitting and tput("fp8") >= tput(None) * 1.5:
        print(f"[vllm] {vram:.0f} GiB, capability {cap}: bf16 fits, but fp8 gives ~{seqs('fp8')} concurrent "
              f"{context:,}-token sequences against ~{seqs(None)} AND halves the bytes read per decode step "
              f"-> ~{tput('fp8') / max(tput(None), 1e-9):.1f}x throughput -> fp8")
        print(f"[vllm]   fp8 is 8-bit with fused W8A8 kernels: a small, well-characterised quality cost for "
              f"most of the throughput. --quantization none forces bf16.")
        return "fp8"
    print(f"[vllm] {vram:.0f} GiB, capability {cap} -> {best or 'bf16'} (quality first): "
          f"~{_weights_gb(model, best):.0f} GiB weights, room for ~{seqs(best)} concurrent "
          f"{context:,}-token sequences")
    better = next((q for q in fitting[1:] if seqs(q) >= max(seqs(best) * 3, MIN_CONCURRENCY)), None)
    if better:
        print(f"[vllm]   NOTE: --quantization {better} would give ~{seqs(better)} concurrent sequences "
              f"({seqs(better) / max(seqs(best), 1):.0f}x). 4-bit fluency on the low-resource languages is "
              f"unmeasured here, so it is opt-in: --quantization throughput picks it automatically.")
    if seqs(best) < 4:
        print(f"[vllm]   WARNING: ~{seqs(best)} concurrent sequences is near-serial decoding. This rental "
              f"will be slow; --context 16384 or --quantization throughput is probably the better trade.")
    return best


def kv_free_gib(model: str, gpus: list[dict[str, Any]], tp: int, quant: Optional[str],
                gpu_mem: float) -> float:
    """VRAM left for the KV cache after weights, activations and fragmentation."""
    p = preset_for(model)
    if not gpus:
        return 0.0
    vram = sum(g["total_gb"] for g in gpus[:tp]) * (gpu_mem or p["gpu_memory_utilization"])
    return max(0.0, vram - _weights_gb(model, quant) - 4.0)   # 4 GiB: activations, CUDA graphs, fragmentation


def max_sequences(model: str, gpus: list[dict[str, Any]], tp: int, quant: Optional[str],
                  gpu_mem: float, context: int) -> int:
    """How many concurrent `context`-token sequences the KV cache has room for.

    This is what actually bounds throughput, and getting it wrong means an OOM twenty minutes into a rental.
    Both terms matter for gemma-4:
      per-token  10 full-attention layers x (4 kv heads x 512) x 2 B          = 40,960 B / token
      per-SEQUENCE 50 sliding layers x (16 x 256) x window 1024 x 2 B         = 419 MB, length-independent
    Ignoring the second term overstates concurrency badly: at 11 sequences it is another 4.3 GiB.
    """
    p = preset_for(model)
    per_token = int(p.get("kv_bytes_per_token") or 131_072)
    fixed = int(p.get("kv_bytes_fixed") or 0)
    per_seq = per_token * max(context, 1) + fixed
    if per_seq <= 0:
        return 0
    return int(kv_free_gib(model, gpus, tp, quant, gpu_mem) * 2**30 // per_seq)


def kv_tokens_available(model: str, gpus: list[dict[str, Any]], tp: int, quant: Optional[str],
                        gpu_mem: float) -> int:
    """Total KV tokens the box has room for, ignoring the per-sequence fixed cost. Reported for context only --
    use max_sequences() for anything that decides a batch size."""
    per_token = int(preset_for(model).get("kv_bytes_per_token") or 131_072)
    if not per_token:
        return 0
    return int(kv_free_gib(model, gpus, tp, quant, gpu_mem) * 2**30 / per_token)


# --------------------------------------------------------------------------- engine


def ensure_weights(model: str, *, expected_gb: float = 0.0) -> None:
    """Download the weights BEFORE starting the engine, with the disk checked first and Xet disabled on retry.

    vLLM downloads the weights inside engine startup, and when that fails the traceback is ~200 lines of
    engine-core plumbing with the real cause on the last line. The observed failure was

        RuntimeError: File reconstruction error: Internal Writer Error: Background writer channel closed

    from huggingface_hub's Xet backend -- which is what it reports when the writer cannot write, i.e. the disk
    filled. HF_HOME defaults to ~/.cache/huggingface, which on a rented box is usually the small root
    filesystem while the large volume is mounted elsewhere, so 62 GB of weights has nowhere to go.

    Checking here turns forty minutes of downloading into an instant, readable error, and a rental is billed by
    the second.
    """
    import shutil

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        # A public model still downloads, but unauthenticated requests are rate-limited and slower, and a gated
        # one fails with a 401 that looks nothing like a licence problem. Worth saying before 62 GB of transfer.
        print("[weights] WARNING: HF_TOKEN is not set in this process. Downloads will be rate-limited, and a "
              "gated model will fail with a 401.\n"
              "          export HF_TOKEN=... (a shell that ran `export` in a DIFFERENT window does not "
              "share it).", flush=True)
    expected_gb = expected_gb or (preset_for(model).get("weight_gb_bf16") or 0.0)
    cache = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    cache.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(cache).free / 2**30
    # The download needs room for the files plus Xet's staging copies; 1.4x is the margin that has held.
    need = expected_gb * 1.4
    print(f"[weights] cache {cache}  free {free_gb:,.0f} GiB  need ~{need:,.0f} GiB for {model}")
    if expected_gb and free_gb < need:
        biggest = _biggest_writable_mount()
        raise SystemExit(
            f"[weights] only {free_gb:,.0f} GiB free where the HF cache lives ({cache}), and {model} needs "
            f"~{need:,.0f} GiB.\n"
            f"This is what produces 'Internal Writer Error: Background writer channel closed' forty minutes "
            f"into a download.\n"
            f"Point the cache at the big volume before starting:\n"
            f"    export HF_HOME={biggest}/hf\n"
            f"and re-run. Nothing has been spent on GPU time yet.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return
    last = ""
    for attempt in range(3):
        # Xet is the default transfer backend and is where the observed failure came from. Fall back to the
        # plain HTTP path on the second attempt rather than retrying the same way three times.
        if attempt == 1:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            print("[weights] retrying with HF_HUB_DISABLE_XET=1 (classic HTTP transfer)", flush=True)
        try:
            snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
                              max_workers=4)
            print(f"[weights] {model} is present", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            print(f"[weights] attempt {attempt + 1}/3 failed: {last[:200]}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise SystemExit(
        f"[weights] could not download {model} after 3 attempts.\n  last error: {last[:400]}\n"
        f"Check: HF_TOKEN is set and has accepted the model's licence; there is disk space where HF_HOME "
        f"points; and the box has outbound network. Nothing has been spent on GPU time yet.")


def _biggest_writable_mount() -> str:
    """The mount with the most free space, for the HF_HOME suggestion above."""
    import shutil

    best, best_free = "/workspace", 0.0
    for cand in ("/workspace", "/data", "/mnt", "/scratch", "/root", "/tmp", "/"):
        try:
            free = shutil.disk_usage(cand).free
        except OSError:
            continue
        if free > best_free:
            best, best_free = cand, free
    return best


def warn_slow_load(model: str, gpus: list[dict[str, Any]], gpu_mem: float) -> None:
    """Say up front when the weights will take many minutes to load, and why.

    vLLM streams the checkpoint through the host page cache, and disables its own prefetch when the checkpoint
    does not fit:

        Filesystem type for checkpoints: OVERLAY. Checkpoint size: 58.25 GiB. Available RAM: 44.57 GiB.
        Auto-prefetch is disabled because ... the checkpoint size exceeds 90% of available RAM

    after which "Loading safetensors checkpoint shards 0/2" sits still for 10-25 minutes. That is normal and it
    IS progressing, but with no message saying so it looks like a hang, and on a box billed by the second the
    natural reaction is to kill it and start again -- paying the cost twice.

    On a UNIFIED-MEMORY box (GB10 and the Grace-Blackwell family) there is a second effect: the GPU allocates
    from the same LPDDR5X as the host, so gpu_memory_utilization also decides how much page cache is left to
    stream through. Reserving 0.90 there can leave less RAM than the checkpoint.
    """
    if not gpus:
        return
    weights_gb = preset_for(model).get("weight_gb_bf16") or 0.0
    ram = gpus[0].get("host_ram_gb") or 0.0
    vram = sum(g["total_gb"] for g in gpus)
    if not weights_gb or not ram:
        return
    util = gpu_mem or preset_for(model)["gpu_memory_utilization"]
    # Unified memory shows host RAM and VRAM as the same size; treat within 15% as the same pool.
    unified = abs(ram - vram) / max(ram, 1) < 0.15
    spare = ram * (1 - util) if unified else ram
    if spare < weights_gb:
        print(f"[vllm] the weights ({weights_gb:.0f} GiB) do not fit the {spare:.0f} GiB of RAM left for the "
              f"page cache, so vLLM will stream them from disk.")
        print(f"[vllm]   EXPECT 10-25 MINUTES at 'Loading safetensors checkpoint shards 0/N'. It is not hung; "
              f"nvidia-smi will show memory climbing. This happens once per process start.")
        if unified:
            lower = max(0.55, round(1 - (weights_gb * 1.25 / max(ram, 1)), 2))
            print(f"[vllm]   this box looks UNIFIED-MEMORY ({ram:.0f} GiB shared between host and GPU), so "
                  f"--gpu-mem {util} leaves only {spare:.0f} GiB for the page cache.")
            print(f"[vllm]   --gpu-mem {lower} would leave ~{ram * (1 - lower):.0f} GiB and load faster, at "
                  f"the cost of some KV cache. Worth trying if startup dominates a short run.", flush=True)


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


def _chat_resilient(engine: Any, convos: list[list[dict]], params: Any, *, depth: int = 0) -> list[Any]:
    """engine.chat, but a failure costs a few samples instead of the whole rental.

    A generation run is hours long on a paid box, and the failures that actually happen mid-run are (a) a CUDA
    OOM when several long sequences land in the same step, and (b) one malformed conversation upsetting the
    whole batch. Both used to kill the process and lose everything not yet flushed. Now an OOM halves the batch
    and retries, and a batch that fails at size 1 returns None for that one item and moves on.
    """
    if not convos:
        return []
    try:
        return list(engine.chat(convos, params, use_tqdm=(depth == 0)))
    except Exception as exc:  # noqa: BLE001 -- vLLM raises a wide zoo, and none of it should end the run
        msg = f"{type(exc).__name__}: {exc}"
        if len(convos) == 1:
            print(f"    [vllm] dropping 1 request: {msg[:160]}", flush=True)
            return [None]
        oom = any(k in msg.lower() for k in ("out of memory", "oom", "no available block", "kv cache"))
        half = len(convos) // 2
        print(f"    [vllm] {'OOM' if oom else 'error'} on a batch of {len(convos)}; splitting to "
              f"{half}+{len(convos) - half}: {msg[:120]}", flush=True)
        if oom:
            try:                       # free what the failed attempt left behind before trying again
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:          # noqa: BLE001
                pass
        return (_chat_resilient(engine, convos[:half], params, depth=depth + 1)
                + _chat_resilient(engine, convos[half:], params, depth=depth + 1))


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
        plan_only: bool, context: Optional[int] = None, run_tag: Optional[str] = None,
        preflight_n: int = 0, min_clean_rate: float = 0.35, push_every: int = 5) -> int:
    from assemble import set_response_budget
    from budgets import budget_for
    from generate import namespace
    import longdocs

    budget = budget_for(kind, context)
    # One place decides the phase's lengths, and both halves of the pipeline read it: the assembler's response
    # ceiling, the document size bands, the engine's context window and the completion cap.
    set_response_budget(budget.max_response_tokens)
    longdocs.set_context(kind, context)
    ctx = budget.context
    # The ENGINE window, not the model's context: vLLM reserves KV per sequence at max_model_len, so a phase
    # whose longest sample is 5,200 tokens must not ask for 32,768 (see Budget.engine_len).
    max_model_len = max_model_len or budget.engine_len or ctx
    max_tokens = max_tokens or budget.max_output_tokens
    ns = namespace(kind, run_tag)
    print(f"[{kind}] context {ctx:,} | engine window {max_model_len:,} | "
          f"response<={budget.max_response_tokens:,} | completion<={max_tokens:,} | {budget.notes}")

    rows, seed = build_work(kind, langs, limit)
    if kind == "judge":
        reqs, todo = rows, []
    else:
        todo = rows
        reqs = []
    print(f"[{ns}] {len(rows):,} {'requests' if kind == 'judge' else 'rows'} outstanding for this worker")
    if not rows:
        print("nothing to do")
        return 0

    gpus = gpu_report()
    for g in gpus:
        print(f"[gpu {g['index']}] {g['name']}  {g['total_gb']:.0f} GiB  capability {g['capability']}")
    if gpus and tp > len(gpus):
        raise SystemExit(f"--tp {tp} but only {len(gpus)} GPU(s) visible")
    quant = choose_quantization(model, quantization, gpus, tp, max_model_len, gpu_mem)
    from providers.base import warn_if_wrong_model_for_phase
    warn_if_wrong_model_for_phase(model, kind, langs)
    warn_slow_load(model, gpus, gpu_mem)

    if plan_only:
        in_tok_est = sum(len(m["content"]) for m in
                         (reqs[0].messages if reqs else build_request(seed, todo[0]).messages)) / 4
        _plan_report(kind, reqs or [build_request(seed, r) for r in todo[:1]] * len(todo),
                     in_tok_est, max_tokens, gpu_cost, model)
        if gpus:
            seqs = max_sequences(model, gpus, tp, quant, gpu_mem, max_model_len)
            print(f"  room for ~{seqs} concurrent {ctx:,}-token sequences with quantization={quant or 'bf16'}")
        return 0

    # Weights first, with the disk checked: a failure here is instant and readable, whereas the same failure
    # inside engine startup is 200 lines of plumbing with the cause on the last line.
    ensure_weights(model)
    engine = load_engine(model, tp=tp, max_model_len=max_model_len, gpu_mem=gpu_mem, quantization=quant)

    def params_for(mt: int):
        # Documents are free-form prose, so they are NOT schema-constrained; conversations are.
        return sampling_params("document", temperature=temperature, top_p=top_p,
                               max_tokens=min(mt, max_model_len), guided=False)

    # STAGE 0 + 1: the documents that long-document and RAG rows are built around. Runs on this same engine.
    docs: dict[str, dict] = {}
    if kind != "judge" and seed is not None:
        provider = EngineProvider(engine, model, params_for)
        docs = stage_one(seed, todo, provider)
        from longdocs import DOCUMENT_TASKS
        deferred = {r["custom_id"] for r in todo
                    if r["task"] in DOCUMENT_TASKS and r["custom_id"] not in docs}
        if deferred:
            todo = [r for r in todo if r["custom_id"] not in deferred]
            print(f"  deferring {len(deferred):,} document rows with no document yet")
        reqs = [build_request(seed, r, docs.get(r["custom_id"])) for r in todo]
        # Re-sort AFTER the documents are attached: a row carrying a 20k-token document has a completely
        # different prompt from its neighbours, so grouping it with them wins nothing. Rows without documents
        # keep their (lang, task) grouping, which is where the prefix cache pays.
        reqs.sort(key=lambda r: (bool(r.metadata.get("document")), r.metadata.get("lang", ""),
                                 r.metadata.get("task", ""), r.custom_id))

    params = sampling_params(kind, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                             guided=guided)

    from postprocess_gen import STATS, summary, to_record

    # PREFLIGHT: prove each language before spending the rest of the rental on it. Measured on the API path,
    # pretrain's clean rate ranged from 83% (ewe) to 8% (efi) -- at 8% a 30,000-document target needs ~375,000
    # requests, and discovering that forty hours in with nobody watching is what this prevents.
    if preflight_n and kind != "judge" and seed is not None and reqs:
        from preflight import run_preflight

        present = sorted({r.metadata.get("lang", "") for r in reqs} - {""})

        def _probe(probe_langs: list[str], n: int) -> dict[str, tuple[int, int]]:
            sample: list[Request] = []
            for lg in probe_langs:
                sample += [r for r in reqs if r.metadata.get("lang") == lg][:n]
            if not sample:
                return {}
            outs = _chat_resilient(engine, [r.messages for r in sample], params)
            tally: dict[str, list[int]] = {lg: [0, 0] for lg in probe_langs}
            for req, out in zip(sample, outs):
                lg = req.metadata.get("lang", "")
                tally.setdefault(lg, [0, 0])[1] += 1
                if out is None:
                    continue
                comp = out.outputs[0]
                rec = to_record(seed, Response(req.custom_id, comp.text, True, None,
                                              len(out.prompt_token_ids), len(comp.token_ids), model,
                                              metadata=req.metadata))
                if rec:
                    tally[lg][0] += 1
                    # The probe's output is real data: keep it rather than paying for it twice.
                    writer.write(rec["lang"], rec)
            return {k: (v[0], v[1]) for k, v in tally.items()}

        keep = run_preflight(kind, present, generate=_probe, per_lang=preflight_n,
                             out_dir=OUT_ROOT / kind, min_rate=min_clean_rate)
        STATS.clear()                      # the probe's drops are reported above; do not double-count them
        before = len(reqs)
        reqs = [r for r in reqs if r.metadata.get("lang") in set(keep)]
        if len(reqs) != before:
            print(f"  preflight removed {before - len(reqs):,} of {before:,} requests", flush=True)
        if not reqs:
            print("nothing left after preflight")
            writer.close()
            return 0
    if kind == "judge":
        from judge_gen import JUDGED_KIND, apply_verdict
        writer = ShardWriter(JUDGED_KIND, _shard_tag())
    else:
        writer = ShardWriter(ns, _shard_tag())

    kept = failed = 0
    out_tokens = 0
    pushed = 0
    t0 = time.time()
    # One random kept record per chunk, so a multi-hour run is watchable. Random rather than the first: work is
    # sorted by (lang, task) for prefix-cache locality, so the first record of every chunk has the same shape.
    from inspect_sample import print_one
    peek_rng = random.Random(0)

    for chunk_no, start in enumerate(range(0, len(reqs), chunk)):
        batch = reqs[start:start + chunk]
        chunk_kept: list[dict] = []
        # One engine call per chunk: vLLM does continuous batching internally, so a big chunk is what
        # actually saturates the GPU. Chunking exists only so shards flush and progress is visible.
        outs = _chat_resilient(engine, [r.messages for r in batch], params)
        for req, out in zip(batch, outs):
            if out is None:
                failed += 1
                continue
            comp = out.outputs[0]
            out_tokens += len(comp.token_ids)
            resp = Response(req.custom_id, comp.text, True, None,
                            len(out.prompt_token_ids), len(comp.token_ids), model,
                            metadata=req.metadata)
            try:
                rec = apply_verdict(resp) if kind == "judge" else to_record(seed, resp)
            except Exception as exc:  # noqa: BLE001 -- one malformed sample must not end the rental
                print(f"    [postprocess] {type(exc).__name__} on {req.custom_id}", flush=True)
                rec = None
            if rec is None:
                failed += 1
                continue
            writer.write(rec["lang"], rec)
            chunk_kept.append(rec)
            kept += 1
        dt = time.time() - t0
        tps = out_tokens / max(dt, 1e-9)
        done = start + len(batch)
        cost = gpu_cost * dt / 3600
        print(f"  [{ns}] {done:,}/{len(reqs):,}  kept {kept:,}  dropped {failed:,}  "
              f"{tps:,.0f} out-tok/s  {dt/60:.1f}m  ${cost:.2f} so far"
              + (f"  (${cost / (out_tokens / 1e6):.3f}/1M out tok, "
                 f"${cost / max(kept, 1):.5f}/kept sample)" if out_tokens > 1e5 else ""), flush=True)
        if STATS:
            print("   ", summary(), flush=True)
        print_one(chunk_kept, kind=kind, rng=peek_rng)

        # INCREMENTAL PUSH. The push used to happen only after the whole run, so a box that died -- or was
        # outbid, which is the normal way a spot rental ends -- lost everything generated. Pushing mid-run needs
        # the shard ROTATED rather than re-uploaded: hub.push_shards skips a path that already exists, on the
        # premise that shards are immutable, so re-pushing a growing file would silently upload nothing. Closing
        # the writer and opening a new one keeps every shard complete and immutable, and caps the loss at
        # push_every chunks.
        if push and push_every and (chunk_no + 1) % push_every == 0:
            done_paths = writer.close()
            if done_paths:
                from hub import push_shards
                try:
                    push_shards(JUDGED_KIND if kind == "judge" else kind, done_paths, repo_id=repo_id)
                    pushed += len(done_paths)
                except Exception as exc:  # noqa: BLE001 -- a Hub hiccup must not end a paid run
                    print(f"  [hub] push failed, keeping the shards locally and carrying on: "
                          f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
            writer = ShardWriter(JUDGED_KIND if kind == "judge" else ns, _shard_tag())

    paths = writer.close()
    dt = time.time() - t0
    total_cost = gpu_cost * dt / 3600
    print(f"\n[{ns}] kept {kept:,}  dropped {failed:,}  "
          f"(yield {kept / max(kept + failed, 1):.1%})  in {dt/60:.1f}m")
    print(f"  {out_tokens:,} output tokens at {out_tokens/max(dt,1e-9):,.0f} tok/s")
    if kept:
        print(f"  GPU cost ${total_cost:.2f} = ${total_cost / kept:.5f} per kept sample"
              + (f", ${total_cost / (out_tokens / 1e6):.3f} per 1M output tokens" if out_tokens else ""))
    print(" ", summary())
    for p in paths:
        print(f"  wrote {p}")
    if push and paths:
        from hub import push_shards
        try:
            push_shards(JUDGED_KIND if kind == "judge" else kind, paths, repo_id=repo_id)
            pushed += len(paths)
        except Exception as exc:  # noqa: BLE001
            print(f"  [hub] final push failed: {type(exc).__name__}: {str(exc)[:200]}\n"
                  f"  the shards are on disk; push them with hub.push_shards when the Hub is reachable.",
                  flush=True)
    if push:
        print(f"  pushed {pushed} shard file(s) to {repo_id} in total")
    elif paths:
        print(f"  NOT pushed (--push was not given). Shards are under {OUT_ROOT / kind}/<lang>/")
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


def resolve_langs(seed_kind: str, langs: Optional[str]) -> Optional[list[str]]:
    """Parse and VALIDATE --langs. An unknown code used to yield a silent "nothing to do", which on a rented
    GPU means paying for an hour of nothing because you typed `yoruba` instead of `yor`."""
    if not langs:
        return None
    from schemas.seed import Seed

    want = [s.strip() for s in langs.split(",") if s.strip()]
    known = {l.code: l.name for l in Seed.load(seed_kind).languages}
    bad = [c for c in want if c not in known]
    if bad:
        raise SystemExit(
            f"unknown language code(s) {bad} for kind={seed_kind}.\nAvailable:\n  "
            + "\n  ".join(f"{c:5} {n}" for c, n in known.items()))
    return want


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["pretrain", "sft", "rl", "judge"])
    ap.add_argument("--model", default=None,
                    help="default is the measured best for the phase: google/gemma-3-27b-it for pretrain "
                         "(89%% clean on the low-resource languages against gemma-4-31b's 6%%), "
                         "google/gemma-4-31b-it for sft and rl.")
    ap.add_argument("--langs", default=None,
                    help="comma list of language codes to generate for this run, e.g. yor,hau. "
                         "An unknown code is an error listing the valid ones.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size = number of GPUs")
    ap.add_argument("--max-model-len", type=int, default=0, help="0 = the model preset")
    ap.add_argument("--gpu-mem", type=float, default=0.0, help="0 = the model preset")
    ap.add_argument("--context", type=int, default=0,
                    help="target context in TOKENS -- not a sample count (use --limit for that). Max 32768, the model's own block_size. For PRETRAIN this is the size of one document, since a pretraining sample IS one document (default 1024); for SFT and RL it is the training window a conversation must fit in (default 32768, and the training side must match). 0 = the phase default.")
    ap.add_argument("--quantization", default="auto",
                    help="auto (default, highest precision that fits) | throughput (highest precision that "
                         "also leaves room to batch) | none | fp8 | bitsandbytes | any vLLM name. AWQ and "
                         "GPTQ need a pre-quantized --model; point at that repo and auto detects it.")
    ap.add_argument("--chunk", type=int, default=2048,
                    help="requests per engine call; only affects flush cadence, not batching")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = the phase budget (budgets.py)")
    ap.add_argument("--no-guided", dest="guided", action="store_false",
                    help="disable grammar-constrained JSON (expect more json_invalid drops)")
    ap.add_argument("--push", action="store_true",
                    help="push shards to the Hub. Without this NOTHING is uploaded and the shards stay on the "
                         "box -- which is lost when the rental ends.")
    ap.add_argument("--push-every", type=int, default=5, metavar="CHUNKS",
                    help="with --push, also push every CHUNKS chunks rather than only at the end, so a box "
                         "that dies or is outbid loses at most that much. 0 = only at the end.")
    ap.add_argument("--repo-id", default="BeardedMonster/data-gen")
    ap.add_argument("--gpu-cost", type=float, default=2.0, help="USD per hour for the whole box")
    ap.add_argument("--plan-only", action="store_true", help="cost model only; no GPU, no generation")
    ap.add_argument("--preflight", type=int, default=8, metavar="N",
                    help="probe N samples per language first and DROP languages whose clean rate is below "
                         "--min-clean-rate. 0 disables. The probe's output is kept, not thrown away. Measured: "
                         "pretrain clean rates ran from 83%% (ewe) to 8%% (efi), and at 8%% a 30,000-document "
                         "target needs ~375,000 requests.")
    ap.add_argument("--min-clean-rate", type=float, default=0.35,
                    help="preflight floor. 0 keeps every language regardless.")
    ap.add_argument("--run-tag", default=None,
                    help="namespace local shards so several models can generate the same rows independently")
    a = ap.parse_args()
    if a.kind == "judge":
        a.temperature = 0.0  # a ranking should be reproducible
        a.max_tokens = 700
    if not a.model:
        from providers.base import RECOMMENDED_MODEL
        a.model = RECOMMENDED_MODEL.get(a.kind, "google/gemma-4-31b-it")
        print(f"[model] no --model given; using the measured best for {a.kind}: {a.model}")
    langs = resolve_langs("rl" if a.kind == "judge" else a.kind, a.langs)
    return run(a.kind, a.model, langs=langs, limit=a.limit, tp=a.tp, max_model_len=a.max_model_len,
               gpu_mem=a.gpu_mem, quantization=a.quantization, chunk=a.chunk,
               temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, guided=a.guided,
               push=a.push, repo_id=a.repo_id, gpu_cost=a.gpu_cost, plan_only=a.plan_only,
               context=a.context or None, run_tag=a.run_tag, preflight_n=a.preflight,
               min_clean_rate=a.min_clean_rate, push_every=a.push_every)


if __name__ == "__main__":
    raise SystemExit(main())
