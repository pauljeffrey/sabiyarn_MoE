"""Reward functions for rl/policy_gradient.py.

A reward is a callable `(items, completions) -> list[float]` where `items[i]` is the data record the prompt came
from (`reference`, `source`, ... as the dataset provides them) and `completions[i]` the sampled text.

  chrf       sentence chrF++ against `reference` (sacrebleu, already a dependency). Cheap, needs no extra install:
             use it to smoke-test a pipeline, or as the reward when you do not want a learned metric.
  africomet  AfriCOMET (masakhane/africomet-stl) on (source, completion, reference). It is a *learned* metric:
             optimising it hard finds its blind spots (reward hacking), so keep the KL penalty on, hold out
             chrF++/BLEU (eval_suite) as the honest measure, and look at samples. See rl/comet_worker.py for why it
             runs in its own process by default.
  verifier   The programmatic scorers in rl/verifiers.py, driven by the `checks` each RL sample carries from
             the seed. No model in the loop: it scores procedure and form -- did it call a tool before
             claiming to know, is that figure actually in the document, was the stated constraint obeyed --
             which is most of what this phase teaches. Cheap, reproducible, and the only reward here that is
             not itself a learned metric with blind spots to exploit.
  verifier+africomet
             Both, blended by `reward_blend` (default 0.5). The sensible default for a mixed corpus: the
             verifier carries the behaviour and reasoning tasks, AfriCOMET the translation ones.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import structlog

from rl.config import RLConfig

LOG = structlog.get_logger()
ROOT = Path(__file__).resolve().parents[1]


class RewardFn(Protocol):
    def __call__(self, items: list[dict], completions: list[str]) -> list[float]: ...


class ChrfReward:
    def __init__(self, word_order: int = 2):
        from sacrebleu.metrics import CHRF

        self.metric = CHRF(word_order=word_order)  # word_order=2 -> chrF++

    def __call__(self, items, completions):
        return [self.metric.sentence_score(c.strip(), [i["reference"]]).score / 100.0 for i, c in zip(items, completions)]


def _comet_rows(items: list[dict], completions: list[str]) -> list[dict]:
    return [{"src": i.get("source", ""), "mt": c.strip(), "ref": i["reference"]} for i, c in zip(items, completions)]


class CometWorkerReward:
    """AfriCOMET behind a subprocess (see rl/comet_worker.py). One worker per rank, pinned to that rank's GPU."""

    def __init__(self, python: str, model_id: str, batch_size: int, use_gpu: bool, fake: bool = False):
        env = {k: v for k, v in os.environ.items()
               if k not in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "LOCAL_WORLD_SIZE")}
        if use_gpu and "LOCAL_RANK" in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
        elif not use_gpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [python, "-m", "rl.comet_worker", "--model", model_id] + (["--fake"] if fake else [])
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        self.batch_size = batch_size
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, cwd=ROOT)
        atexit.register(self.close)
        ready = self._readline()
        if not ready.get("ready"):
            raise RuntimeError(f"COMET worker failed to start: {ready}")

    def _readline(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(
                "COMET worker exited (see its stderr above). `reward_python` must be a Python with `unbabel-comet` "
                "installed, e.g.  python -m venv .venv-comet && .venv-comet/bin/pip install unbabel-comet")
        return json.loads(line)

    def __call__(self, items, completions):
        self.proc.stdin.write(json.dumps({"data": _comet_rows(items, completions), "batch_size": self.batch_size}) + "\n")
        self.proc.stdin.flush()
        reply = self._readline()
        if "error" in reply:
            raise RuntimeError(f"COMET worker error: {reply['error']}")
        return [float(s) for s in reply["scores"]]

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


class CometInProcessReward:
    """AfriCOMET loaded into this process. Works only where `import comet` works next to the installed
    transformers; otherwise use `reward_python`."""

    def __init__(self, model_id: str, batch_size: int, device: str, use_gpu: bool):
        from comet import download_model, load_from_checkpoint

        self.model = load_from_checkpoint(download_model(model_id))
        self.model.eval()
        self.batch_size, self.device = batch_size, device if use_gpu else "cpu"
        self.model.to(self.device)

    def __call__(self, items, completions):
        import torch

        rows = _comet_rows(items, completions)
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(rows), self.batch_size):
                inputs = self.model.prepare_sample(rows[i : i + self.batch_size], stage="predict")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores += [float(s) for s in self.model(**inputs).score.flatten().tolist()]
        return scores


class VerifierReward:
    """The programmatic checks from rl/verifiers.py, one reward per (item, completion).

    Each item carries its own `checks` (written into the seed by data-gen/rewards_spec.py), so a reasoning
    sample and a summarisation sample are scored by different predicates without this class knowing which is
    which. An item with no checks scores `neutral` rather than 0: a missing-checks row is a gap in the data,
    and scoring it zero would train the policy to avoid whatever that row happened to contain.
    """

    def __init__(self, neutral: float = 0.5, log_every: int = 0):
        # `neutral` is returned for a sample this reward must not score -- a judge-only task, or a row with
        # no checks. Not 0: a zero would teach the policy to avoid whatever those rows contain.
        from rl.verifiers import score, score_group  # local import: keeps torch-free callers torch-free

        self._score = score
        self._score_group = score_group
        self.neutral = neutral
        self.log_every = log_every
        self._n = 0

    def __call__(self, items, completions):
        # Score each PROMPT'S GROUP together: consecutive entries sharing an id are the k completions
        # sampled for one prompt. Scoring them jointly is what stops an unscoreable candidate from winning
        # (see rl.verifiers.score_group), and the advantage is a within-group difference anyway.
        out: list[Optional[float]] = [None] * len(items)
        groups: dict[Any, list[int]] = {}
        for i, item in enumerate(items):
            key = item.get("id") or id(item)
            groups.setdefault(key, []).append(i)

        for idxs in groups.values():
            item = items[idxs[0]]
            checks = item.get("checks") or []
            # Some tasks the programmatic checks measurably CANNOT rank: where the two candidates differ in
            # factual truth rather than in procedure, this reward scored the worse one higher 60% of the
            # time. Returning `neutral` defers to the judge half of a blended reward instead of contributing
            # a confidently wrong signal. See data-gen/rewards_spec.JUDGE_ONLY_TASKS.
            if item.get("judge_required") or not checks:
                for i in idxs:
                    out[i] = self.neutral
                continue
            rs, details = self._score_group([completions[i] for i in idxs], item, checks)
            for i, r, detail in zip(idxs, rs, details):
                out[i] = r
                self._n += 1
                if self.log_every and self._n % self.log_every == 0:
                    LOG.info("verifier", reward=round(r, 3), task=(item.get("tasks") or [None])[0],
                             lang=item.get("lang"), detail=detail)
        return [self.neutral if r is None else r for r in out]


class BlendedReward:
    """Weighted sum of two rewards on the same batch. `w` is the weight on the FIRST."""

    def __init__(self, first: RewardFn, second: RewardFn, w: float = 0.5):
        self.first, self.second, self.w = first, second, max(0.0, min(1.0, w))

    def __call__(self, items, completions):
        a = self.first(items, completions)
        b = self.second(items, completions)
        return [self.w * x + (1.0 - self.w) * y for x, y in zip(a, b)]


class StructuredTranslationReward:
    """Reward for a turn that is  SCAFFOLD + CONTENT  (see rl/structure.py).

        R = w_f * R_format + (1 - w_f) * gate * R_content        (then, optionally, shaped: see below)

    R_format  field-wise match of the scaffold against the sample's expected one (order-agnostic plan F1).
    R_content the translation after <response>: reference-free AfriCOMET-QE on (source, translation), clipped to [0, 1]. No
              reference is needed. (`use_reference=True` switches rows that carry one to reference-based AfriCOMET blended with
              sentence chrF++, `chrf_anchor`, because a learned metric alone is what gets exploited -- Amrhein & Sennrich 2022; Gao
              et al. 2023 on over-optimisation.) A length-ratio guard halves it for implausibly short/long output and a copy of
              the source scores 0, which is the classic way to fool a QE metric.
    shaping   shaping="threshold": R < threshold becomes a large negative number, R >= threshold a positive one (rl.structure.
              shape_reward); order within a prompt's group is unchanged.
    gate      content counts only when the target-language tag is right and the response is non-empty and clean. Without the
              gate the policy could earn content reward in the wrong language; without the additive format term it would get no
              gradient on the scaffold while the gate is closed.

    `wants_raw` makes the trainer pass completions decoded WITH special tokens (the scaffold is made of them).
    The group-mean baseline of rl/policy_gradient.py then turns R into advantages exactly as before (GRPO-style, Shao et al. 2024).
    """

    wants_raw = True

    def __init__(self, reference_reward: Optional[RewardFn], qe_reward: Optional[RewardFn] = None, format_weight: float = 0.2,
                 chrf_anchor: float = 0.2, length_guard: bool = True, weights: Optional[dict] = None, log_every: int = 0,
                 use_reference: bool = False, shaping: str = "none", threshold: float = 0.46, negative_scale: float = 2.0,
                 shaper=None, train_unusable: bool = False):
        self.ref_reward, self.qe_reward = reference_reward, qe_reward
        self.wf, self.alpha, self.length_guard, self.weights, self.log_every = format_weight, chrf_anchor, length_guard, weights, log_every
        self.use_reference, self.shaping, self.threshold, self.negative_scale = use_reference, shaping, threshold, negative_scale
        self.shaper = shaper                     # rl.shaping.ContentShaper for shaping in ("calibrated", "adaptive")
        self.train_unusable = train_unusable
        self._chrf = None
        self._n = 0
        self.last_stats: dict[str, float] = {}      # read by the trainer's logger: where the reward comes from
        self.last_weights: Optional[list[float]] = None   # per-completion advantage weights (QE reliability of the pair), read by the trainer

    def _sentence_chrf(self, hyp: str, ref: str) -> float:
        if self._chrf is None:
            from sacrebleu.metrics import CHRF
            self._chrf = CHRF(word_order=2)
        return self._chrf.sentence_score(hyp, [ref]).score / 100.0

    @staticmethod
    def _norm(t: str) -> str:
        return " ".join(t.lower().split())

    def __call__(self, items, completions):
        from rl.structure import expected_translation_structure, format_score

        fmt = []
        for it, c in zip(items, completions):
            exp = it.get("expected_structure") or expected_translation_structure(it.get("source_lang", ""), it.get("targ_lang", ""))
            fmt.append(format_score(c, exp, self.weights))
        # content, scored only for gated rows, in one batch per scorer
        has_ref = [bool(self.use_reference and it.get("reference")) for it in items]
        idx_ref = [i for i, f in enumerate(fmt) if f["gate"] and has_ref[i]]
        idx_qe = [i for i, f in enumerate(fmt) if f["gate"] and not has_ref[i]]
        content = [0.0] * len(items)
        if idx_ref and self.ref_reward is not None:
            sc = self.ref_reward([items[i] for i in idx_ref], [fmt[i]["response"] for i in idx_ref])
            for i, s in zip(idx_ref, sc):
                base = max(0.0, min(1.0, float(s)))
                if self.alpha > 0:
                    base = (1 - self.alpha) * base + self.alpha * self._sentence_chrf(fmt[i]["response"], items[i]["reference"])
                content[i] = base
        if idx_qe and self.qe_reward is not None:
            qitems = [{**items[i], "reference": ""} for i in idx_qe]
            for i, s in zip(idx_qe, self.qe_reward(qitems, [fmt[i]["response"] for i in idx_qe])):
                content[i] = max(0.0, min(1.0, float(s)))
        out = []
        idx_gate_scores = {i: content[i] for i, f in enumerate(fmt) if f["gate"]}
        n = max(1, len(items))
        self.last_stats = {"fmt_score": sum(f["score"] for f in fmt) / n, "fmt_target_lang": sum(f["target_lang"] for f in fmt) / n,
                           "fmt_input_lang": sum(f["input_lang"] for f in fmt) / n, "fmt_plan_f1": sum(f["plan"] for f in fmt) / n,
                           "fmt_structure": sum(f["structure"] for f in fmt) / n, "gate_rate": sum(f["gate"] for f in fmt) / n,
                           "content_mean_gated": (sum(content) / max(1, sum(f["gate"] for f in fmt)))}
        self.last_weights = [1.0] * len(items)
        for i, (it, f) in enumerate(zip(items, fmt)):
            c = content[i]
            if f["gate"]:
                resp = f["response"]
                if self._norm(resp) == self._norm(it.get("source", "")) and it.get("source"):
                    c = 0.0                                   # copying the input is not translating it
                elif self.length_guard:
                    ref = (it.get("reference") if self.use_reference else "") or it.get("source") or ""
                    ratio = len(resp) / max(1, len(ref))
                    if ratio < 0.3 or ratio > 3.0:
                        c *= 0.5
            if self.shaper is not None and self.shaping in ("calibrated", "adaptive"):
                # content curve (failures steeply negative, good -> great resolved) + scaffold on a -1..1 scale
                key = f"{it.get('source_lang') or ''}>{it.get('targ_lang') or ''}"
                scaffold = 4.0 * f["score"] - 2.0                                    # scaffold on the same -2 .. +2 scale
                if not self.shaper.is_usable(key):
                    # QE cannot tell real from mismatched translations for this pair (measured): train the scaffold only
                    r, self.last_weights[i] = scaffold, 0.0 if not self.train_unusable else 1.0
                    out.append(r)
                    self._n += 1
                    continue
                if f["gate"] and idx_gate_scores.get(i) is not None:
                    self.shaper.observe(key, [idx_gate_scores[i]])
                cs = self.shaper(c, key) if f["gate"] else self.shaper(0.0, key)
                # the content curve rides on the pair's measured reliability; the advantage is also down-weighted by it (trainer),
                # because per-group standardisation would otherwise cancel any per-pair scale
                self.last_weights[i] = self.shaper.reliability(key)
                r = self.wf * scaffold + (1 - self.wf) * cs
                out.append(r)
                self._n += 1
                continue
            r = self.wf * f["score"] + (1 - self.wf) * (c if f["gate"] else 0.0)
            if self.shaping == "threshold":
                from rl.structure import shape_reward
                r = shape_reward(r, self.threshold, self.negative_scale)
            out.append(r)
            self._n += 1
            if self.log_every and self._n % self.log_every == 0:
                LOG.info("structured_reward", reward=round(r, 3), fmt=round(f["score"], 3), content=round(c, 3), gate=f["gate"],
                         plan=round(f["plan"], 2), target=f["target_lang"])
        return out


def _africomet(cfg: RLConfig, device: str, model_id: Optional[str] = None) -> RewardFn:
    model_id = model_id or cfg.reward_id
    if cfg.reward_python:
        return CometWorkerReward(cfg.reward_python, model_id, cfg.reward_batch_size, cfg.reward_gpu)
    try:
        return CometInProcessReward(model_id, cfg.reward_batch_size, device, cfg.reward_gpu)
    except ImportError as exc:
        raise SystemExit(
            "reward: africomet needs `unbabel-comet`, which requires transformers<5 and so cannot share this "
            "environment. Create a second one and point `reward_python` at it:\n"
            "    python -m venv .venv-comet && .venv-comet/bin/pip install unbabel-comet\n"
            "    (config: reward_python: .venv-comet/bin/python   or   RL_REWARD_PYTHON=...)\n"
            f"original error: {exc}")


def build_reward(cfg: RLConfig, device: str = "cpu") -> Callable[[list[dict], list[str]], list[float]]:
    if cfg.reward in ("structured_africomet", "structured_chrf"):
        if cfg.reward == "structured_chrf":                        # no COMET install needed: wiring / smoke tests
            ref, qe = ChrfReward(), None
        else:
            ref = _africomet(cfg, device) if cfg.reward_use_reference else None
            qe = _africomet(cfg, device, cfg.reward_qe_id) if cfg.reward_qe_id else None
            if not cfg.reward_use_reference and qe is None:
                raise SystemExit("reward: structured_africomet without references needs reward_qe_id (a reference-free AfriCOMET)")
        shaper = None
        if cfg.reward_shaping in ("calibrated", "adaptive"):
            from rl.shaping import ContentShaper
            shaper = ContentShaper.from_file(cfg.reward_calibration_path, mode=cfg.reward_shaping, neg_scale=cfg.reward_negative_scale,
                                             window=cfg.reward_adaptive_window, warmup=cfg.reward_adaptive_warmup,
                                             default_anchors=tuple(float(x) for x in str(cfg.reward_default_anchors).split(",")))
        return StructuredTranslationReward(ref, qe, cfg.reward_format_weight, cfg.reward_chrf_anchor, cfg.reward_length_guard,
                                           shaper=shaper,
                                           log_every=getattr(cfg, "reward_log_every", 0), use_reference=cfg.reward_use_reference,
                                           shaping=cfg.reward_shaping, threshold=cfg.reward_threshold,
                                           negative_scale=cfg.reward_negative_scale)
    if cfg.reward == "chrf":
        return ChrfReward()
    if cfg.reward == "verifier":
        return VerifierReward(log_every=getattr(cfg, "reward_log_every", 0))
    if cfg.reward in ("verifier+africomet", "africomet+verifier"):
        blend = float(getattr(cfg, "reward_blend", 0.5))
        return BlendedReward(VerifierReward(log_every=getattr(cfg, "reward_log_every", 0)),
                             _africomet(cfg, device), w=blend)
    if cfg.reward == "africomet":
        return _africomet(cfg, device)
    raise ValueError(
        f"unknown reward {cfg.reward!r} -- expected one of: chrf, africomet, verifier, verifier+africomet, structured_africomet, structured_chrf")
