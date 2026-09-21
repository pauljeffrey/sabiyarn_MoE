"""Reward functions for rl/policy_gradient.py.

A reward is a callable `(items, completions) -> list[float]` where `items[i]` is the data record the prompt came
from (`reference`, `source`, ... as the dataset provides them) and `completions[i]` the sampled text.

  chrf       sentence chrF++ against `reference` (sacrebleu, already a dependency). Cheap, needs no extra install:
             use it to smoke-test a pipeline, or as the reward when you do not want a learned metric.
  africomet  AfriCOMET (masakhane/africomet-stl) on (source, completion, reference). It is a *learned* metric:
             optimising it hard finds its blind spots (reward hacking), so keep the KL penalty on, hold out
             chrF++/BLEU (eval_suite) as the honest measure, and look at samples. See rl/comet_worker.py for why it
             runs in its own process by default.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Protocol

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


def build_reward(cfg: RLConfig, device: str = "cpu") -> Callable[[list[dict], list[str]], list[float]]:
    if cfg.reward == "chrf":
        return ChrfReward()
    if cfg.reward == "africomet":
        if cfg.reward_python:
            return CometWorkerReward(cfg.reward_python, cfg.reward_id, cfg.reward_batch_size, cfg.reward_gpu)
        try:
            return CometInProcessReward(cfg.reward_id, cfg.reward_batch_size, device, cfg.reward_gpu)
        except ImportError as exc:
            raise SystemExit(
                "reward: africomet needs `unbabel-comet`, which requires transformers<5 and so cannot share this "
                "environment. Create a second one and point `reward_python` at it:\n"
                "    python -m venv .venv-comet && .venv-comet/bin/pip install unbabel-comet\n"
                "    (config: reward_python: .venv-comet/bin/python   or   RL_REWARD_PYTHON=...)\n"
                f"original error: {exc}")
    raise ValueError(f"unknown reward {cfg.reward!r}")
