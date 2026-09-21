#!/usr/bin/env python3
"""AfriCOMET scoring worker: `<python-with-unbabel-comet> -m rl.comet_worker --model masakhane/africomet-stl`.

Why a separate process: `unbabel-comet` pins transformers<5, huggingface_hub<1 and numpy<2, which conflicts with the
transformers 5.x this repo's model code is written against (requirements.txt). Installing it into its own
virtualenv and talking to it over stdin/stdout keeps both stacks intact. Protocol: one JSON object per line in,
one per line out --
    in : {"data": [{"src": ..., "mt": ..., "ref": ...}, ...], "batch_size": 16}
    out: {"scores": [0.71, ...]}   or   {"error": "..."}
Library chatter is sent to stderr; only protocol lines reach the real stdout. `--fake` scores by character-level
overlap of mt and ref (no model, no comet install) so the wiring can be tested anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _fake_scores(data: list[dict]) -> list[float]:
    out = []
    for d in data:
        a, b = set(d["mt"].lower()), set(d["ref"].lower())
        out.append(len(a & b) / max(len(a | b), 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="masakhane/africomet-stl")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    proto = os.fdopen(os.dup(1), "w", buffering=1)  # keep the real stdout for the protocol ...
    os.dup2(2, 1)  # ... and send anything a library prints to stderr

    model = None
    if not args.fake:
        from comet import download_model, load_from_checkpoint

        model = load_from_checkpoint(download_model(args.model))
        model.eval()
    proto.write(json.dumps({"ready": True}) + "\n")

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if args.fake:
                scores = _fake_scores(req["data"])
            else:
                import torch

                out = model.predict(req["data"], batch_size=int(req.get("batch_size", 16)),
                                    gpus=1 if torch.cuda.is_available() else 0, progress_bar=False)
                scores = [float(s) for s in (out.scores if hasattr(out, "scores") else out)]
            proto.write(json.dumps({"scores": scores}) + "\n")
        except Exception as exc:  # keep serving; the client decides what to do
            proto.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
