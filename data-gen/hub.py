"""Push generated data to the Hugging Face dataset repo, incrementally.

    python hub.py --push-seeds                      # upload seeds/*.json + the dataset card
    python hub.py --push sft                        # upload any local shards the repo does not have
    python hub.py --status                          # what is on the Hub right now

Layout on the Hub (BeardedMonster/data-gen):

    seeds/{pretrain,sft,rl}.json          the briefs that produced the data
    pretrain/<lang>/shard-<id>.jsonl
    sft/<lang>/shard-<id>.jsonl
    rl/<lang>/shard-<id>.jsonl
    README.md                             dataset card, regenerated with live counts

Shards are immutable and content-addressed by the run that wrote them, so "incremental" is simply: list what
is already there, upload what is not. Nothing is ever re-uploaded or rewritten, which is what keeps this
cheap to run after every generation batch. All shards of one push go up in a SINGLE commit -- pushing 300
files as 300 commits makes the repo history unusable and is much slower.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from providers.base import load_env  # noqa: E402

REPO_ID = os.environ.get("DATA_GEN_HF_REPO", "BeardedMonster/data-gen")
KINDS = ("pretrain", "sft", "rl")
OUT_ROOT = Path(os.environ.get("DATA_GEN_OUTPUT_DIR", HERE / "data")) / "out"


def _api():
    load_env()
    from huggingface_hub import HfApi
    token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if not token:
        raise SystemExit("no HF token: set HF_WRITE_TOKEN (or HF_TOKEN) in the repo-root .env")
    return HfApi(token=token)


def ensure_repo(repo_id: str = REPO_ID, private: bool = True) -> None:
    api = _api()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)


def remote_files(repo_id: str = REPO_ID) -> set[str]:
    api = _api()
    try:
        return set(api.list_repo_files(repo_id, repo_type="dataset"))
    except Exception:
        ensure_repo(repo_id)
        return set()


def _repo_path(kind: str, path: Path) -> str:
    """local .../out/<kind>/<lang>/shard-x.jsonl -> '<kind>/<lang>/shard-x.jsonl'"""
    return f"{kind}/{path.parent.name}/{path.name}"


def push_shards(kind: str, paths: Iterable[Path], repo_id: str = REPO_ID) -> int:
    """Upload shards the repo does not already have, in one commit. Returns how many were uploaded."""
    from huggingface_hub import CommitOperationAdd

    api = _api()
    ensure_repo(repo_id)
    have = remote_files(repo_id)
    ops = []
    for p in paths:
        if not p.exists() or p.stat().st_size == 0:
            continue
        dest = _repo_path(kind, p)
        if dest in have:
            continue  # immutable shards: already there means identical
        ops.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(p)))
    if not ops:
        print(f"[hub] {kind}: nothing new to push")
        return 0
    mb = sum(Path(o.path_or_fileobj).stat().st_size for o in ops) / 2**20
    api.create_commit(repo_id, repo_type="dataset", operations=ops,
                      commit_message=f"Add {len(ops)} {kind} shard(s) ({mb:.1f} MB)")
    print(f"[hub] {kind}: pushed {len(ops)} shard(s), {mb:.1f} MB -> {repo_id}")
    return len(ops)


def push_all_local(kind: str, repo_id: str = REPO_ID) -> int:
    root = OUT_ROOT / kind
    if not root.exists():
        print(f"[hub] no local data at {root}")
        return 0
    return push_shards(kind, sorted(root.glob("*/shard-*.jsonl")), repo_id)


def push_seeds(repo_id: str = REPO_ID) -> None:
    from huggingface_hub import CommitOperationAdd

    api = _api()
    ensure_repo(repo_id)
    ops = [CommitOperationAdd(path_in_repo=f"seeds/{p.name}", path_or_fileobj=str(p))
           for p in sorted((HERE / "seeds").glob("*.json"))]
    card = build_card(repo_id)
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode("utf-8")))
    api.create_commit(repo_id, repo_type="dataset", operations=ops,
                      commit_message="Update seeds and dataset card")
    print(f"[hub] pushed {len(ops) - 1} seed file(s) + dataset card -> {repo_id}")


def status(repo_id: str = REPO_ID) -> dict[str, dict[str, int]]:
    files = remote_files(repo_id)
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for f in files:
        parts = f.split("/")
        if len(parts) == 3 and parts[0] in KINDS:
            out[parts[0]][parts[1]] = out[parts[0]].get(parts[1], 0) + 1
    return dict(out)


def build_card(repo_id: str = REPO_ID) -> str:
    """Dataset card. Reads the seeds for intent and the Hub for what actually landed."""
    from schemas.seed import Seed

    lines = [
        "---", "license: cc-by-4.0", "language:",
        *[f"- {c}" for c in ("en", "yo", "ha", "ig", "pcm", "tw", "ee", "fon", "ak", "ff")],
        "task_categories:", "- text-generation", "tags:", "- west-african-languages", "- tool-calling",
        "- synthetic", "---", "",
        "# SabiYarn data-gen", "",
        "Synthetic training data for [SabiYarn](https://huggingface.co/Aletheia-ng/SabiYarn_MoE-280M), a ~306M "
        "parameter MoE model for 12 West African languages plus English.", "",
        "## Design", "",
        "The model is small, so this corpus does **not** try to pack world facts into its weights. It teaches "
        "general understanding of how things work, plus two reflexes: reason inside `<think>...</think>`, then "
        "either use a tool or say plainly that it does not know. A confident invention is the worst outcome; "
        "an honest \"I don't know\" is always preferred.", "",
        "## Splits", "",
        "| folder | contents |", "|---|---|",
        "| `pretrain/` | plain prose documents, no markup, per language |",
        "| `sft/` | 6-10 message conversations with tool calls, `messages` + rendered `text` |",
        "| `rl/` | conversation prefix + 2-3 ranked candidate final replies |",
        "| `seeds/` | the briefs that generated all of it -- read these first |", "",
    ]
    for kind in KINDS:
        try:
            seed = Seed.load(kind)
        except Exception:
            continue
        lines += [f"### {kind}", "", f"Target: **{seed.total_samples():,}** samples across "
                  f"{len(seed.languages)} languages.", ""]
        if seed.tasks:
            lines.append("| task | tags | share |")
            lines.append("|---|---|---|")
            shares = seed.task_shares()
            for t in sorted(seed.tasks, key=lambda x: -x.share):
                lines.append(f"| `{t.name}` | {', '.join(t.tags)} | {shares[t.name]*100:.1f}% |")
            lines.append("")
    st = status(repo_id)
    if st:
        lines += ["## Shards currently in this repo", "", "| split | languages | shards |", "|---|---|---|"]
        for kind, langs in sorted(st.items()):
            lines.append(f"| {kind} | {len(langs)} | {sum(langs.values())} |")
        lines.append("")
    lines += ["## Provenance", "",
              "Generated with `data-gen/` in the SabiYarn repo, via Together AI / OpenRouter "
              "(`openai/gpt-oss-120b` and `gemma-3-27b`). Every sample carries the `id` that ties it back to a "
              "deterministic plan row in its seed file, so any sample can be traced to the exact brief, "
              "language, task and domain that produced it.", "",
              "Synthetic data in low-resource languages is imperfect. Efik, Urhobo, Fon, Ewe and Fulfulde in "
              "particular should be spot-checked by speakers before being trusted.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--push", default=None, choices=list(KINDS), help="push all local shards of this kind")
    ap.add_argument("--push-seeds", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--card", action="store_true", help="print the dataset card without pushing")
    a = ap.parse_args()
    if a.card:
        print(build_card(a.repo_id))
        return 0
    if a.status:
        st = status(a.repo_id)
        if not st:
            print(f"{a.repo_id}: empty or missing")
        for kind, langs in sorted(st.items()):
            print(f"{kind}: {sum(langs.values())} shards across {len(langs)} languages")
            for lang, n in sorted(langs.items()):
                print(f"    {lang}: {n}")
        return 0
    if a.push_seeds:
        push_seeds(a.repo_id)
    if a.push:
        push_all_local(a.push, a.repo_id)
    if not (a.push or a.push_seeds):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
