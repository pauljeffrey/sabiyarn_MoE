"""Add `<tools>` and `</tools>` to the SabiYarn tokenizer as special tokens (ids 52117 and 52118).

The chat template renders function definitions as `<tools>\n{json}\n{json}\n</tools>` in the system turn. Without these tokens the tags cost several byte-pair
pieces each, on every tool-using sample. The two ids sit inside the SFT embedding growth already planned (52,160 rows hold ids up to 52,159), so the model
needs no further resize.

    python scripts/add_tool_tokens.py --out DIR            # DRY RUN: writes the patched tokenizer files to DIR and checks them. Pushes nothing.
    python scripts/add_tool_tokens.py --out DIR --push-to BeardedMonster/SabiYarn-32k Aletheia-ng/SabiYarn_MoE-280M
        # one commit per repo with tokenizer.json, tokenizer_config.json (incl. its chat_template key), special_tokens_map.json and chat_template.jinja.
        # The patched files are built from --repo (the reference tokenizer) and copied unchanged to every target, so all repos end up identical.

Needs HF_WRITE_TOKEN (or HF_TOKEN) in the environment / data-gen/.env for --push. The Hub repo defaults to BeardedMonster/SabiYarn-32k.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

NEW_TOKENS = ["<tools>", "</tools>"]


def entry(i: int, content: str) -> dict:
    return {"id": i, "content": content, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="BeardedMonster/SabiYarn-32k")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--push-to", nargs="+", default=[], metavar="REPO")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    a.out.mkdir(parents=True, exist_ok=True)
    files = {n: Path(hf_hub_download(a.repo, n, token=token)) for n in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")}
    tj = json.loads(files["tokenizer.json"].read_text(encoding="utf-8"))
    cfg = json.loads(files["tokenizer_config.json"].read_text(encoding="utf-8"))
    sm = json.loads(files["special_tokens_map.json"].read_text(encoding="utf-8"))
    have = {t["content"]: t["id"] for t in tj["added_tokens"]}
    top = max(t["id"] for t in tj["added_tokens"])
    todo = [t for t in NEW_TOKENS if t not in have]
    if not todo:
        print("tokens already present:", {t: have[t] for t in NEW_TOKENS})
    used = set(have.values()) | set(tj["model"]["vocab"].values())
    for k, tok in enumerate(todo, start=1):
        i = top + k
        assert i not in used, f"id {i} is already taken"
        tj["added_tokens"].append(entry(i, tok))
        cfg.setdefault("added_tokens_decoder", {})[str(i)] = {k2: v for k2, v in entry(i, tok).items() if k2 != "id"}
        print(f"{tok} -> {i}")
    template = (Path(__file__).resolve().parents[1] / "sabiyarn" / "chat_template.jinja").read_text(encoding="utf-8")
    cfg["chat_template"] = template
    (a.out / "chat_template.jinja").write_text(template, encoding="utf-8")
    extra = sm.get("additional_special_tokens")
    if isinstance(extra, list):
        sm["additional_special_tokens"] = extra + todo
    cfg_extra = cfg.get("additional_special_tokens")
    if isinstance(cfg_extra, list):
        cfg["additional_special_tokens"] = cfg_extra + todo
    for name, obj in (("tokenizer.json", tj), ("tokenizer_config.json", cfg), ("special_tokens_map.json", sm)):
        (a.out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    # check: the patched files load, the new tags are single tokens, and nothing else moved
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.out))
    ids = tok("<tools>\n{}\n</tools>", add_special_tokens=False)["input_ids"]
    assert ids[0] == tok.convert_tokens_to_ids("<tools>") and ids[-1] == tok.convert_tokens_to_ids("</tools>"), ids
    ref = AutoTokenizer.from_pretrained(a.repo, token=token)
    probe = "<|user|>Tani Biyi Bandele?</s><|assistant|><tool_call>{\"name\": \"f\"}</tool_call>"
    assert tok(probe)["input_ids"] == ref(probe)["input_ids"], "existing ids changed"
    print("checked: new tags are single tokens; existing encodings unchanged; vocabulary size", len(tok))
    # the template as transformers renders it must equal the template as our data build renders it (tools block, tool call, single turn)
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Tani Biyi Bandele?"},
            {"role": "assistant", "content": "<|input_lang|><yor><task_plan><|RAG|></task_plan>", "tool_calls": [{"function": {"name": "f", "arguments": {"q": "Ẹ̀gbọ́n 'x'"}}}]},
            {"role": "tool", "content": "R"}, {"role": "assistant", "content": "<response>ok"}]
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
    got = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data-gen"))
    from sft_new.render import render
    assert got == render(msgs, tools=tools), "transformers renders the template differently from the data build"
    print("checked: apply_chat_template(tools=...) is byte-identical to the data build's rendering")
    for repo in a.push_to:
        from huggingface_hub import CommitOperationAdd
        api = HfApi(token=token)
        ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(a.out / n))
               for n in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")]
        info = api.create_commit(repo_id=repo, operations=ops, commit_message="chat template v2 (standard JSON tool calls, one turn per tool exchange, <tools> block) + <tools>/</tools> special tokens 52117/52118")
        print("pushed to", repo, info.oid[:10])
    if not a.push_to:
        print("dry run: nothing pushed")


if __name__ == "__main__":
    main()
