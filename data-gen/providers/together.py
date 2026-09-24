"""Together AI. OpenAI-compatible chat completions, plus a real batch queue at ~50% price.

    python -m providers.together --check                      # 1 live request, prints the reply
    from providers import get_provider; p = get_provider("together", "openai/gpt-oss-120b")

Batch flow (half price, 24h window): submit_batch writes a JSONL of OpenAI-shaped request lines, uploads it
with purpose="batch-api", creates the batch, and returns its id. poll_batch reports status; fetch_batch
downloads the output file and parses it back into Response objects. The uploaded file and the returned
output are both kept on disk so a crashed run never has to pay twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from providers.base import Provider, Request, Response


class TogetherProvider(Provider):
    name = "together"
    base_url = "https://api.together.xyz/v1"
    env_key = "TOGETHER_API_KEY"
    supports_batch = True
    # USD per 1M (input, output). VERIFY at https://www.together.ai/pricing -- these move.
    pricing = {
        "gpt-oss-120b": (0.15, 0.60),
        "gpt-oss-20b": (0.05, 0.20),
        "Qwen3-235B": (0.20, 0.60),
        "Llama-3.3-70B": (0.88, 0.88),
        "gemma-3-27b": (0.20, 0.30),
        "gemma-2-27b": (0.80, 0.80),
    }
    default_pricing = (0.20, 0.60)

    def _batch_line(self, req: Request) -> dict[str, Any]:
        return {"custom_id": req.custom_id, "method": "POST", "url": "/v1/chat/completions",
                "body": self._payload(req)}

    def submit_batch(self, requests: Iterable[Request], workdir: Path) -> str:
        workdir.mkdir(parents=True, exist_ok=True)
        reqs = list(requests)
        path = workdir / "batch_input.jsonl"
        meta = {r.custom_id: r.metadata for r in reqs}
        path.write_text("".join(json.dumps(self._batch_line(r), ensure_ascii=False) + "\n" for r in reqs),
                        encoding="utf-8")
        (workdir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        client = self._client()
        with path.open("rb") as fh:
            uploaded = client.files.create(file=fh, purpose="batch-api")
        batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions",
                                      completion_window="24h")
        (workdir / "batch.json").write_text(
            json.dumps({"batch_id": batch.id, "input_file_id": uploaded.id, "n": len(reqs),
                        "model": self.model}, indent=2), encoding="utf-8")
        print(f"[together] submitted {len(reqs)} requests -> batch {batch.id} (~50% price, 24h window)")
        return batch.id

    def poll_batch(self, batch_id: str) -> dict[str, Any]:
        b = self._client().batches.retrieve(batch_id)
        counts = getattr(b, "request_counts", None)
        return {"id": b.id, "status": getattr(b, "status", "unknown"),
                "completed": getattr(counts, "completed", 0) if counts else 0,
                "failed": getattr(counts, "failed", 0) if counts else 0,
                "total": getattr(counts, "total", 0) if counts else 0,
                "output_file_id": getattr(b, "output_file_id", None),
                "error_file_id": getattr(b, "error_file_id", None)}

    def fetch_batch(self, batch_id: str, out_path: Path) -> list[Response]:
        st = self.poll_batch(batch_id)
        if st["status"] != "completed":
            raise RuntimeError(f"batch {batch_id} is {st['status']}, not completed")
        client = self._client()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = client.files.content(st["output_file_id"])
        data = content.read() if hasattr(content, "read") else content.content
        out_path.write_bytes(data)
        meta_path = out_path.parent / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return [_parse_batch_line(json.loads(l), meta, self.model)
                for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _parse_batch_line(row: dict, meta: dict, model: str) -> Response:
    cid = row.get("custom_id", "")
    md = meta.get(cid, {})
    body = (row.get("response") or {}).get("body") or {}
    if row.get("error") or not body.get("choices"):
        return Response(cid, "", False, json.dumps(row.get("error") or "no choices")[:400], metadata=md)
    msg = body["choices"][0].get("message", {})
    usage = body.get("usage", {})
    return Response(cid, msg.get("content") or "", True, None, usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0), body.get("model", model),
                    msg.get("tool_calls"), md)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    p = TogetherProvider(a.model)
    if a.check:
        r = p.complete(Request("check", [{"role": "user", "content": "Reply with exactly: ready"}], max_tokens=16))
        print("ok" if r.ok else "FAILED", "->", (r.text or r.error)[:200])
        print(p.usage_line())
