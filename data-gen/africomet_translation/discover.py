"""Find which columns of which Hugging Face datasets actually contain our languages.

For every candidate dataset (language-tagged or name-matched), read its first rows through the datasets-server API
(nothing is downloaded), run GlotLID on every text column, and record the language each column really contains plus
the dataset's row count. This answers "which column is the Efik one?" without trusting dataset cards.

    python -m africomet_translation.discover --candidates candidates.json --out discovered.json --cache CACHE
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://datasets-server.huggingface.co"
OUR = {"pcm_Latn", "efi_Latn", "ibb_Latn", "fon_Latn", "urh_Latn", "twi_Latn", "aka_Latn", "ewe_Latn", "eng_Latn",
       "yor_Latn", "hau_Latn", "ibo_Latn"}
LANG_CONFIG_PAT = re.compile(r"(pcm|efi|ibb|fon|urh|twi|aka|ewe|(?<![a-z])ee(?![a-z])|(?<![a-z])tw(?![a-z])|"
                             r"(?<![a-z])ak(?![a-z])|pidgin|efik|ibibio|urhobo)", re.I)
SKIP_ID = re.compile(r"africomet-translation|(^|/)(flores|sib200|afrimmlu|afrixnli|afrimgsm|masakhaner|belebele|mmlu|"
                     r"global-piqa|hypa-voices-asr|hypa_aime)", re.I)


def _get(url: str, token: str, params: dict | None = None, retries: int = 3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=40)
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
            return r.json() if r.status_code == 200 else {"error": r.status_code, "msg": r.text[:120]}
        except requests.RequestException as e:
            time.sleep(2)
    return {"error": "network"}


def text_columns(rows: list[dict]) -> dict[str, list[str]]:
    """column name -> sample strings. Dict-valued cells (e.g. {'en':..., 'ee':...}) become 'col.key' columns."""
    cols: dict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        for k, v in r["row"].items():
            if isinstance(v, str):
                cols[k].append(v)
            elif isinstance(v, dict) and v and all(isinstance(x, str) for x in v.values()):
                for kk, vv in v.items():
                    cols[f"{k}.{kk}"].append(vv)
    return {k: [s for s in v if len(s) >= 15][:40] for k, v in cols.items() if sum(len(s) >= 15 for s in v) >= 5}


def inspect_dataset(ds: str, token: str) -> dict:
    out = {"dataset": ds, "views": []}
    sp = _get(f"{API}/splits", token, {"dataset": ds})
    if "splits" not in sp:
        out["error"] = f"splits: {sp.get('error')} {sp.get('msg', '')}"
        return out
    splits = sp["splits"]
    configs = sorted({s["config"] for s in splits})
    if len(configs) > 12:
        configs = [c for c in configs if LANG_CONFIG_PAT.search(c)][:30]
    size = _get(f"{API}/size", token, {"dataset": ds})
    nrows = {}
    for s in (size.get("size", {}) or {}).get("splits", []) or []:
        nrows[(s["config"], s["split"])] = s.get("num_rows")
    for c in configs:
        sps = [s["split"] for s in splits if s["config"] == c]
        split = "train" if "train" in sps else sps[0]
        fr = _get(f"{API}/first-rows", token, {"dataset": ds, "config": c, "split": split})
        if "rows" not in fr:
            out["views"].append({"config": c, "split": split, "error": fr.get("error")})
            continue
        out["views"].append({"config": c, "split": split, "n_rows": nrows.get((c, split)),
                             "columns": text_columns(fr["rows"])})
    return out


def label_columns(views: list[dict], lid) -> None:
    """Replace each view's sample strings with {label share} -- the language the column really contains."""
    for v in views:
        cols = v.pop("columns", None) or {}
        v["lang"] = {}
        for name, texts in cols.items():
            labels = lid(texts)
            c = collections.Counter(labels)
            top, n = c.most_common(1)[0]
            v["lang"][name] = {"label": top, "share": round(n / len(labels), 2), "sample": texts[0][:100]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--extra", default="", help="comma-separated extra dataset ids")
    a = ap.parse_args()
    os.environ["HF_HUB_CACHE"] = a.cache
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from africomet_translation.build import hf_token
    token = hf_token()
    os.environ["HF_TOKEN"] = token
    from africomet_translation import lid
    cand = json.load(open(a.candidates))
    ids = set()
    for l, d in cand.items():
        ids |= set(d["tagged"]) | set(d["named"])
    ids |= {x for x in a.extra.split(",") if x}
    ids = sorted(i for i in ids if not SKIP_ID.search(i))
    print(f"{len(ids)} datasets to inspect", flush=True)
    results = {}
    outp = Path(a.out)
    if outp.exists():
        results = {r["dataset"]: r for r in json.load(open(outp))}
    todo = [i for i in ids if i not in results]
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for n, r in enumerate(ex.map(lambda d: inspect_dataset(d, token), todo), 1):
            label_columns(r["views"], lid.top_labels)
            results[r["dataset"]] = r
            if n % 50 == 0:
                outp.write_text(json.dumps(list(results.values())))
                print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    outp.write_text(json.dumps(list(results.values())))
    print("done", len(results), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
