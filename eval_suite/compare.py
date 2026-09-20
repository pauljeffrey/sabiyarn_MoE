"""Compare two eval runs: python -m eval_suite.compare eval_results/A/results.json eval_results/B/results.json"""

from __future__ import annotations

import json
import sys


def flatten(results: dict) -> dict:
    flat = {}
    for task, by_lang in results.items():
        for lang, m in by_lang.items():
            if "skipped" in m:
                continue
            for k, v in m.items():
                if isinstance(v, dict):  # translation directions
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)) and kk != "n":
                            flat[(task, lang, f"{k} {kk}")] = vv
                elif isinstance(v, (int, float)) and k not in ("n", "shots", "chance", "majority_baseline"):
                    flat[(task, lang, k)] = v
    return flat


def main() -> None:
    a_path, b_path = sys.argv[1:3]
    a = flatten(json.load(open(a_path))["results"])
    b = flatten(json.load(open(b_path))["results"])
    print(f"{'task':22s}{'lang':6s}{'metric':22s}{'A':>9s}{'B':>9s}{'B-A':>9s}")
    for key in sorted(set(a) & set(b)):
        print(f"{key[0]:22s}{key[1]:6s}{key[2]:22s}{a[key]:9.3f}{b[key]:9.3f}{b[key] - a[key]:+9.3f}")


if __name__ == "__main__":
    main()
