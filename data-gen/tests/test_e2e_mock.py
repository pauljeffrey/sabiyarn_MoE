"""End-to-end offline loops: build -> mock_generate -> postprocess -> stats (-> judge).

Everything runs on a tiny per-language count against a temp data dir. The
"bad output" tests overwrite chosen mock responses with hand-made defective
ones and check that the quality filters really drop them, for the right reason.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import pytest

from config.corpus_config import TARGET_LANGUAGE_CODES
from config.settings import DataPaths
from conftest import ROOT, tiny
from pipeline import judge
from pipeline.build_corpus import build_kind
from pipeline.postprocess_corpus import kind_files, load_manifest, process_kind
from scripts import mock_generate
from scripts.mock_generate import _sentences

KINDS = ["pretrain", "sft", "dpo"]
DELETE = object()


# ------------------------------------------------------------------ helpers


def build_and_mock(cfg, data_dir: Path, **build_kw):
    build_kind(cfg, data_dir, **build_kw)
    paths = DataPaths(data_dir)
    mock_generate.run(paths.batch_input, paths.batch_output, [cfg.kind])
    return paths


def rewrite_outputs(paths: DataPaths, kind: str, edits: dict[str, object]) -> None:
    """Edit mock responses in place. edit = DELETE | str (raw content) | dict (new value) | callable(value)->value."""
    for path in kind_files(paths.batch_output, kind, ".output.jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        out = []
        for row in rows:
            edit = edits.get(row["custom_id"])
            if edit is DELETE:
                continue
            if edit is not None:
                content = row["response"]["body"]["choices"][0]["message"]["content"]
                if callable(edit):
                    content = json.dumps(edit(json.loads(content)), ensure_ascii=False)
                elif isinstance(edit, dict):
                    content = json.dumps(edit, ensure_ascii=False)
                else:
                    content = edit
                row["response"]["body"]["choices"][0]["message"]["content"] = content
            out.append(row)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")


def ids_where(paths: DataPaths, kind: str, pred: Callable[[dict], bool], language: Optional[str] = None) -> list[str]:
    manifest = load_manifest(kind, paths.batch_input)
    return sorted(cid for cid, e in manifest.items() if pred(e["context"]) and (language is None or e["language"] == language))


def with_(**over):
    def edit(v):
        v = dict(v)
        v.update(over)
        return v
    return edit


def hits(stats, language: str) -> dict[str, int]:
    """reason prefix (before ':') -> count of records that hit it, for one language."""
    out: dict[str, int] = {}
    for reason, n in stats.reason_hits[language].items():
        out[reason] = out.get(reason, 0) + n
    return out


ENGLISH_PARA = (
    "The farmers in the village are going to the market because they have been told that the price of maize will not be "
    "low this year and there is a lot of work to do before the rains come back to the fields where they will plant again"
)


# ------------------------------------------------------------------ clean loops


@pytest.mark.parametrize("kind", KINDS)
def test_clean_loop_keeps_everything_and_writes_expected_records(presets, data_dir, kind):
    cfg = tiny(presets[kind], 3)
    paths = build_and_mock(cfg, data_dir)
    records, stats = process_kind(cfg, data_dir)

    assert len(records) == 36
    assert stats.totals() == {"generated": 36, "kept": 36, "dropped": 0}
    assert {r["language"] for r in records} == set(TARGET_LANGUAGE_CODES)
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids) and all(i.startswith(f"{kind}__") for i in ids)

    out_file = paths.processed / f"{kind}.jsonl"
    on_disk = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert on_disk == records

    r = records[0]
    if kind == "pretrain":
        assert {"id", "language", "text", "title", "domain", "subtopic", "genre"} <= set(r)
        assert r["text"] and r["title"] and r["n_words"] > 60
    elif kind == "sft":
        assert {"id", "language", "task", "domain", "instruction", "input", "response", "messages", "text"} <= set(r)
        for rec in records:
            assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
            expected_user = rec["instruction"] + ("\n\n" + rec["input"] if rec["input"] else "")
            assert rec["messages"][1]["content"] == expected_user
            assert rec["messages"][2]["content"] == rec["response"]
            assert rec["text"].startswith("<s>") and "<|user|>" in rec["text"] and "<|assistant|>" in rec["text"]
            assert rec["response"] in rec["text"] and rec["instruction"] in rec["text"]
        assert any(rec["input"] for rec in records) and any(not rec["input"] for rec in records)  # both shapes occur
    else:
        assert {"id", "language", "task", "domain", "instruction", "input", "prompt_messages", "chosen", "rejected", "rejection_type"} <= set(r)
        for rec in records:
            assert [m["role"] for m in rec["prompt_messages"]] == ["system", "user"]
            assert rec["chosen"] != rec["rejected"] and rec["chosen"] and rec["rejected"]

    summary = json.loads((paths.reports / f"{kind}_summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["kept"] == 36
    for lang in TARGET_LANGUAGE_CODES:
        lr = summary["languages"][lang]
        assert lr["kept"] == 3 and lr["generated"] == 3
        assert lr["domain_counts"] and "coverage" in lr and "pair" in lr["coverage"]
        assert "entropy_norm" in lr["coverage"]["pair"]
    assert "TOTAL" in stats.as_table()


def test_mock_outputs_route_by_schema_and_fill_dpo_flaw_shapes(presets, data_dir):
    cfg = tiny(presets["dpo"], 40, ["yor"])
    paths = build_and_mock(cfg, data_dir)
    manifest = load_manifest("dpo", paths.batch_input)
    out = {json.loads(l)["custom_id"]: json.loads(json.loads(l)["response"]["body"]["choices"][0]["message"]["content"])
           for l in (paths.batch_output / "dpo.output.jsonl").read_text(encoding="utf-8").splitlines()}
    for cid, v in out.items():
        assert v["rejection_type"] == manifest[cid]["context"]["rejection_type"]
        is_json = manifest[cid]["context"]["attributes"]["task"] in ("ner_extraction", "info_extraction_json")
        if is_json:
            json.loads(v["chosen"]); json.loads(v["rejected"])  # extraction tasks stay valid JSON
            assert v["chosen"] != v["rejected"]
            continue
        if v["rejection_type"] == "incomplete":
            assert len(v["rejected"]) < len(v["chosen"])
        if v["rejection_type"] == "rambling_verbose":
            assert len(v["rejected"]) > 1.5 * len(v["chosen"])


def test_manifest_coverage_audit_after_generation(presets, data_dir):
    """The manifest `context` alone lets you audit coverage without any API output."""
    from sampling.sampler import coverage_report

    cfg = tiny(presets["pretrain"], 700, ["yor", "efi"])
    build_kind(cfg, data_dir)
    manifest = load_manifest("pretrain", DataPaths(data_dir).batch_input)
    for lang in ("yor", "efi"):
        attrs = [e["context"]["attributes"] for e in manifest.values() if e["language"] == lang]
        rep = coverage_report(attrs)
        assert len(attrs) == 700
        assert rep["attributes"]["pair"]["max"] <= 2 and rep["attributes"]["pair"]["entropy_norm"] > 0.99
        assert rep["attributes"]["domain"]["distinct"] == len(__import__("sampling.taxonomy", fromlist=["DOMAINS"]).DOMAINS)


def test_build_audit_reports_even_coverage(presets, data_dir, capsys):
    from pipeline.build_corpus import audit_manifest, print_audit

    cfg = tiny(presets["sft"], 700, ["yor", "fon"])
    build_kind(cfg, data_dir)
    audit = audit_manifest("sft", data_dir)
    assert set(audit) == {"yor", "fon"}
    for a in audit.values():
        assert a["requests"] == 700 and a["pair_max_min"][1] - a["pair_max_min"][0] <= 1 and a["pair_entropy"] > 0.99
    print_audit("sft", data_dir)
    assert "requested-coverage audit" in capsys.readouterr().out


# ------------------------------------------------------------------ bad outputs: pretrain


def test_pretrain_bad_documents_are_dropped(presets, data_dir):
    cfg = tiny(presets["pretrain"], 16, ["yor"])
    paths = build_and_mock(cfg, data_dir)
    cid = lambda i: f"pretrain__yor__{i:05d}__base"  # noqa: E731
    long_ok = lambda seed: _sentences(random.Random(seed), "yor", 260)  # noqa: E731

    edits = {
        cid(0): with_(text="ka ba to ne " * 120),  # degenerate repetition
        cid(1): with_(text=ENGLISH_PARA + ". " + ENGLISH_PARA + " today"),  # English
        cid(2): lambda v: {**v, "text": "Here is a document about the topic:\n" + v["text"]},  # meta text
        cid(3): lambda v: {**v, "text": v["text"] + " [...] " + long_ok(3)},  # placeholder
        cid(4): with_(language_self_check=False),
        cid(5): with_(text=_sentences(random.Random(5), "yor", 12)),  # far too short
        cid(6): lambda v: {**v, "text": "# Àkọ́lé\n" + v["text"]},  # markdown heading
        cid(7): with_(text="Привет мир это тест неправильного алфавита " * 12),  # wrong script
        cid(8): "this is not json",
        cid(9): {"title": "only a title"},  # schema-invalid
        cid(10): DELETE,  # missing output
    }
    dup_text = long_ok(99)
    edits[cid(11)] = with_(text=dup_text)
    edits[cid(12)] = with_(text=dup_text)  # exact duplicate of 11 -> one of them dropped
    rewrite_outputs(paths, "pretrain", edits)
    # an API-level error row
    out = paths.batch_output / "pretrain.output.jsonl"
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["custom_id"] == cid(13):
            row["response"], row["error"] = None, {"code": "server_error"}
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    records, stats = process_kind(cfg, data_dir)
    kept_ids = {r["id"] for r in records}
    h = hits(stats, "yor")
    for expected in ("repetition_ngram:text", "english_leak:text", "meta_text:text", "placeholder:text", "self_check_failed",
                     "too_short:text", "markdown_heading:text", "wrong_script:text", "invalid_json", "schema_invalid",
                     "missing_output", "api_error", "duplicate_exact"):
        assert h.get(expected, 0) >= 1, f"{expected} not hit; got {h}"
    for i in list(range(0, 11)) + [13]:
        assert cid(i) not in kept_ids, f"bad document {i} survived"
    assert (cid(11) in kept_ids) != (cid(12) in kept_ids)  # exactly one of the exact duplicates kept
    assert cid(14) in kept_ids and cid(15) in kept_ids  # untouched documents survive
    assert stats.totals()["kept"] + stats.totals()["dropped"] == 16


# ------------------------------------------------------------------ bad outputs: sft


def test_sft_bad_examples_are_dropped(presets, data_dir):
    cfg = tiny(presets["sft"], 60, ["yor", "hau"])
    paths = build_and_mock(cfg, data_dir)
    need_input = ids_where(paths, "sft", lambda c: c["input_mode"] == "required" and not c["english_fields"], "yor")
    no_english = ids_where(paths, "sft", lambda c: not c["english_fields"] and c["attributes"]["task"] not in ("safe_decline", "ner_extraction", "info_extraction_json"), "yor")
    assert len(need_input) >= 3 and len(no_english) >= 12

    a = need_input[0]
    bad = [i for i in no_english if i != a]
    edits = {
        a: with_(input=""),  # input required but empty
        bad[0]: with_(response="I'm sorry, but I cannot help with that request. " + _sentences(random.Random(1), "yor", 20)),
        bad[1]: with_(response=ENGLISH_PARA),
        bad[2]: with_(confidence="low"),
        bad[3]: with_(response="Ọjà " + _sentences(random.Random(2), "yor", 15) + " [...]"),
        bad[4]: with_(response="```\n" + _sentences(random.Random(3), "yor", 15) + "\n```"),
        bad[5]: with_(response="Here is the answer: " + _sentences(random.Random(4), "yor", 15)),
        bad[6]: with_(response="ka ba to ne " * 50),
        bad[7]: with_(response=""),
        bad[8]: with_(instruction=ENGLISH_PARA),
    }
    # near-duplicate instructions inside one language
    dup_a, dup_b = bad[9], bad[10]
    same_instruction = _sentences(random.Random(77), "yor", 12).rstrip(".") + "?"
    edits[dup_a] = with_(instruction=same_instruction, input="")
    edits[dup_b] = with_(instruction=same_instruction, input="")
    rewrite_outputs(paths, "sft", edits)

    records, stats = process_kind(cfg, data_dir)
    kept = {r["id"] for r in records}
    h = hits(stats, "yor")
    for expected in ("missing_input", "refusal:response", "english_leak:response", "low_confidence", "placeholder:response",
                     "markdown_fence:response", "meta_text:response", "repetition_ngram:response", "empty_field:response",
                     "english_leak:instruction"):
        assert h.get(expected, 0) >= 1, f"{expected} not hit; got {h}"
    for cid in [a] + bad[:9]:
        assert cid not in kept
    assert (dup_a in kept) != (dup_b in kept)
    assert stats.drop_first_reason["yor"]  # per-language drop reasons are reported
    good_yor = [i for i in ids_where(paths, "sft", lambda c: True, "yor") if i not in edits]
    assert all(i in kept for i in good_yor)


def test_sft_cross_language_duplicates_are_dropped(presets, data_dir):
    cfg = tiny(presets["sft"], 6, ["yor", "hau"])
    paths = build_and_mock(cfg, data_dir)
    yor = ids_where(paths, "sft", lambda c: c["input_mode"] == "empty", "yor")[0]
    hau = ids_where(paths, "sft", lambda c: c["input_mode"] == "empty", "hau")[0]
    shared = _sentences(random.Random(5), "yor", 12).rstrip(".") + "?"
    rewrite_outputs(paths, "sft", {yor: with_(instruction=shared, input=""), hau: with_(instruction=shared, input="")})
    records, stats = process_kind(cfg, data_dir)
    kept = {r["id"] for r in records}
    assert (yor in kept) != (hau in kept)
    assert sum(hits(stats, l).get("cross_language_duplicate_exact", 0) + hits(stats, l).get("cross_language_duplicate_near", 0) for l in ("yor", "hau")) == 1


# ------------------------------------------------------------------ bad outputs: dpo


def test_dpo_bad_pairs_are_dropped_and_flaw_exemptions_hold(presets, data_dir):
    cfg = tiny(presets["dpo"], 60, ["yor"])
    paths = build_and_mock(cfg, data_dir)
    no_eng = lambda c: not c["english_fields"] and c["attributes"]["task"] != "safe_decline"  # noqa: E731
    is_json = lambda c: c["attributes"]["task"] in ("ner_extraction", "info_extraction_json")  # noqa: E731

    def pick(rtype: str, n: int = 1):
        got = ids_where(paths, "dpo", lambda c: c["rejection_type"] == rtype and no_eng(c) and not is_json(c), "yor")
        assert len(got) >= n, f"not enough {rtype} pairs"
        return got

    # pairs whose flaw type has no length/language exemption: safe to corrupt in the ways below
    plain = ids_where(paths, "dpo", lambda c: not c["length_related"] and c["rejection_type"] != "wrong_language" and no_eng(c) and not is_json(c), "yor")
    assert len(plain) >= 8
    same_id, empty_id, lowconf_id, eng_rej_id, ratio_id, mism_id, chosen_refusal_id = plain[:7]
    incomplete_id = pick("incomplete")[0]
    wronglang_id = pick("wrong_language")[0]
    refusal_type_id = pick("unhelpful_refusal")[0]
    rambling_id = pick("rambling_verbose")[0]

    def same_pair(v):
        return {**v, "rejected": v["chosen"]}

    edits = {
        same_id: same_pair,
        empty_id: with_(rejected=""),
        lowconf_id: with_(chosen_confidence="low"),
        eng_rej_id: with_(rejected=ENGLISH_PARA),
        ratio_id: with_(rejected="ka"),
        mism_id: with_(rejection_type="factual_error"),
        chosen_refusal_id: lambda v: {**v, "chosen": "I'm sorry, but I cannot help with that. " + v["chosen"]},
        # exemptions: these SHOULD survive
        incomplete_id: with_(rejected="ka ba"),  # tiny rejected is fine for a length-related flaw
        wronglang_id: with_(rejected=ENGLISH_PARA),  # English rejected is the point of wrong_language
        refusal_type_id: lambda v: {**v, "rejected": "I'm sorry, but I cannot help with that request. " + v["rejected"]},
        rambling_id: lambda v: {**v, "rejected": v["rejected"] + " " + _sentences(random.Random(9), "yor", 200)},
    }
    rewrite_outputs(paths, "dpo", edits)
    records, stats = process_kind(cfg, data_dir)
    kept = {r["id"] for r in records}
    h = hits(stats, "yor")

    for cid in (same_id, empty_id, lowconf_id, eng_rej_id, ratio_id, mism_id, chosen_refusal_id):
        assert cid not in kept, f"{cid} should have been dropped"
    for expected in ("chosen_eq_rejected", "empty_field:rejected", "low_confidence", "rejected_wrong_language", "length_ratio",
                     "rejection_type_mismatch", "refusal:chosen"):
        assert h.get(expected, 0) >= 1, f"{expected} not hit; got {h}"
    for cid in (incomplete_id, wronglang_id, refusal_type_id, rambling_id):
        assert cid in kept, f"{cid} (length/language-related flaw) wrongly dropped"
    kept_by_id = {r["id"]: r for r in records}
    assert kept_by_id[wronglang_id]["rejection_type"] == "wrong_language"


# ------------------------------------------------------------------ splitting, submit, judge


def test_file_splitting_and_downstream_globs(presets, data_dir, monkeypatch):
    cfg = tiny(presets["sft"], 4)  # 48 requests
    written = build_kind(cfg, data_dir, max_requests_per_file=20)
    paths = DataPaths(data_dir)
    assert [p.name for p in written] == ["sft__part0.jsonl", "sft__part1.jsonl", "sft__part2.jsonl"]
    assert not (paths.batch_input / "sft.jsonl").exists()
    counts = [sum(1 for _ in p.open(encoding="utf-8")) for p in written]
    assert counts == [20, 20, 8]
    # byte-limit splitting
    written = build_kind(cfg, data_dir, max_file_bytes=20_000)
    assert len(written) > 3 and all(p.stat().st_size <= 30_000 for p in written)
    # a rebuild removes stale parts and single file uses the plain name
    written = build_kind(cfg, data_dir)
    assert [p.name for p in written] == ["sft.jsonl"]
    assert not list(paths.batch_input.glob("sft__part*"))

    # submit's file discovery must not pick side-car manifests up as batch inputs
    build_kind(cfg, data_dir, max_requests_per_file=20)
    from pipeline import submit_batch

    monkeypatch.setattr(submit_batch, "BATCH_INPUT_DIR", paths.batch_input)
    files = submit_batch._find_batch_files(["sft"])
    assert [f.name for f in files] == ["sft__part0.jsonl", "sft__part1.jsonl", "sft__part2.jsonl"]

    # postprocess reads every part
    mock_generate.run(paths.batch_input, paths.batch_output, ["sft"])
    records, stats = process_kind(cfg, data_dir)
    assert len(records) == 48 and stats.totals()["generated"] == 48


def test_judge_build_mock_apply_loop(presets, data_dir):
    cfg = tiny(presets["sft"], 10, ["yor", "hau", "efi"])
    paths = build_and_mock(cfg, data_dir)
    records, _ = process_kind(cfg, data_dir)
    assert len(records) == 30

    written = judge.run_build(cfg, data_dir, fraction=0.5, seed=None)
    assert [p.name for p in written] == ["judge_sft.jsonl"]
    lines = [json.loads(l) for l in written[0].read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 15  # 50% of each language
    assert all(l["body"]["response_format"]["json_schema"]["name"] == "judge_scores" and l["body"]["response_format"]["json_schema"]["strict"] for l in lines)
    assert all(l["custom_id"].startswith("judge_sft__") for l in lines)
    assert lines[0]["body"]["model"] == "gpt-4o-mini"
    again = judge.build_judge_requests("sft", records, cfg, 0.5, cfg.seed)
    assert [s.custom_id for s in again] == [l["custom_id"] for l in lines]  # deterministic sample

    mock_generate.run(paths.batch_input, paths.batch_output, ["judge_sft"])
    thresholds = {k: float(v) for k, v in cfg.judge["min_scores"].items()}
    kept, report = judge.run_apply(cfg, data_dir, thresholds)
    judged_total = sum(r["judged"] for r in report["languages"].values())
    assert judged_total == 15
    assert sum(r["passed"] + r["failed"] for r in report["languages"].values()) == 15
    assert len(kept) == 30 - sum(r["failed"] for r in report["languages"].values())  # unjudged records are kept by default
    judged_ids = {e["context"]["record_id"] for e in judge.load_manifest("judge_sft", paths.batch_input).values()}
    assert all(("judge_scores" in r) == (r["id"] in judged_ids) for r in kept)  # scores attached to judged survivors only
    assert (paths.processed / "sft.judged.jsonl").exists()
    assert (paths.reports / "judge_sft_summary.json").exists()

    # stricter thresholds filter more; --drop-unjudged removes the unsampled records
    strict = {**thresholds, "fluency": 5.0, "language_correctness": 5.0}
    kept_strict, _ = judge.run_apply(cfg, data_dir, strict)
    assert len(kept_strict) <= len(kept)
    kept_only_judged, rep = judge.run_apply(cfg, data_dir, thresholds, drop_unjudged=True)
    assert len(kept_only_judged) == sum(r["passed"] for r in rep["languages"].values())


def test_judge_dpo_requires_chosen_better(presets, data_dir):
    cfg = tiny(presets["dpo"], 5, ["yor"])
    paths = build_and_mock(cfg, data_dir)
    process_kind(cfg, data_dir)
    judge.run_build(cfg, data_dir, fraction=1.0, seed=None)
    mock_generate.run(paths.batch_input, paths.batch_output, ["judge_dpo"])
    first = judge.load_manifest("judge_dpo", paths.batch_input)
    victim = sorted(first)[0]
    rewrite_outputs(paths, "judge_dpo", {victim: lambda v: {**v, "language_correctness": 5, "fluency": 5, "factuality": 5,
                                                          "instruction_following": 5, "usefulness": 5, "chosen_better_than_rejected": False}})
    kept, report = judge.run_apply(cfg, data_dir, {k: 1.0 for k in judge.SCORE_KEYS})
    assert first[victim]["context"]["record_id"] not in {r["id"] for r in kept}
    assert report["languages"]["yor"]["fail_reasons"].get("chosen_better_than_rejected") == 1


# ------------------------------------------------------------------ CLI


def _run(args: list[str], env_extra: Optional[dict] = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    env.pop("OPENAI_API_KEY", None)
    return subprocess.run([sys.executable, str(ROOT / "run.py"), *args], capture_output=True, text=True, cwd=ROOT, env=env, timeout=300)


def test_cli_build_mock_postprocess_judge_and_submit_dry_run(tmp_path):
    d = str(tmp_path / "cli")
    env = {"DATA_GEN_OUTPUT_DIR": str(tmp_path / "unused_default")}
    r = _run(["estimate", "--kind", "sft", "--per-language", "5", "--sample", "3"], env)
    assert r.returncode == 0 and "== sft" in r.stdout and "No API calls" in r.stdout, r.stderr
    r = _run(["build", "--kind", "dpo", "--config", "configs/dpo.yaml", "--per-language", "2", "--data-dir", d, "--languages", "yor,hau"], env)
    assert r.returncode == 0, r.stderr
    assert (Path(d) / "batch_input" / "dpo.jsonl").exists()
    r = _run(["build", "--kind", "pretrain", "--dry-run", "--per-language", "2"], env)
    assert r.returncode == 0 and "== pretrain" in r.stdout
    assert _run(["mock", "--data-dir", d], env).returncode == 0
    r = _run(["postprocess", "--kind", "dpo", "--data-dir", d], env)
    assert r.returncode == 0 and "kept 4" in r.stdout, r.stdout + r.stderr
    assert _run(["judge", "build", "--kind", "dpo", "--fraction", "1.0", "--data-dir", d], env).returncode == 0
    assert _run(["mock", "--data-dir", d, "--only", "judge_dpo"], env).returncode == 0
    r = _run(["judge", "apply", "--kind", "dpo", "--data-dir", d, "--min", "fluency=1"], env)
    assert r.returncode == 0 and "judged=" in r.stdout, r.stderr
    # submit is a dry run without --confirm (no API key present, so a real call would fail loudly)
    r = _run(["submit", "--kind", "dpo"], {"DATA_GEN_OUTPUT_DIR": d})
    assert r.returncode == 0 and "Dry run only" in r.stdout and "dpo.jsonl: 4 requests" in r.stdout, r.stdout + r.stderr


def test_legacy_six_task_smoke_flow_still_works(tmp_path):
    env = {"DATA_GEN_OUTPUT_DIR": str(tmp_path / "legacy"), "DATA_GEN_PER_CELL": "2"}
    r = _run(["build"], env)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "legacy" / "batch_input" / "rag.jsonl").exists()
    assert (tmp_path / "legacy" / "batch_input" / "translation.jsonl").exists()
    r = _run(["mock"], env)
    assert r.returncode == 0, r.stderr
    r = _run(["postprocess"], env)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "legacy" / "processed" / "all.jsonl").stat().st_size > 0
    assert "Totals:" in r.stdout
