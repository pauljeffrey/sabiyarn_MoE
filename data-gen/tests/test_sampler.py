"""Sampler balance, determinism, compatibility, weights and reporting."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from config.corpus_config import TARGET_LANGUAGE_CODES, parse_config
from generators import dpo, pretrain, sft_tasks
from sampling import taxonomy as t
from sampling.sampler import AttributeSpec, CoverageSampler, coverage_report, derive_seed, merge_weights, normalized_entropy, pair_weights_from_groups


@pytest.fixture(scope="module")
def cfgs():
    return {k: parse_config({"kind": k, "samples_per_language": {"default": 10}}) for k in ("pretrain", "sft", "dpo")}


def _pretrain_targets():
    return {
        "genre": t.option_weights(t.GENRES), "register": t.option_weights(t.REGISTERS), "audience": t.option_weights(t.AUDIENCES),
        "length_bucket": t.option_weights(t.LENGTH_BUCKETS), "perspective": t.option_weights(t.PERSPECTIVES),
        "era": t.option_weights(t.ERAS), "difficulty": t.option_weights(t.DIFFICULTIES),
    }


# ------------------------------------------------------------------ primary key


@pytest.mark.parametrize("factor", [0.4, 1.0, 3.3, 7.9])
def test_primary_pair_coverage_is_balanced(cfgs, factor):
    sampler = pretrain.build_pretrain_sampler(cfgs["pretrain"], "yor")
    p = len(sampler.pairs)
    n = int(p * factor)
    draws = sampler.draw_many(n)
    counts = Counter((d["domain"], d["subtopic"]) for d in draws)
    hi = math.ceil(n / p) + 1
    assert max(counts.values()) <= hi
    if n >= p:
        assert len(counts) == p  # every pair used
        assert min(counts.values()) >= math.floor(n / p) - 1
    else:
        assert max(counts.values()) == 1  # no pair repeats before all pairs were used once


def test_cycle_uses_every_pair_before_repeating(cfgs):
    sampler = pretrain.build_pretrain_sampler(cfgs["pretrain"], "hau")
    p = len(sampler.pairs)
    first = [(d["domain"], d["subtopic"]) for d in sampler.draw_many(p)]
    second = [(d["domain"], d["subtopic"]) for d in sampler.draw_many(p)]
    assert len(set(first)) == p and len(set(second)) == p
    assert first != second  # reshuffled each cycle
    assert sampler.cycles_started == 2


# ------------------------------------------------------------------ secondary attributes


def test_secondary_attributes_track_targets(cfgs):
    sampler = pretrain.build_pretrain_sampler(cfgs["pretrain"], "yor")
    n = 4000
    draws = sampler.draw_many(n)
    assert sampler.relaxed_count == 0  # compat rules never had to be dropped
    report = coverage_report(draws, targets=_pretrain_targets())
    for name in _pretrain_targets():
        dev = report["attributes"][name]["max_abs_dev_from_target"]
        assert dev <= 4, f"{name} deviates by {dev} draws from its target"
    assert report["attributes"]["locale"]["max_min_ratio"] == pytest.approx(1.0, abs=0.05)
    assert report["attributes"]["genre"]["entropy_norm"] > 0.99


def test_sft_and_dpo_attributes_are_balanced(cfgs):
    for kind, build in (("sft", sft_tasks.build_sft_sampler), ("dpo", dpo.build_dpo_sampler)):
        sampler = build(cfgs[kind], "efi")
        n = 3000
        draws = sampler.draw_many(n)
        assert sampler.relaxed_count == 0
        report = coverage_report(draws, targets={"task": sft_tasks.task_weights(), "register": t.option_weights(t.USER_REGISTERS),
                                                 "instruction_style": t.option_weights(t.INSTRUCTION_STYLES)})
        assert report["attributes"]["task"]["max_abs_dev_from_target"] <= 4
        assert report["attributes"]["register"]["max_abs_dev_from_target"] <= 4
        assert report["attributes"]["instruction_style"]["max_abs_dev_from_target"] <= 4
        assert report["attributes"]["task"]["distinct"] == len(sft_tasks.TASKS)  # every task used
        if kind == "dpo":
            assert report["attributes"]["rejection_type"]["distinct"] == len(dpo.REJECTIONS)


def test_least_used_first_respects_weights():
    spec = AttributeSpec("color", {"red": 2.0, "blue": 1.0, "off": 0.0})
    s = CoverageSampler(kind="k", language="l", seed=1, pairs=[("d", "s")], attributes=[spec])
    draws = s.draw_many(300)
    c = Counter(d["color"] for d in draws)
    assert c["off"] == 0
    assert abs(c["red"] - 200) <= 2 and abs(c["blue"] - 100) <= 2


# ------------------------------------------------------------------ determinism


def test_sampler_is_deterministic(cfgs):
    a = pretrain.build_pretrain_sampler(cfgs["pretrain"], "yor").draw_many(300)
    b = pretrain.build_pretrain_sampler(cfgs["pretrain"], "yor").draw_many(300)
    assert a == b
    other_lang = pretrain.build_pretrain_sampler(cfgs["pretrain"], "hau").draw_many(300)
    assert [x["subtopic"] for x in a] != [x["subtopic"] for x in other_lang]
    other_seed = pretrain.build_pretrain_sampler(parse_config({"kind": "pretrain", "seed": 7, "samples_per_language": {"default": 1}}), "yor").draw_many(300)
    assert a != other_seed
    assert derive_seed(1, "sft", "yor") != derive_seed(1, "sft", "hau") != derive_seed(1, "dpo", "hau")


def test_same_coverage_structure_for_every_language(cfgs):
    """Same algorithm/seed scheme => the multiset of per-pair counts is identical across languages."""
    profiles = set()
    for lang in TARGET_LANGUAGE_CODES:
        s = pretrain.build_pretrain_sampler(cfgs["pretrain"], lang)
        s.draw_many(1500)
        profiles.add(tuple(sorted(Counter(s.pair_counts.values()).items())))
        assert set(s.counts["locale"]) <= {loc.label for loc in t.LOCALES[lang]}
    assert len(profiles) == 1


# ------------------------------------------------------------------ compatibility


def test_compatibility_rules_hold_for_all_draws(cfgs):
    sampler = pretrain.build_pretrain_sampler(cfgs["pretrain"], "ibo")
    for d in sampler.draw_many(2500):
        tags = t.DOMAINS[d["domain"]].tags
        g = t.GENRES[d["genre"]]
        assert g.domain_ok(tags)
        assert d["register"] in g.registers
        assert d["perspective"] in g.perspectives
        if g.lengths:
            assert d["length_bucket"] in g.lengths
        if g.eras:
            assert d["era"] in g.eras
        assert t.AUDIENCES[d["audience"]].domain_ok(tags)
        assert t.ERAS[d["era"]].domain_ok(tags)
        assert t.DIFFICULTIES[d["difficulty"]].domain_ok(tags)


def test_sft_dpo_compatibility(cfgs):
    for d in sft_tasks.build_sft_sampler(cfgs["sft"], "yor").draw_many(2000):
        task = sft_tasks.TASKS[d["task"]]
        assert task.domain_ok(t.DOMAINS[d["domain"]].tags)
        assert d["response_length"] in task.lengths
    for d in dpo.build_dpo_sampler(cfgs["dpo"], "yor").draw_many(2000):
        task = sft_tasks.TASKS[d["task"]]
        rej = dpo.REJECTIONS[d["rejection_type"]]
        assert rej.task_ok(task.tags)
        if d["rejection_type"] == "poor_translation":
            assert "translation" in task.tags
        if d["rejection_type"] == "wrong_label":
            assert "classification" in task.tags
        if d["rejection_type"] == "ignores_constraint":
            assert d["instruction_style"] == "constraint_included"


# ------------------------------------------------------------------ weights + reporting


def test_domain_group_weights(cfgs):
    pairs = t.all_pairs()
    group_of = {k: d.group for k, d in t.DOMAINS.items()}
    weights = pair_weights_from_groups(pairs, group_of, {"health": 0.0, "culture_arts": 2.0})
    s = CoverageSampler(kind="k", language="l", seed=3, pairs=pairs, attributes=[], pair_weights=weights)
    n_arts = sum(1 for p in pairs if group_of[p[0]] == "culture_arts")
    n_health = sum(1 for p in pairs if group_of[p[0]] == "health")
    cycle_len = len(pairs) - n_health + n_arts  # health excluded, culture_arts twice per cycle
    draws = s.draw_many(cycle_len * 3)
    groups = Counter(t.DOMAINS[d["domain"]].group for d in draws)
    assert groups["health"] == 0
    assert groups["culture_arts"] == 2 * 3 * n_arts
    with pytest.raises(KeyError):
        pair_weights_from_groups(pairs, group_of, {"nonexistent_group": 1.0})


def test_merge_weights_validates():
    assert merge_weights({"a": 1, "b": 1}, {"a": 3}, name="x") == {"a": 3.0, "b": 1}
    with pytest.raises(KeyError):
        merge_weights({"a": 1}, {"zzz": 1}, name="x")
    with pytest.raises(ValueError):
        merge_weights({"a": 1}, {"a": 0}, name="x")


def test_coverage_report_shape():
    draws = [{"domain": "d1", "subtopic": "s", "genre": "g1"}, {"domain": "d1", "subtopic": "s", "genre": "g1"}, {"domain": "d2", "subtopic": "s", "genre": "g2"}]
    rep = coverage_report(draws, vocabularies={"genre": ["g1", "g2", "g3"]})
    g = rep["attributes"]["genre"]
    assert g["counts"] == {"g1": 2, "g2": 1}
    assert g["vocabulary"] == 3 and g["max_min_ratio"] is None  # g3 never used
    assert 0 < g["entropy_norm"] < 1
    assert rep["attributes"]["pair"]["max"] == 2
    assert normalized_entropy({"a": 5, "b": 5}) == pytest.approx(1.0)
    assert normalized_entropy({"a": 10, "b": 0}) == 0.0
