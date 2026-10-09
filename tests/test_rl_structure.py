import pytest

from rl import rewards as R
from rl import structure as S

EXP = "<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><zul><response>"
GOOD = EXP + "Sawubona mhlaba</s>"


def test_parse_fields():
    p = S.parse("<s>" + GOOD)
    assert (p.input_lang, p.target_lang, p.plan, p.response) == ("<eng>", "<zul>", ["<translate>"], "Sawubona mhlaba")
    assert p.ordered and p.closed and not p.leak


def test_perfect_turn_scores_one_and_opens_the_gate():
    f = S.format_score(GOOD, EXP)
    assert f["score"] == pytest.approx(1.0) and f["gate"] and f["response"] == "Sawubona mhlaba"


def test_wrong_target_language_closes_the_gate_and_costs_its_weight():
    f = S.format_score(GOOD.replace("<zul>", "<xho>"), EXP)
    assert not f["gate"] and f["target_lang"] == 0 and f["score"] == pytest.approx(0.7)


def test_plan_is_order_agnostic_and_partial_credit_is_recall_weighted():
    """`plan` is a SET score, and the set score weights recall 4x (F-beta, beta=2), not evenly (F1).

    Missing a task the reference declares is the expensive mistake -- the model simply did not do part of the
    work -- while declaring an extra verb is noise. Under F1 those two were worth the same; they are not.
    """
    exp = "<|input_lang|><swh><task_plan><classify><intent></task_plan><|target_lang|><eng><response>"
    swapped = "<|input_lang|><swh><task_plan><intent><classify></task_plan><|target_lang|><eng><response>book_flight</s>"
    assert S.format_score(swapped, exp)["plan"] == 1.0
    beta_sq = S.PLAN_BETA ** 2
    partial = "<|input_lang|><swh><task_plan><classify></task_plan><|target_lang|><eng><response>x</s>"
    # precision 1.0, recall 0.5 -- a MISSED task
    assert S.format_score(partial, exp)["plan"] == pytest.approx(
        (1 + beta_sq) * 1.0 * 0.5 / (beta_sq * 1.0 + 0.5))
    extra = "<|input_lang|><swh><task_plan><classify><intent><toxic></task_plan><|target_lang|><eng><response>x</s>"
    # precision 2/3, recall 1.0 -- a SPURIOUS task, and it must cost less than the miss above
    assert S.format_score(extra, exp)["plan"] == pytest.approx(
        (1 + beta_sq) * (2 / 3) * 1.0 / (beta_sq * (2 / 3) + 1.0))
    assert S.format_score(extra, exp)["plan"] > S.format_score(partial, exp)["plan"]


def test_wrong_order_keeps_a_perfect_plan_but_costs_the_total():
    """Order is scored, at a third of the weight of the set: a swapped plan is right, and not quite as right."""
    exp = "<|input_lang|><swh><task_plan><|is_technical|><|RAG|><qa></task_plan><|target_lang|><swh><response>x</s>"
    swapped = exp.replace("<|is_technical|><|RAG|><qa>", "<qa><|RAG|><|is_technical|>")
    right, wrong = S.format_score(exp, exp), S.format_score(swapped, exp)
    assert right["plan"] == wrong["plan"] == 1.0          # same set
    assert wrong["plan_order"] < right["plan_order"]      # reversed ranking
    assert wrong["score"] < right["score"]                # and that now costs something
    assert "plan_order" in wrong["scored_terms"]
    # but it costs far less than getting the set wrong
    missed = exp.replace("<|is_technical|><|RAG|><qa>", "<qa>")
    assert wrong["score"] > S.format_score(missed, exp)["score"]


def test_order_is_undefined_and_unweighted_for_a_single_verb_plan():
    """A one-verb plan has no order to get wrong, so it is not scored on one."""
    exp = "<|input_lang|><swh><task_plan><summarize></task_plan><|target_lang|><swh><response>x</s>"
    f = S.format_score(exp, exp)
    assert f["plan_order"] is None
    assert "plan_order" not in f["scored_terms"]
    assert f["score"] == pytest.approx(1.0)


def test_missing_eos_or_leaked_scaffold_breaks_structure_only():
    f = S.format_score(EXP + "Sawubona", EXP)
    assert f["structure"] == pytest.approx(0.8) and f["gate"] and f["score"] == pytest.approx(0.96)
    leaked = S.format_score(EXP + "Sawubona <|target_lang|><eng> again</s>", EXP)
    assert leaked["structure"] < 0.8 and not leaked["gate"]


def test_order_of_the_landmarks_matters_but_order_inside_the_plan_does_not():
    wrong = "<|target_lang|><zul><|input_lang|><eng><task_plan><translate></task_plan><response>Sawubona</s>"
    f = S.format_score(wrong, EXP)
    assert f["input_lang"] == 1 and f["target_lang"] == 1 and f["plan"] == 1
    assert 0 < f["structure"] < S.format_score(GOOD, EXP)["structure"] and f["score"] < 1.0
    missing_close = "<|input_lang|><eng><task_plan><translate><|target_lang|><zul><response>Sawubona</s>"
    assert S.parse(missing_close).order < 1.0


def test_shape_reward_is_monotone_negative_below_threshold_and_positive_above():
    xs = [i / 20 for i in range(21)]
    ys = [S.shape_reward(x) for x in xs]
    assert all(b > a for a, b in zip(ys, ys[1:]))
    assert S.shape_reward(0.0) == pytest.approx(-2.0) and S.shape_reward(0.46) == pytest.approx(0.0) and S.shape_reward(1.0) == pytest.approx(2.0)
    assert S.shape_reward(0.45) < 0 < S.shape_reward(0.5)


def test_tagging_content_may_contain_tag_markers():
    exp = "<|input_lang|><afr><task_plan><POS><tag></task_plan><|target_lang|><afr><response>"
    out = exp + "Die <tag> DET doel <tag> NOUN</s>"
    assert S.format_score(out, exp)["structure"] == 1.0


class FixedScorer:
    def __init__(self, v): self.v, self.calls = v, []
    def __call__(self, items, completions):
        self.calls.append(list(completions)); return [self.v] * len(items)


def _item(**kw):
    base = {"source": "Hello world, how are you today", "expected_structure": EXP, "reference": "Sawubona mhlaba unjani namhlanje"}
    base.update(kw); return base


def test_reward_is_reference_free_by_default():
    ref, qe = FixedScorer(0.8), FixedScorer(0.5)
    rw = R.StructuredTranslationReward(ref, qe, format_weight=0.2)
    r = rw([_item(), _item(reference="")], [EXP + "Sawubona mhlaba unjani namhlanje</s>"] * 2)
    assert r[0] == pytest.approx(0.2 + 0.8 * 0.5) and r[1] == pytest.approx(0.2 + 0.8 * 0.5)
    assert ref.calls == [] and len(qe.calls[0]) == 2          # the reference is ignored; QE scored both rows


def test_use_reference_switches_rows_with_a_reference_to_the_reference_model():
    ref, qe = FixedScorer(0.8), FixedScorer(0.5)
    rw = R.StructuredTranslationReward(ref, qe, format_weight=0.3, chrf_anchor=0.0, use_reference=True)
    r = rw([_item(), _item(reference="")], [EXP + "Sawubona mhlaba unjani namhlanje</s>"] * 2)
    assert r[0] == pytest.approx(0.3 + 0.7 * 0.8) and r[1] == pytest.approx(0.3 + 0.7 * 0.5)


def test_shaped_reward_is_large_negative_when_bad_and_positive_when_good():
    rw = R.StructuredTranslationReward(None, FixedScorer(0.9), format_weight=0.2, shaping="threshold", threshold=0.46, negative_scale=2.0)
    good = rw([_item()], [GOOD])[0]
    wrong_lang = rw([_item()], [GOOD.replace("<zul>", "<xho>")])[0]
    assert good > 1.0 and wrong_lang < -1.0


def test_content_reward_is_withheld_when_the_target_tag_is_wrong():
    ref = FixedScorer(0.9)
    rw = R.StructuredTranslationReward(None, ref, format_weight=0.3)
    r = rw([_item()], [GOOD.replace("<zul>", "<xho>")])
    assert r[0] == pytest.approx(0.3 * 0.7) and ref.calls == []


def test_copying_the_source_scores_no_content_and_length_guard_halves():
    rw = R.StructuredTranslationReward(None, FixedScorer(0.9), format_weight=0.0)
    copy = rw([_item()], [EXP + "Hello world, how are you today</s>"])
    assert copy[0] == 0.0
    tiny = rw([_item()], [EXP + "Sa</s>"])
    assert tiny[0] == pytest.approx(0.45)


def test_wants_raw_and_stats_for_the_logger():
    rw = R.StructuredTranslationReward(None, FixedScorer(0.5))
    assert rw.wants_raw
    rw([_item()], [GOOD])
    assert rw.last_stats["gate_rate"] == 1.0 and rw.last_stats["fmt_plan_f1"] == 1.0


# ------------------------------------------------------------------------- rejection sampling
from rl import rejection_sampling as RS  # noqa: E402


def test_best_of_n_keeps_the_best_that_clear_the_bar():
    rewards = [0.2, 0.9, 0.6, 0.1,   0.3, 0.4, 0.5, 0.2]          # two prompts, n=4
    assert RS.select_best_of_n(rewards, 4, 1, 0.55) == [[1], []]
    assert RS.select_best_of_n(rewards, 4, 2, 0.55) == [[1, 2], []]


def test_filter_ranks_inside_each_language_pair_and_applies_the_floor():
    rows = [{"source_lang": "eng", "targ_lang": "zul", "input": f"s{i}", "response": f"t{i}"} for i in range(4)] + \
           [{"source_lang": "eng", "targ_lang": "wol", "input": f"w{i}", "response": f"v{i}"} for i in range(4)]
    scores = [0.9, 0.8, 0.2, 0.1,   0.3, 0.25, 0.2, 0.1]          # Wolof scores are low across the board
    kept = RS.filter_by_score(rows, scores, keep_frac=0.5, min_score=0.0)
    assert sorted(r["input"] for r in kept) == ["s0", "s1", "w0", "w1"]       # wolof is not wiped out
    floor = RS.filter_by_score(rows, scores, keep_frac=0.5, min_score=0.5)
    assert sorted(r["input"] for r in floor) == ["s0", "s1"] and all("qe_score" in r for r in floor)


def test_run_filter_end_to_end_with_a_fake_scorer(tmp_path):
    from rl.config import RLConfig

    cfg = RLConfig(out_dir=str(tmp_path), rs_filter_keep_frac=0.5)
    rows = [{"source_lang": "eng", "targ_lang": "zul", "input": f"s{i}", "response": f"t{i}",
             "output_structure": "<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><zul><response>"} for i in range(6)]
    rows.append({"source_lang": "swh", "targ_lang": "eng", "input": "x", "response": "y", "output_structure": "<task_plan><classify>"})
    out = RS.run_filter(cfg, scorer=lambda items, comps: [int(c[1:]) / 10 for c in comps], rows=rows)
    kept = [__import__("json").loads(l) for l in open(out)]
    assert [r["input"] for r in kept] == ["s3", "s4", "s5"]            # the classify row never reached the scorer


# ------------------------------------------------------------------------- calibrated / adaptive shaping
from rl import shaping as SH  # noqa: E402


def test_curve_is_monotone_with_four_regions_and_resolves_good_vs_great():
    a = (0.46, 0.6, 0.8)
    xs = [i / 100 for i in range(101)]
    ys = [SH.shape(x, a) for x in xs]
    assert all(b >= a_ for a_, b in zip(ys, ys[1:]))
    assert SH.shape(0.0, a) == pytest.approx(-2.0) and SH.shape(0.46, a) == pytest.approx(0.0)
    assert SH.shape(0.6, a) == pytest.approx(1.0) and SH.shape(0.8, a) == pytest.approx(2.0) and SH.shape(1.0, a) == pytest.approx(2.25)
    assert SH.shape(0.7, a) - SH.shape(0.6, a) > 0.4            # good -> great keeps real slope
    assert SH.shape(0.95, a) - SH.shape(0.8, a) < 0.2           # beyond human-level quality earns almost nothing
    assert SH.shape(0.45, a) < 0 < SH.shape(0.47, a)            # the 0.46 line from hand inspection
    assert SH.shape(0.3, SH.DEFAULT_ANCHORS) < -0.5             # gibberish is penalised hard
    assert SH.shape(0.55, (0.3, 0.5, 0.7)) == SH.shape(0.55, (0.46, 0.5, 0.7))   # t_fail never below the floor


def test_adaptive_raises_the_bar_as_the_policy_improves_but_keeps_the_floor():
    sh = SH.ContentShaper(mode="adaptive", calibration={"eng>zul": (0.5, 0.6, 0.7)}, window=500, warmup=50, ema=0.0)
    before = sh(0.6, "eng>zul")
    sh.observe("eng>zul", [0.8 + 0.001 * i for i in range(100)])      # the policy now scores ~0.8-0.9
    after = sh(0.6, "eng>zul")
    assert after < before                                            # 0.6 is no longer 'good' for this policy
    assert sh.anchors("eng>zul")[0] == 0.5                           # t_fail stays anchored to the human reference


def test_auc_and_anchors():
    assert SH.auc([0.9, 0.8, 0.7], [0.1, 0.2, 0.3]) == 1.0 and SH.auc([0.1], [0.9]) == 0.0
    assert SH.compute_anchors(list(range(101)))[1] == pytest.approx(50.0)


def test_calibrate_flags_pairs_where_the_metric_cannot_tell_real_from_mismatched():
    from importlib import util
    spec = util.spec_from_file_location("cal", "rlhf/scripts/calibrate_shaping.py")
    cal = util.module_from_spec(spec); spec.loader.exec_module(cal)
    rows = [{"source_lang": "eng", "targ_lang": "zul", "input": f"s{i}", "response": f"t{i}"} for i in range(20)] + \
           [{"source_lang": "yor", "targ_lang": "hau", "input": f"s{i}", "response": f"t{i}"} for i in range(20)]
    # a scorer that recognises the aligned reference (same index) for eng>zul but is blind (constant) for yor>hau
    def scorer(items, comps):
        out = []
        for it, c in zip(items, comps):
            blind = it["source"].startswith("s") and False
            out.append(0.9 if it["source"][1:] == c[1:] else 0.2)
        return out
    res = cal.calibrate(rows, scorer, max_per_pair=20)
    assert res["eng>zul"]["usable"] and res["eng>zul"]["auc_vs_mismatched"] == 1.0
    blind = cal.calibrate(rows[20:], lambda items, comps: [0.5] * len(items), max_per_pair=20)
    assert not blind["yor>hau"]["usable"]


def test_user_chosen_default_anchors_can_be_swapped_in():
    sh = SH.ContentShaper(mode="calibrated", default_anchors=(0.4, 0.6, 0.8))
    assert sh.anchors("any>pair") == (0.46, 0.6, 0.8)      # t_fail still floored at 0.46
    sh2 = SH.ContentShaper(mode="calibrated", calibration={"eng>zul": (0.5, 0.65, 0.78)}, default_anchors=(0.4, 0.6, 0.8))
    assert sh2.anchors("eng>zul") == (0.5, 0.65, 0.78)


def test_pair_anchors_put_the_failure_line_above_fluent_nonsense_and_respect_the_floor():
    human = [0.5 + 0.01 * i for i in range(40)]            # 0.50 .. 0.89, median ~0.70
    nonsense = [0.40 + 0.002 * i for i in range(40)]       # 0.40 .. 0.48, p90 ~0.47
    t_fail, t_mid, t_great = SH.pair_anchors(human, nonsense)
    assert t_fail == pytest.approx(0.47, abs=0.01) and t_fail > 0.46
    assert t_mid == pytest.approx(0.70, abs=0.02) and t_great > t_mid
    low = SH.pair_anchors(human, [0.2] * 10)
    assert low[0] == 0.46                                  # never below the hand-checked floor
    assert SH.reliability_from_auc(0.95) == 1.0 and SH.reliability_from_auc(0.65) == 0.0
    assert 0 < SH.reliability_from_auc(0.79) < 1


def test_unusable_pairs_get_scaffold_only_reward_and_zero_advantage_weight(tmp_path):
    cal = {"eng>zul": {"anchors": [0.5, 0.65, 0.8], "usable": True, "reliability": 0.5},
           "tsn>sot": {"anchors": [0.5, 0.6, 0.7], "usable": False, "reliability": 0.0}}
    path = tmp_path / "c.json"; path.write_text(__import__("json").dumps(cal))
    sh = SH.ContentShaper.from_file(str(path), mode="calibrated")
    assert sh.is_usable("eng>zul") and not sh.is_usable("tsn>sot") and sh.reliability("eng>zul") == 0.5 and sh.reliability("tsn>sot") == 0.0
    rw = R.StructuredTranslationReward(None, FixedScorer(0.9), format_weight=0.2, shaping="calibrated", shaper=sh)
    exp_ok = "<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><zul><response>"
    exp_bad = "<|input_lang|><tsn><task_plan><translate></task_plan><|target_lang|><sot><response>"
    items = [{"source": "hello world friend", "expected_structure": exp_ok, "source_lang": "eng", "targ_lang": "zul"},
             {"source": "dumela lefatshe mogwera", "expected_structure": exp_bad, "source_lang": "tsn", "targ_lang": "sot"}]
    r = rw(items, [exp_ok + "Sawubona mhlaba mngani</s>", exp_bad + "Dumela lefatshe mogwera wa me</s>"])
    assert rw.last_weights == [0.5, 0.0]
    assert r[1] == pytest.approx(2.0)                       # scaffold-only, perfect scaffold on the -2..+2 scale
    assert r[0] > 1.0                                       # content 0.9 is above t_great: near the top of the scale


def test_missing_calibration_file_falls_back_to_defaults(tmp_path, capsys):
    sh = SH.ContentShaper.from_file(str(tmp_path / "nope.json"), mode="calibrated")
    assert sh.anchors("eng>zul") == SH.DEFAULT_ANCHORS and "not found" in capsys.readouterr().err


def test_input_lang_lists_every_language_of_the_request_in_reading_order():
    from rl.structure import expected_translation_structure as E, format_score, input_lang_tags as T, parse
    assert T("efi", "ibo", "prepend") == "<ibo><efi>" and T("efi", "ibo", "append") == "<efi><ibo>" and T("efi", "efi", "prepend") == "<efi>" and T("eng") == "<eng>"
    exp = E("efi", "eng", "ibo", "prepend")
    assert exp == "<|input_lang|><ibo><efi><task_plan><translate></task_plan><|target_lang|><eng><response>"
    assert parse(exp + "Hello</s>").input_lang == "<ibo><efi>"
    assert format_score(exp + "Hello</s>", exp)["input_lang"] == 1.0
    old = "<|input_lang|><efi><task_plan><translate></task_plan><|target_lang|><eng><response>Hello</s>"        # the v4 mistake: the instruction language is missing
    assert format_score(old, exp)["input_lang"] == 0.25
    swapped = "<|input_lang|><efi><ibo><task_plan><translate></task_plan><|target_lang|><eng><response>Hello</s>"
    assert format_score(swapped, exp)["input_lang"] == 0.5
    assert parse("<|input_lang|> unsure, but most likely <ibo><efi><task_plan><translate></task_plan>").input_lang == "<ibo><efi>"


def test_dataset_side_helper_agrees_with_the_reward_side():
    import sys
    sys.path.insert(0, "data-gen")
    from africomet_translation.instructions import input_lang_tags as D
    from rl.structure import input_lang_tags as R
    for src in ("efi", "eng"):
        for ins in ("ibo", "eng", None):
            for pos in ("prepend", "append"):
                assert D(src, ins, pos) == R(src, ins, pos)
