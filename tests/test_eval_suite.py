import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_suite import data, metrics, prompts, tasks  # noqa: E402
from eval_suite.langs import LANGS  # noqa: E402


# ------------------------------------------------------------------ metrics
def test_chrf_and_bleu_are_100_for_identical_text_and_low_for_unrelated():
    pytest.importorskip("sacrebleu")
    refs = ["Ilé ìwé náà wà nítòsí ọjà.", "Mo fẹ́ jẹ ìrẹsì."]
    perfect = metrics.chrf_bleu(refs, refs)
    assert perfect["chrf++"] == pytest.approx(100.0) and perfect["bleu"] == pytest.approx(100.0)
    assert metrics.chrf_bleu(["xyz qrs", "lmn"], refs)["chrf++"] < 20


def test_classification_metrics_macro_f1_accuracy_and_majority_baseline():
    gold = ["a", "a", "a", "b"]
    pred = ["a", "a", "b", "b"]
    m = metrics.classification_metrics(gold, pred, ["a", "b", "c"])
    assert m["accuracy"] == 0.75 and m["majority_baseline"] == 0.75 and m["n"] == 4
    # a: p=1, r=2/3 -> f1 .8 ; b: p=.5, r=1 -> f1 2/3 ; c: 0  -> macro over 3 labels
    assert m["macro_f1"] == pytest.approx((0.8 + 2 / 3 + 0) / 3)
    assert m["acc_ci95"][0] < 0.75 < m["acc_ci95"][1]


def test_bio_entities_handles_adjacent_entities_and_stray_inside_tags():
    assert metrics.bio_entities(["B-PER", "I-PER", "O", "B-LOC"]) == {("PER", 0, 2), ("LOC", 3, 4)}
    assert metrics.bio_entities(["B-PER", "B-PER"]) == {("PER", 0, 1), ("PER", 1, 2)}
    assert metrics.bio_entities(["O", "I-ORG", "I-ORG"]) == {("ORG", 1, 3)}  # stray I- starts a span
    assert metrics.bio_entities(["B-PER", "I-LOC"]) == {("PER", 0, 1), ("LOC", 1, 2)}


def test_entity_f1_needs_exact_type_and_boundaries_unlike_label_presence():
    gold = [["B-PER", "I-PER", "O", "B-LOC"]]
    perfect = metrics.entity_f1(gold, gold)
    assert perfect["f1"] == 1.0 and perfect["token_accuracy"] == 1.0
    wrong_boundary = metrics.entity_f1(gold, [["B-PER", "O", "O", "B-LOC"]])
    assert wrong_boundary["f1"] == pytest.approx(0.5)  # LOC right, PER span wrong: p=r=1/2
    # the legacy per-sentence label-presence metric would have called this a perfect match
    assert set(gold[0]) - {"I-PER"} <= {"B-PER", "O", "B-LOC"}


def test_parse_tags_pads_truncates_and_replaces_invalid_tags():
    valid = {"O", "B-PER", "I-PER"}
    assert tasks.parse_tags("B-PER I-PER junk", 3, valid) == ["B-PER", "I-PER", "O"]
    assert tasks.parse_tags("B-PER", 3, valid) == ["B-PER", "O", "O"]
    assert tasks.parse_tags("B-PER I-PER O O O", 2, valid) == ["B-PER", "I-PER"]


# ------------------------------------------------------------------ data helpers
def test_mc_row_parses_afrimmlu_string_choices_and_letter_answers():
    r = data._mc_row({"subject": "x", "question": "q", "choices": "['a', 'b', 'c', 'd']", "answer": "C"})
    assert r["choices"] == ["a", "b", "c", "d"] and r["answer"] == 2
    assert data._mc_row({"subject": "x", "question": "q", "choices": ["a", "b", "c", "d"], "answer": 3})["answer"] == 3


def test_subsample_is_deterministic_ordered_and_a_noop_below_the_limit():
    items = list(range(100))
    a, b = data.subsample(items, 10), data.subsample(items, 10)
    assert a == b and a == sorted(a) and len(a) == 10
    assert data.subsample(items, 500) == items and data.subsample(items, None) == items


def test_script_datasets_fall_back_to_the_parquet_conversion(monkeypatch):
    import datasets

    calls = []

    def fake_load(path, name=None, split=None, data_files=None, token=None):
        calls.append((path, name, data_files))
        if path != "parquet":
            raise RuntimeError("Dataset scripts are no longer supported, but found x.py")
        return "PARQUET"

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    assert data._load("masakhane/masakhaner2", "yor", "test") == "PARQUET"
    assert calls[-1][2] == {"test": "hf://datasets/masakhane/masakhaner2@refs%2Fconvert%2Fparquet/yor/test/0000.parquet"}


def test_language_registry_marks_missing_benchmarks():
    assert LANGS["efi"].flores is None and LANGS["urh"].sib is None  # no standard benchmark
    assert LANGS["yor"].ner == "yor" and LANGS["pcm"].flores is None and LANGS["fon"].mafand is None
    assert LANGS["yor"].tag == "<yor>"


# ------------------------------------------------------------------ prompts
def test_tag_prompts_match_the_pretraining_format():
    yor, eng = LANGS["yor"], LANGS["eng"]
    assert prompts.translation("tag", None, eng, yor, "I love rice") == "<translate> I love rice <yor>"
    assert prompts.classification("tag", None, "topic", "T", ["a"]) == ("<classify> T <topic> :", " ")
    assert prompts.classification("tag", None, "sentiment", "T", ["a"])[0] == "<classify> T <sentiment> :"
    assert prompts.ner(["Ade", "went"]) == "<NER> Ade went <tag> :"


def test_mmlu_prompt_layout():
    shot = {"question": "1+1?", "choices": ["1", "2", "3", "4"], "answer": 1}
    p = prompts.mmlu("elementary_mathematics", "2+2?", ["1", "2", "3", "4"], [shot])
    assert p.startswith("The following are multiple choice questions (with answers) about elementary mathematics.")
    assert "1+1?\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer: B\n\n2+2?" in p and p.endswith("Answer:")


# ------------------------------------------------------------------ tasks, with a scripted runner
class FakeRunner:
    """Implements the ModelRunner surface. `oracle` maps prompt text -> desired output."""

    class tok:  # noqa: N801
        @staticmethod
        def apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
            return "<s>" + "".join(m["content"] for m in messages)

    def __init__(self, answers=None, pick=None):
        self.answers, self.pick = answers or {}, pick

    def encode(self, text, style):
        return text  # contexts stay strings; the fake generate/score understand them

    def encode_continuation(self, text):
        return text

    def generate(self, contexts, max_new_tokens=128, num_beams=1, repetition_penalty=1.0):
        return [self.answers.get(c, "") for c in contexts]

    def score(self, context, continuations):
        want = self.pick(context) if self.pick else None
        return [1.0 if c.strip() == want else 0.0 for c in continuations]


def test_translation_task_scores_both_directions(monkeypatch):
    pytest.importorskip("sacrebleu")
    pairs = [{"eng": "I love rice", "xx": "Mo fẹ́ràn ìrẹsì"}, {"eng": "Good morning", "xx": "Ẹ kú àárọ̀"}]
    monkeypatch.setattr(data, "load_translation", lambda *a, **k: (pairs, "stub"))
    answers = {}
    for p in pairs:
        answers[f"<translate> {p['eng']} <yor>"] = p["xx"]  # eng->yor perfect
        answers[f"<translate> {p['xx']} <eng>"] = "garbage"  # yor->eng wrong
    m, rows = tasks.run_translation(FakeRunner(answers), LANGS["yor"], "tag", 2, "auto", None, 64, 1, 1.0)
    assert m["eng->yor"]["chrf++"] == pytest.approx(100.0) and m["yor->eng"]["chrf++"] < 20
    assert m["source"] == "stub" and len(rows) == 4


def test_topic_task_ranks_labels_by_likelihood(monkeypatch):
    items = [{"text": "goal scored", "label": "sports"}, {"text": "new vaccine", "label": "health"}]
    monkeypatch.setattr(data, "load_topic", lambda *a, **k: (items, ["sports", "health", "politics"]))
    truth = {"<classify> goal scored <topic> :": "sports", "<classify> new vaccine <topic> :": "politics"}
    m, rows = tasks.run_classification(FakeRunner(pick=truth.get), "topic", LANGS["yor"], "tag", None, "sib200", None)
    assert m["accuracy"] == 0.5 and rows[1]["pred"] == "politics"


def test_ner_task_uses_entity_level_f1(monkeypatch):
    items = [{"tokens": ["Ade", "Bello", "in", "Lagos"], "tags": ["B-PER", "I-PER", "O", "B-LOC"]}]
    monkeypatch.setattr(data, "load_ner", lambda *a, **k: (items, data.NER_TAGS))
    ctx = "<NER> Ade Bello in Lagos <tag> :"
    good, _ = tasks.run_ner(FakeRunner({ctx: "B-PER I-PER O B-LOC"}), LANGS["yor"], "tag", None, None, 64, 1, 1.0)
    bad, _ = tasks.run_ner(FakeRunner({ctx: "B-PER O O B-LOC"}), LANGS["yor"], "tag", None, None, 64, 1, 1.0)
    assert good["f1"] == 1.0 and bad["f1"] == pytest.approx(0.5)
    with pytest.raises(data.DataUnavailable):
        tasks.run_ner(FakeRunner(), LANGS["yor"], "chat", None, None, 64, 1, 1.0)


def test_mmlu_task_accuracy_and_confidence_interval(monkeypatch):
    test = [{"subject": "s", "question": f"q{i}", "choices": list("wxyz"), "answer": i % 4} for i in range(8)]
    monkeypatch.setattr(data, "load_mmlu", lambda *a, **k: (test, {}, "stub"))
    # oracle answers the first four correctly, the rest wrongly
    def pick(ctx):
        i = int(ctx.rsplit("q", 1)[1].split("\n")[0])
        return "ABCD"[i % 4] if i < 4 else "ABCD"[(i + 1) % 4]

    m, _ = tasks.run_mmlu(FakeRunner(pick=pick), LANGS["yor"], "tag", None, None, 0)
    assert m["accuracy"] == 0.5 and m["n"] == 8 and m["chance"] == 0.25 and m["acc_ci95"][0] < 0.5 < m["acc_ci95"][1]


# ------------------------------------------------------------------ the real ModelRunner on a tiny random model
def _tiny_runner(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from sabiyarn.model.configuration import GPTJXMoEConfig
    from sabiyarn.model.modeling import GPTJXMoEForCausalLM
    from eval_suite.model import ModelRunner

    torch.manual_seed(0)
    cfg = GPTJXMoEConfig(block_size=256, vocab_size=52050, n_layer=2, n_heads=2, n_embd=32, use_moe=True,
                         num_experts=4, num_experts_per_tok=2, moe_dim=64, expert_per_layer={"0": 2, "1": 3})
    GPTJXMoEForCausalLM(cfg).save_pretrained(tmp_path)
    try:
        return ModelRunner(str(tmp_path), batch_size=4, device="cpu")
    except Exception as exc:  # offline: the tokenizer lives on the Hub
        pytest.skip(f"tokenizer unavailable: {exc}")


def test_score_is_invariant_to_batching_and_right_padding(tmp_path):
    import torch

    r = _tiny_runner(tmp_path)
    ctx = r.encode("<classify> Ilé ìwé <topic> :", "tag")
    short, long_ = r.encode_continuation(" sports"), r.encode_continuation(" science and technology news")
    alone = [r.score(ctx, [c])[0] for c in (short, long_)]
    together = r.score(ctx, [short, long_])
    assert together == pytest.approx(alone, rel=1e-4, abs=1e-4)
    # manual reference for the short continuation
    ids = torch.tensor([ctx + short])
    lp = torch.log_softmax(r.model(input_ids=ids).logits.float(), -1)[0]
    manual = sum(lp[len(ctx) - 1 + i, t].item() for i, t in enumerate(short))
    assert alone[0] == pytest.approx(manual, rel=1e-4, abs=1e-4)


def test_generate_batches_equal_length_prompts_and_matches_single_prompt_decoding(tmp_path):
    r = _tiny_runner(tmp_path)
    ctxs = [r.encode(t, "tag") for t in ("<translate> a b <yor>", "<translate> c d <yor>", "<translate> a b <yor>")]
    assert len({len(c) for c in ctxs}) == 1
    batched = r.generate(ctxs, max_new_tokens=6)
    single = [r.generate([c], max_new_tokens=6)[0] for c in ctxs]
    assert batched == single and batched[0] == batched[2]
