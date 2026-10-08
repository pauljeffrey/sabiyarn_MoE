import json
import random

from sft_new import build as B


def test_assistant_turn_carries_the_translate_plan_between_the_language_tags():
    row = B.make_row("eng", "zul", "Good morning, how are you?", "Sawubona, unjani?", "x", 8.0, random.Random(0))
    msgs = json.loads(row["messages"])
    assert msgs[1]["content"] == ("<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><zul><response>"
                                  "Sawubona, unjani?")
    assert row["input"] in msgs[0]["content"] and row["instruction"] in msgs[0]["content"]
    assert [row[k] for k in ("source_lang", "targ_lang")] == ["eng", "zul"]


def test_pair_filter_rejects_short_and_badly_proportioned_pairs():
    assert not B.pair_ok("Hi.", "Sawubona.")
    assert not B.pair_ok("This is a fairly ordinary English sentence.", "Ok")
    assert B.pair_ok("This is a fairly ordinary English sentence.", "Lona yisigwebo esijwayelekile sesiNgisi kakhulu.")
