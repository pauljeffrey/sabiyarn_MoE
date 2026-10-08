"""Offline tests for the combined-SFT converters (no network, no language-ID model)."""

import json
import random

import pytest

from sft_new import old_format as O
from sft_new import tagged_parse as TP
from sft_new import templates as T
from sft_new import textutil as U
from sft_new.compose import Rec, Skip, finish
from sft_new.grammar import normalize_turn


def test_unsure_language_is_written_and_survives_normalisation():
    rec = Rec("translation", "t", "yor", "eng", "Hello", input="Bawo", instruction="Translate:", plan=["<translate>"], unsure_src=True)
    row = finish(rec, random.Random(0))
    a = json.loads(row["messages"])[-1]["content"]
    assert a.startswith("<|input_lang|> unsure, but most likely <yor><task_plan><translate></task_plan><|target_lang|><eng><response>Hello")
    assert normalize_turn(a) == a
    assert row["output_structure"].endswith("<response>")


def test_label_token_moves_into_the_plan():
    rec = Rec("sentiment", "t", "ibo", "eng", "Positive", input="ọma", instruction="Sentiment?", plan=["<classify>"], label="<sentiment>")
    a = json.loads(finish(rec, random.Random(0))["messages"])[-1]["content"]
    assert "<task_plan><classify><sentiment></task_plan>" in a and a.endswith("<response>Positive")


def test_tag_stream_keeps_tag_markers():
    rec = Rec("pos", "t", "hau", "hau", "Hakan <tag> ADP na <tag> AUX", prompt_override="POS: Hakan na", plan=["<POS>"], sequence_labels=True)
    a = json.loads(finish(rec, random.Random(0))["messages"])[-1]["content"]
    assert "<task_plan><POS><tag></task_plan>" in a


def test_empty_response_is_skipped():
    with pytest.raises(Skip):
        finish(Rec("x", "t", "eng", "eng", "  ", input="hi", plan=["<|chat|>"]), random.Random(0))


def test_ids_are_content_hashes():
    a = finish(Rec("x", "t", "eng", "eng", "ans", input="hi", plan=["<|chat|>"]), random.Random(0))
    b = finish(Rec("x", "other", "eng", "eng", "ans", input="hi", plan=["<|chat|>"]), random.Random(1))
    assert a["id"] == b["id"]                                    # same content from two sources collapses on dedupe
    # a templated pair found twice gets two random instructions but is still ONE example
    t1 = finish(Rec("translation", "s1", "yor", "eng", "Hello", input="Bawo", instruction="Translate:", plan=["<translate>"]), random.Random(0))
    t2 = finish(Rec("translation", "s2", "yor", "eng", "Hello", input="Bawo", instruction="Fassara wannan:", plan=["<translate>"]), random.Random(0))
    assert t1["id"] == t2["id"]
    # prompt-only rows (old repos) with the same label but DIFFERENT texts must never collapse (this once dropped 99.9% of BeardedMonster/hate)
    h1 = finish(Rec("hate", "h", "hau", "eng", "Normal", prompt_override="text one", plan=["<classify>"], label="<|hate|>"), random.Random(0))
    h2 = finish(Rec("hate", "h", "hau", "eng", "Normal", prompt_override="text two", plan=["<classify>"], label="<|hate|>"), random.Random(0))
    assert h1["id"] != h2["id"]


OLD = "<|input_lang|><hau><ewe><task_plan><intent></task_plan><|target_lang|><eng><response>alarm"


def test_old_turn_parse_and_plan_mapping():
    p = O.parse_old(OLD)
    assert p["inl"] == "<hau><ewe>" and p["resp"] == "alarm"
    assert O.lang_codes(p["inl"]) == ["hau", "ewe"]
    assert O.plan_of(p["plan"]) == (["<classify>"], "<intent>")      # a label with no verb means classification
    assert O.plan_of("<pos><tag>") == (["<POS>"], None)
    assert O.plan_of(" <|generate|> ")[0] == ["<|generate|>"]        # padded plans are tolerated


def test_old_language_names_and_unknowns():
    assert O.lang_codes("Hausa<ewe>") == ["hau", "ewe"]
    assert O.lang_codes("<knc><yor>") == [None, "yor"]               # Kanuri is not ours
    assert O.lang_codes("<swa>") == ["swh"]


def test_word_less_tags_are_aligned_to_words():
    instr, text, words = O.split_prompt_text("Provide the tag for each token: A b c", 3, None)
    assert words == ["A", "b", "c"] and instr.startswith("Provide")
    _, text, words = O.split_prompt_text("Ti ŋkɔ le: <NER> A saurari Ibrahim Gblɔ nyawo le Hausa me", 3, "<NER>")
    assert words == ["A", "saurari", "Ibrahim"]
    with pytest.raises(Skip):
        O.split_prompt_text("too short", 9, None)


def test_tagged_split_requires_two_tags():
    v, s, a, b = TP.split("<prompt> Translate the following sentence to Zulu (zul). Return the translated sentence only.\nLayout  <response>: Ukubonakala ngaphandle")
    assert (v, s) == ("prompt", "response") and b == "Ukubonakala ngaphandle"
    m = TP.TRANSLATE_PROMPT.match(a)
    assert m.group("code") == "zul" and m.group("src") == "Layout"
    v, s, a, b = TP.split("<translate> Ị ma na ndị uweojii? <eng>\nYou know that the police")
    assert (v, s, a, b) == ("translate", "eng", "Ị ma na ndị uweojii?", "You know that the police")
    v, s, a, b = TP.split("<classify> Anotanga kaita denderedzwa <topic>: Weather")
    assert (v, s, b) == ("classify", "topic", "Weather")
    with pytest.raises(Skip):
        TP.split("<translate> only one tag here")
    with pytest.raises(Skip):
        TP.split("no tags at all")


def test_degenerate_and_alignment_filters():
    assert U.has_loop("a b (a.s) (a.s) (a.s) (a.s) (a.s) c")
    assert not U.has_loop("the quick brown fox jumps over the lazy dog")
    assert U.degenerate("dal " * 40)
    assert U.aligned("Czechoslovakia 1948 Soviet Union Prague 1990", "Czechoslovakia 1948 Soviet Union Prague 1990 ni")
    assert not U.aligned("Ayatullah Khamene'i said 1948 Tehran Iran", "Reynolds Jefferson Everson 1879 1947 Hugo Black")
    assert U.aligned("a short text", "ọ̀rọ̀ kúkúrú")                    # too few names/numbers to judge


def test_calculator_annotations_are_removed():
    assert U.strip_calc("sold 48/2 = <<48/2=24>>24 clips") == "sold 48/2 = 24 clips"
    assert U.strip_calc("then <30*1/2=15>>15 more") == "then 15 more"


def test_lowercased_text_is_recased_only_when_all_lower():
    assert U.recase("ntụpọ. spot hụrụ ụgbọ ala. kitty gara.") == "Ntụpọ. Spot hụrụ ụgbọ ala. Kitty gara."
    assert U.recase("Already fine. Second one.") == "Already fine. Second one."


def test_placeholder_values_count_as_empty():
    assert U.real("None") == "" and U.real("#VALUE!") == "" and U.real(" text ") == "text"


def test_every_template_task_renders_in_every_language():
    rng = random.Random(0)
    for task in list(T._POOLS):
        for lang in T.languages(task):
            for _ in range(3):
                ins = T.make(task, rng, target="yor", source="hau", lang=lang, title="Biyi Bandele", seed="Once upon a time.", n=3)
                assert ins.text.strip() and "{" not in ins.text, (task, lang)
                assert ins.position in ("prepend", "append")


def test_template_pools_cover_the_seven_instruction_languages_for_main_tasks():
    for task in ("summarize", "headline", "sentiment", "topic", "story", "answer_in", "wiki_ask", "mcq", "math"):
        assert {"eng", "pcm", "yor", "hau", "ibo"} <= set(T.languages(task)), task


def test_stopword_lists_exclude_english_function_words():
    from sft_new import langdetect as L
    sw = L.stopwords()
    assert "the" not in sw["pcm"] and "the" in sw["eng"]
    assert L.code_of("<swa>") == "swh" and L.code_of("Hausa") == "hau" and L.code_of("lug") is None


def _first_turn(rec):
    return json.loads(finish(rec, random.Random(0))["messages"])[-1]["content"]


def test_input_lang_lists_instruction_and_text_languages_in_reading_order():
    base = dict(task="translation", dataset="t", src="efi", tgt="eng", response="Hello", input="Ekem ɔyɔhɔ", instruction="Sụgharịa:", plan=["<translate>"])
    assert _first_turn(Rec(**base, position="prepend", meta={"instruction_language": "ibo"})).startswith("<|input_lang|><ibo><efi><task_plan>")
    assert _first_turn(Rec(**base, position="append", meta={"instruction_language": "ibo"})).startswith("<|input_lang|><efi><ibo><task_plan>")
    assert _first_turn(Rec(**base, meta={"instruction_language": "efi"})).startswith("<|input_lang|><efi><task_plan>")     # one language -> one tag
    assert _first_turn(Rec(**base)).startswith("<|input_lang|><efi><task_plan>")                                          # unknown instruction language
    unsure = _first_turn(Rec(**base, unsure_src=True, meta={"instruction_language": "ibo"}))
    assert unsure.startswith("<|input_lang|> unsure, but most likely <ibo><efi><task_plan>")
    assert normalize_turn(unsure) == unsure


def test_explicit_input_langs_and_the_three_language_limit():
    from sft_new.compose import A
    rec = Rec("x", "t", "yor", "yor", "ans", input="q", instruction="i", plan=["<|chat|>"], input_langs=["yor", "eng"])
    assert _first_turn(rec).startswith("<|input_lang|><yor><eng><task_plan>")
    with pytest.raises(A.AssemblyError):
        A.assistant_content(input_lang=["yor", "eng", "hau", "ibo"], target_lang="yor", task_plan=["<|chat|>"], response="x")
    with pytest.raises(A.AssemblyError):
        A.assistant_content(input_lang=["yor", "yor"], target_lang="yor", task_plan=["<|chat|>"], response="x")


def test_prompt_languages_follow_the_hints_and_reading_order(monkeypatch):
    from sft_new import langdetect as L

    def fake(texts):
        out = []
        for t in texts:
            code = "ibo" if "Igbo" in t else "efi" if "Efik" in t else None
            out.append(L.Detection(code, "certain" if code else "none", (code,), code, len(t.split())))
        return out
    monkeypatch.setattr(L, "detect_many", fake)
    assert L.prompt_languages("Igbo instruction here now\nEfik passage with enough words", hints=["ibo", "efi"]) == ["ibo", "efi"]
    assert L.prompt_languages("Efik passage with enough words\nIgbo instruction here now", hints=["ibo", "efi"]) == ["efi", "ibo"]
    assert L.prompt_languages("Igbo only text with enough words", hints=["ibo", "efi"]) == ["ibo", "efi"]          # the undetected hint keeps its side
    assert L.prompt_languages("Efik only text with enough words", hints=["ibo", "efi"]) == ["ibo", "efi"]
    assert L.prompt_languages("Igbo text and\nEfik text here too") == ["ibo", "efi"]                                 # no hints: certain detections only


def test_sft_new_existing_task_comes_from_the_folder():
    from sft_new.converters import _sft_new_task
    assert _sft_new_task({"task": "identify", "__file": "data/structured-identify/part-0.parquet"}) == "identify"
    assert _sft_new_task({"__file": "data/translation-multi-open-afr/part-0.parquet"}) == "translation"
    assert _sft_new_task({"task": None, "__file": "data/instruct-swim-ir-swh/part-0.parquet"}) == "instruct"
    assert _sft_new_task({"__file": "data/other/part-0.parquet"}) == "unknown"


def test_empty_or_null_input_and_instruction_are_normalised_in_finish():
    import random
    from sft_new.compose import Rec, Skip, finish, blank
    rng = random.Random(0)
    assert blank(None) == "" and blank(float("nan")) == "" and blank(" None ") == "" and blank("N/A") == "" and blank("  a  b ") == "a  b"
    # instruction only (input null / the string 'None' / empty): the prompt is the instruction, no 'None' leaks into the turn
    for inp in (None, "None", "", "  ", "nan"):
        row = finish(Rec("qa", "t", "eng", "eng", "Four.", input=inp, instruction="What is 2+2?", plan=["<qa>"]), rng)
        assert row["input"] == "" and "None" not in row["messages"] and "nan" not in row["messages"]
        assert json.loads(row["messages"])[0]["content"] == "What is 2+2?"
    # input only
    row = finish(Rec("qa", "t", "eng", "eng", "Four.", input="What is 2+2?", instruction=None, plan=["<qa>"]), rng)
    assert json.loads(row["messages"])[0]["content"] == "What is 2+2?"
    # both empty, or a placeholder response: dropped, not written
    for kw in ({"input": None, "instruction": "None"}, {"input": "x", "instruction": "y", "response": "null"}):
        rec = Rec("qa", "t", "eng", "eng", kw.pop("response", "Four."), plan=["<qa>"], **kw)
        try:
            finish(rec, rng)
            raise AssertionError("should have been skipped")
        except Skip:
            pass


def test_msgs_tolerates_null_conversations():
    from sft_new.converters import _msgs
    assert _msgs({"chat_data": None, "data": None}) == ("", "", 0)
    assert _msgs({"data": '[{"role": "user", "content": null}, {"role": "assistant", "content": "x"}]'}) == ("", "x", 2)
    assert _msgs({"data": '[{"role": "user", "content": "None"}, {"role": "assistant", "content": " nan "}]'}) == ("", "", 2)


def test_blank_num_and_nan_cells():
    from sft_new.textutil import blank, num
    assert [blank(x) for x in (None, float("nan"), 3, "", " ", "None", "NULL", "NaN", "n/a", "#VALUE!", " ok ")] == ["", "", "", "", "", "", "", "", "", "", "ok"]
    assert num(None) == 0.0 and num("") == 0.0 and num("nan") == 0.0 and num(float("nan"), 1.0) == 1.0 and num("3.5") == 3.5


def test_convo_rows_tolerate_null_content_and_dedupe_key_never_raises():
    import random
    from sft_new.compose import Rec, Skip, dedupe_key, finish
    rng = random.Random(0)
    convo = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {}}}]},
             {"role": "tool", "name": "f", "content": "r"}, {"role": "assistant", "content": "done"}]
    rec = Rec("rag_qa", "t", "eng", "eng", None, input=None, instruction=None, convo=convo)
    assert finish(rec, rng)["response"] == ""                   # no Rec.response was given: the row is still written, nothing raised on the None fields
    assert dedupe_key(Rec("qa", "t", "eng", "eng", None, input=None), [{"role": "user", "content": None}]).count("\x1f") == 4
    for bad in ([{"role": "user", "content": None}, {"role": "assistant", "content": "x"}],
                [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "None"}]):
        try:
            finish(Rec("qa", "t", "eng", "eng", "x", convo=bad), rng)
            raise AssertionError("should skip")
        except Skip:
            pass


# --------------------------------------------------------------------------- tool calling in the training text

def _wiki_row(rng_seed=0):
    import random
    from sft_new import converters as C
    from sft_new.compose import finish
    rng = random.Random(rng_seed)
    words = [f"ọ̀rọ̀{i}" for i in range(300)]                                                  # varied enough to pass the repetition filter
    ds = [{"title": "Biyi Bandele", "text": "Biyi Bandele " + " ".join(words[:120]) + ". " + " ".join(words[120:260]) + "."}]
    old = C._rows
    C._rows = lambda repo, limit, **kw: iter(ds * 30)
    try:
        recs = [r for r in C.wiki_yo(rng, None) if getattr(r, "convo", None)]
    finally:
        C._rows = old
    return finish(recs[0], rng), recs[0]


def test_wiki_yo_tool_rows_follow_the_standard_shape():
    row, rec = _wiki_row()
    text = row["text"]
    assert text.count("<|assistant|>") == 1 and text.endswith("</s>") and text.count("</s>") == 3      # system, user, and the ONE closing </s> of the turn
    assert "<tools>\n" in text and text.index("<tools>") < text.index("<|user|>")                         # definitions live in the system turn
    call = text.split("<tool_call>")[1].split("</tool_call>")[0]
    obj = json.loads(call)
    assert set(obj) == {"name", "arguments"} and obj["name"] == rec.tools[0]["function"]["name"]
    assert "</tool_call><tool_response>" in text and "</tool_response><|input_lang|>" in text
    assert json.loads(row["tools"])[0]["function"]["name"] == obj["name"]
    assert "search_wikipedia " not in text.split("<tool_response>")[1][:40] and not text.split("<tool_response>")[1].startswith(obj["name"])   # no function name in the result


def test_render_keeps_non_ascii_and_argument_order_like_transformers():
    from sft_new.render import render
    m = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"zeta": "Ẹ̀gbọ́n 'x' <y> & z", "alpha": 1}}}]}]
    out = render(m)
    assert '"zeta": "Ẹ̀gbọ́n \'x\' <y> & z", "alpha": 1' in out and "\\u" not in out


def test_split_tools_reads_both_conventions():
    from sft_new.converters import split_tools
    tool = {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}
    new = "You are helpful.\n\n<tools>\n" + json.dumps(tool) + "\n</tools>\nBe brief."
    assert split_tools(new) == ("You are helpful.\n\nBe brief.", [tool])
    old = "Intro.\nYou have access to the following tools:\n" + json.dumps([tool], indent=2) + "\nOutro."
    assert split_tools(old) == ("Intro.\n\nOutro.", [tool])
    assert split_tools("plain system prompt") == ("plain system prompt", None)
