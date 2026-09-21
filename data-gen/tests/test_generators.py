"""Request building: strict schemas, unique ids, no English target, manifest context, prompt content."""

from __future__ import annotations

import json
from typing import Any

import pytest

from config.corpus_config import TARGET_LANGUAGE_CODES
from conftest import tiny
from generators import dpo, pretrain, sft_tasks
from generators.base import BatchRequestSpec
from schemas.corpus import RejectionType
from typing import get_args

GENS = {"pretrain": pretrain, "sft": sft_tasks, "dpo": dpo}


def assert_strict(node: Any, path: str = "$") -> None:
    """Every object in an OpenAI strict json_schema: additionalProperties=false, all keys required, no $ref/default."""
    if isinstance(node, dict):
        assert "$ref" not in node and "default" not in node, path
        if node.get("type") == "object" or "properties" in node:
            assert node["additionalProperties"] is False, path
            assert node["required"] == list(node["properties"]), path
            for k, v in node["properties"].items():
                assert_strict(v, f"{path}.{k}")
        for k in ("items", "anyOf", "oneOf"):
            if k in node:
                assert_strict(node[k], f"{path}.{k}")


@pytest.mark.parametrize("kind", GENS)
def test_requests_are_valid_batch_lines(presets, kind):
    cfg = tiny(presets[kind], 4)
    specs = GENS[kind].build_requests(cfg)
    assert len(specs) == 4 * 12
    ids = [s.custom_id for s in specs]
    assert len(set(ids)) == len(ids)
    for s in specs:
        assert isinstance(s, BatchRequestSpec)
        assert s.custom_id.startswith(f"{kind}__{s.language}__")
        assert s.language in TARGET_LANGUAGE_CODES and s.language != "eng"
        assert s.model == "gpt-4o-mini"
        line = json.loads(json.dumps(s.to_batch_line(), ensure_ascii=False))
        assert line["method"] == "POST" and line["url"] == "/v1/chat/completions"
        fmt = line["body"]["response_format"]
        assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
        assert_strict(fmt["json_schema"]["schema"])
        assert [m["role"] for m in line["body"]["messages"]] == ["system", "user"]
        assert line["body"]["max_tokens"] == cfg.max_tokens


@pytest.mark.parametrize("kind", GENS)
def test_english_is_never_a_target_language(presets, kind):
    cfg = tiny(presets[kind], 6)
    for s in GENS[kind].build_requests(cfg):
        assert "Target language: English" not in s.user_prompt
        assert s.language != "eng"
    with pytest.raises(ValueError, match="Unsupported"):
        GENS[kind].build_requests(cfg, ["eng"])


@pytest.mark.parametrize("kind", GENS)
def test_manifest_context_stores_the_sampled_attribute_tuple(presets, kind):
    expected = {
        "pretrain": {"domain", "subtopic", "genre", "register", "audience", "length_bucket", "perspective", "era", "difficulty", "locale"},
        "sft": {"domain", "subtopic", "task", "register", "instruction_style", "response_length", "difficulty", "locale"},
        "dpo": {"domain", "subtopic", "task", "rejection_type", "register", "instruction_style", "response_length", "difficulty", "locale"},
    }[kind]
    for s in GENS[kind].build_requests(tiny(presets[kind], 3)):
        m = s.to_manifest_line()
        assert m["language"] == s.language and m["custom_id"] == s.custom_id
        assert set(m["context"]["attributes"]) == expected
        assert m["context"]["kind"] == kind


@pytest.mark.parametrize("kind", GENS)
def test_generation_is_deterministic(presets, kind):
    cfg = tiny(presets[kind], 3, ["yor", "fuv"])
    a = [s.to_batch_line() for s in GENS[kind].build_requests(cfg)]
    b = [s.to_batch_line() for s in GENS[kind].build_requests(cfg)]
    assert a == b


def test_pretrain_prompt_contains_required_grounding_and_rules(presets):
    spec = pretrain.build_requests(tiny(presets["pretrain"], 1, ["yor"]))[0]
    for needle in ("Yoruba", "Domain:", "Specific topic:", "Genre:", "Register:", "Audience:", "Setting:", "Perspective:", "Time frame:", "Length:", "naira"):
        assert needle in spec.user_prompt, needle
    sp = spec.system_prompt.lower()
    for needle in ("translated english", "never translate", "diacritics", "never invent statistics", "no markdown headings",
                   "never mention that you are an ai", "never repeat a paragraph", "short, grammatically safe", "no english words",
                   "hedged", "local names", "no bracketed placeholders"):
        assert needle in sp, needle
    props = spec.response_format["json_schema"]["schema"]["properties"]
    assert set(props) == {"title", "text", "language_self_check"}


def test_low_resource_languages_get_the_simplicity_instruction(presets):
    spec = pretrain.build_requests(tiny(presets["pretrain"], 1, ["efi"]))[0]
    assert "lower-resource" in spec.user_prompt


def test_sft_catalogue():
    tasks = sft_tasks.TASKS
    assert len(tasks) >= 24
    required_types = ["open_qa", "factual_qa", "reading_comprehension", "summarization", "title_generation", "keyword_extraction",
                      "translation_en_to_lang", "translation_lang_to_en", "paraphrase", "formality_rewrite", "grammar_correction",
                      "text_classification", "ner_extraction", "info_extraction_json", "sentence_completion", "creative_story", "creative_poem",
                      "letter_email_drafting", "explanation_howto", "brainstorming_list", "math_word_problem", "statistics_simple",
                      "commonsense_reasoning", "logical_reasoning", "dialogue_continuation", "proverb_explanation", "data_to_text",
                      "format_constrained", "safe_decline", "advice"]
    assert set(required_types) <= set(tasks)
    for task in tasks.values():
        assert task.weight > 0 and task.what and task.input_note and task.response_note
        assert task.input_mode in ("required", "empty", "optional")
        assert task.lengths
    assert tasks["translation_en_to_lang"].english_fields == ("input",)
    assert tasks["translation_lang_to_en"].english_fields == ("response",)
    assert tasks["reading_comprehension"].input_mode == "required"


def test_sft_prompt_language_directives(presets):
    cfg = tiny(presets["sft"], 60, ["yor"])
    seen = set()
    for s in sft_tasks.build_requests(cfg):
        task = s.context["attributes"]["task"]
        seen.add(task)
        if task == "translation_en_to_lang":
            assert "Write input in plain, natural English" in s.user_prompt
            assert s.context["english_fields"] == ["input"]
        if task == "translation_lang_to_en":
            assert "Write response in plain, natural English" in s.user_prompt
        if task in ("open_qa", "summarization"):
            assert "in plain, natural English" not in s.user_prompt
        props = s.response_format["json_schema"]["schema"]["properties"]
        assert set(props) == {"instruction", "input", "response", "confidence"}
    assert len(seen) >= 20  # 60 draws already spread across many task types


def test_dpo_catalogue_and_prompts(presets):
    assert set(dpo.REJECTIONS) == set(get_args(RejectionType))
    assert len(dpo.REJECTIONS) == 12
    for r in dpo.REJECTIONS.values():
        assert len(r.flaw) > 40
    specs = dpo.build_requests(tiny(presets["dpo"], 40, ["hau"]))
    seen_types = set()
    for s in specs:
        rt = s.context["rejection_type"]
        seen_types.add(rt)
        assert f"Requested flaw type for `rejected`: {rt}" in s.user_prompt
        assert s.context["length_related"] == (rt in dpo.LENGTH_RELATED)
        props = s.response_format["json_schema"]["schema"]["properties"]
        assert set(props) == {"instruction", "input", "chosen", "rejected", "rejection_type", "chosen_confidence"}
        assert "PLAUSIBLE" in s.system_prompt
    assert len(seen_types) >= 8
    assert "same language" in specs[0].system_prompt.lower()


def test_dpo_translation_directive_maps_response_to_chosen_and_rejected():
    text = sft_tasks.language_directive("yor", ("response",), dpo=True)
    assert "chosen, rejected in plain, natural English" in text
    assert "instruction, input in Yoruba" in text
