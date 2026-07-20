"""Special token IDs for label masking."""

from __future__ import annotations

import os

MASK = -100


def _config_path() -> str:
    return os.environ.get(
        "TRAIN_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "train_config.yaml"),
    )


def _tokenizer_name() -> str:
    from omegaconf import OmegaConf

    items = OmegaConf.load(_config_path()).get("tokenizer", [])
    if OmegaConf.is_dict(items):
        # Current train_config.yaml format: `tokenizer: {name: ..., num_proc: ..., ...}`.
        # Iterating a DictConfig yields its keys, not values -- handle it explicitly
        # instead of falling into the list-style loop below and returning the key "name".
        return items.get("name") or "BeardedMonster/SabiYarn-32k"
    for item in items:
        if isinstance(item, str):
            return item
    return "BeardedMonster/SabiYarn-32k"


def _load():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_tokenizer_name())


tokenizer = _load()
_id = lambda s: tokenizer.encode(s, add_special_tokens=False)[-1]

# Chat SFT delimiters
system_token = _id("<|system|>")
user_token = _id("<|user|>")
assistant_token = _id("<|assistant|>")

# Task tokens
lang_id_token = _id("<lang_ID>")
lang_id_label_token = _id("<lang_ID_label>")
classify_token = _id("<classify>")
sentiment_token = _id("<sentiment>")
topic_token = _id("<topic>")
qa_token = _id("<qa>")
answer_token = _id("<answer>")
tag_token = _id("<tag>")
diacritize_token = _id("<diacritize>")
correct_token = _id("<correct>")
clean_token = _id("<clean>")
summarize_token = _id("<summarize>")
summary_token = _id("<summary>")
title_token = _id("<title>")
headline_token = _id("<headline>")
translate_token = _id("<translate>")
ner_token2 = _id("<ner>")
ner_token = _id("<NER>")
str_token = _id("<STR>")
lang_id_token2 = _id("<identify>")
lang_id_label_token2 = _id("<lang_id>")
summary_token2 = _id("<text>")
prompt_token = _id("<prompt>")
response_token = _id("<response>")
end_of_text_token = (
    _id("<|end_of_text|>") if "<|end_of_text|>" in tokenizer.get_vocab() else tokenizer.eos_token_id
)

prompting_tokens = [
    lang_id_token, classify_token, qa_token, diacritize_token, clean_token,
    summarize_token, title_token, translate_token, ner_token2, ner_token,
    str_token, lang_id_token2, summary_token2, prompt_token,
]

action_tokens = [
    assistant_token, response_token, _id("<toxic>"), _id("<intent>"), _id("<score>"),
    answer_token, tag_token, correct_token, lang_id_label_token, lang_id_label_token2,
    sentiment_token, topic_token, summarize_token, headline_token,
    _id("<eng>"), _id("<yor>"), _id("<ibo>"), _id("<hau>"), _id("<pcm>"),
    _id("<ff>"), _id("<fuv>"), _id("<ful>"), _id("<urh>"), _id("<efi>"),
    _id("<kea>"), _id("<lug>"), _id("<tsn>"), _id("<afr>"), _id("<din>"),
    _id("<xsm>"), _id("<zu>"), _id("<tmh>"), _id("<ti>"), _id("<tzm>"),
    _id("<ny>"), _id("<arb>"), _id("<dyu>"), _id("<fra>"), _id("<kab>"),
    _id("<amh>"), _id("<swh>"), _id("<snq>"), _id("<ton>"), _id("<vag>"),
    _id("<nup>"), _id("<kmb>"), _id("<mey>"), _id("<luo>"), _id("<sn>"),
    _id("<nus>"), _id("<ven>"), _id("<oke>"), _id("<xh>"), _id("<son>"),
    _id("<igl>"), _id("<kik>"), _id("<wolof>"), _id("<sag>"), _id("<aku>"),
    _id("<tso>"), _id("<ewe>"), _id("<ngl>"), _id("<run>"), _id("<gah>"),
    _id("<bm>"), _id("<kbp>"), _id("<umb>"), _id("<aka>"), _id("<lin>"),
    _id("<tum>"), _id("<nso>"), _id("<ssw>"), _id("<fat>"), _id("<som>"),
    _id("<vai>"), _id("<tag>"), _id("<sot>"), _id("<mos>"), _id("<tiv>"),
    _id("<kon>"), _id("<fon>"), _id("<twi>"), _id("<nde>"), _id("<bem>"),
    _id("<knc>"), _id("<nya>"), _id("<orm>"), _id("<oro>"), _id("<mlg>"),
    _id("<shi>"), _id("<lus>"), _id("<gaa>"), _id("<ibb>"), _id("<kin>"),
    _id("<mzw>"), _id("<kam>"),
]
