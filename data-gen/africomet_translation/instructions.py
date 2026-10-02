"""Multilingual, structurally diverse translation instructions.

Each instruction language has three banks:
  pre  -- written to PRECEDE the text ("Translate the following ...", "... below")
  post -- written to FOLLOW the text  ("Translate the text above ...")
  any  -- position-neutral            ("Translate this into {target}.")
Placeholders: {target} (always) and {source} (some templates). A name is picked per sample from the
language's display names, so the same template yields e.g. "Yoruba" / "Yorùbá".

COVERAGE AND CONFIDENCE -- read before trusting any of this in a release:
  eng                      high      written natively
  pcm, yor, hau, ibo       medium    competent, not native-reviewed
  twi, ewe                 low       best effort; Ewe especially needs a native speaker
  fon, efi, ibb            NONE      deliberately NOT authored: inventing instruction text in languages this
                                     author cannot write reliably would put wrong grammar into an open dataset.
                                     Add a bank below (or in TEMPLATES) and they join the instruction-language
                                     rotation automatically.
Every record carries `instruction_language` and `template_confidence`, so any tier can be dropped later
without regenerating.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

LANGS = ("eng", "pcm", "yor", "hau", "ibo", "fon", "ewe", "twi", "efi", "ibb")

# Names used when a language is *mentioned* inside an instruction.
DISPLAY_NAMES: dict[str, list[str]] = {
    "eng": ["English"],
    "pcm": ["Nigerian Pidgin", "Pidgin"],
    "yor": ["Yoruba", "Yorùbá"],
    "hau": ["Hausa"],
    "ibo": ["Igbo"],
    "fon": ["Fon", "Fɔ̀ngbè"],
    "ewe": ["Ewe", "Èʋegbe"],
    "twi": ["Twi"],
    "efi": ["Efik", "Efịk"],
    "ibb": ["Ibibio"],
}
# How the word for English reads in an instruction written in that language (confident cases only).
LOCALIZED_ENGLISH = {"yor": "Gẹ̀ẹ́sì", "hau": "Turanci", "ibo": "Bekee"}

CONFIDENCE = {"eng": "high", "pcm": "medium", "yor": "medium", "hau": "medium", "ibo": "medium",
              "twi": "low", "ewe": "low"}

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "eng": {
        "pre": [
            "Translate the following text into {target}.",
            "Translate into {target}:",
            "Please translate the passage below into {target}.",
            "Could you render the following in {target}? Keep the meaning and tone intact.",
            "{target} translation, please:",
            "I would be grateful if you could translate the text that follows into {target}, keeping it natural rather than word-for-word.",
            "You are a professional translator. Translate the {source} passage below into {target}, preserving names, numbers and punctuation. Output only the translation.",
            "Convert this {source} text to {target}.",
            "What is the {target} version of the following text?",
            "Rewrite the passage below in {target}, making sure it reads the way a native speaker would say it.",
            "My colleague sent me this in {source}, but I need it in {target} for a meeting this afternoon. Can you translate it for me?",
            "Give me a faithful {target} translation of what follows. Do not add explanations or notes.",
            "Translate to {target}. Keep any proper names as they are.",
            "Below is a short text in {source}. Provide its translation in {target}, and nothing else.",
            "Render the text below into fluent {target}, retaining the formality of the original.",
            "Help me say the following in {target}:",
            "Express the content of this passage in {target}; accuracy matters more than elegance.",
            "Please produce a {target} rendering of the text below. If a phrase has no direct equivalent, choose the closest natural expression.",
            "From {source} to {target}:",
            "Kindly translate the following into {target} for a community newsletter. Keep the language simple and clear.",
            "Whatever the original wording, I want the passage below conveyed accurately in {target}.",
            "Task: translation. Source language: {source}. Target language: {target}. Translate the text below.",
            "Can you put the following into {target}?",
            "Using natural, idiomatic {target}, translate the text below. Avoid literal calques.",
            "Imagine you are explaining the text below to a {target} speaker who has never heard of its subject; translate it faithfully into {target}.",
            "The paragraph that follows needs to reach {target}-speaking readers. Translate it, taking care that no sentence is dropped or invented.",
        ],
        "post": [
            "Translate the text above into {target}.",
            "Please render the passage above in {target}.",
            "Can you translate this into {target}?",
            "The text above is in {source}. Translate it into {target}.",
            "Translate the above into {target}, keeping names and numbers unchanged.",
            "Give me the {target} translation of the above, with no extra commentary.",
            "How would you say all of this in {target}?",
            "That is the passage. I need it in {target}, translated faithfully and naturally.",
            "Now translate everything above into {target}.",
            "Put the preceding text into {target}, please. Natural phrasing over literal wording.",
            "→ {target}",
            "Translate the preceding {source} text into clear, natural {target}, preserving its meaning and register. Reply with the translation only.",
            "Could you translate it into {target} for me? I would like to share it with someone who reads {target}.",
            "Please provide a translation in {target}.",
            "Translate to {target}.",
            "Everything above has to be available in {target} by tomorrow. Translate it so that a native reader would not guess it was ever written in another language.",
        ],
        "any": [
            "Translate this text into {target}.",
            "Translate this passage into {target}, keeping the meaning the same.",
            "I need a {target} translation of this text.",
            "Say this in {target}.",
            "Translate into {target}.",
            "Would you mind translating this into {target}?",
            "{source} → {target}",
        ],
    },
    "pcm": {
        "pre": [
            "Abeg translate dis text go {target}.",
            "Abeg, make you change dis one to {target}:",
            "I wan make you translate wetin dey below go {target}. Make e sound natural, no be word-for-word.",
            "Translate am to {target}.",
            "You fit help me put dis {source} text for {target}? Abeg write only di translation.",
            "Dis one dey {source}, but I need am for {target}. Abeg translate am well well.",
            "Make you do {target} translation for dis one wey dey follow, but keep di names and numbers as dem be.",
            "Wetin be di {target} version of dis text?",
            "Abeg help me translate di passage wey dey below go {target}; I no want any extra talk, just di translation.",
            "Translate di following text into {target}, make e correct and make e flow.",
        ],
        "post": [
            "Abeg translate di text wey dey up go {target}.",
            "Make you change all dis one go {target}.",
            "Translate wetin dey above into {target}, no add anything.",
            "Dis na {source}. Abeg put am for {target}.",
            "You fit translate di thing wey I just write go {target}?",
            "Go {target}, abeg.",
            "I need di {target} version of di text wey dey up. Make e sound like how person for dat language go talk am.",
        ],
        "any": [
            "Translate dis one go {target}.",
            "Abeg put dis text for {target}.",
            "{target} translation abeg.",
        ],
    },
    "yor": {
        "pre": [
            "Túmọ̀ ọ̀rọ̀ yìí sí èdè {target}.",
            "Jọ̀wọ́, túmọ̀ àyọkà tó wà nísàlẹ̀ yìí sí èdè {target}.",
            "Mo fẹ́ kí o túmọ̀ ìpínrọ̀ yìí sí èdè {target}, kí ìtumọ̀ náà dùn mọ́ni, kí ó má ṣe jẹ́ ọ̀rọ̀-sí-ọ̀rọ̀ lásán.",
            "Ṣé o lè ràn mí lọ́wọ́ láti túmọ̀ èyí tó wà nísàlẹ̀ sí èdè {target}?",
            "Yí ọ̀rọ̀ tó tẹ̀lé e padà sí èdè {target}:",
            "Ọ̀rọ̀ yìí wà ní èdè {source}. Mo nílò ìtumọ̀ rẹ̀ ní èdè {target}; kọ ìtumọ̀ náà nìkan sílẹ̀, má fi àlàyé kankan kún un.",
            "Èdè {target} ni mo fẹ́ kí o fi túmọ̀ àyọkà yìí, ṣùgbọ́n má yí orúkọ àwọn ènìyàn tàbí àwọn ibi padà.",
            "Kọ ìtumọ̀ èdè {target} fún ìpínrọ̀ tó wà nísàlẹ̀ yìí.",
            "Láti èdè {source} sí èdè {target}:",
            "Jọ̀wọ́ túmọ̀ ohun tí ó wà nísàlẹ̀ yìí sí èdè {target} ní ọ̀nà tí ó rọrùn láti lóye, tí ó sì bá ọ̀rọ̀ inú rẹ̀ mu.",
        ],
        "post": [
            "Túmọ̀ ọ̀rọ̀ tó wà lókè yìí sí èdè {target}.",
            "Jọ̀wọ́, túmọ̀ àyọkà òkè yìí sí èdè {target}.",
            "Ṣé o lè túmọ̀ gbogbo èyí sí èdè {target}?",
            "Ọ̀rọ̀ tí mo kọ sókè wà ní èdè {source}; túmọ̀ rẹ̀ sí èdè {target} lọ́nà tí ó yé ni dáadáa.",
            "Èdè {target} ni mo fẹ́ kí o fi kọ ohun tí ó wà lókè yìí.",
            "Túmọ̀ rẹ̀ sí {target}.",
        ],
        "any": [
            "Túmọ̀ èyí sí èdè {target}.",
            "Mo fẹ́ ìtumọ̀ èyí ní èdè {target}.",
            "Sọ èyí ní èdè {target}.",
        ],
    },
    "hau": {
        "pre": [
            "Fassara wannan rubutu zuwa {target}.",
            "Don Allah, ka fassara wannan sakin layi zuwa {target}.",
            "Ina so ka fassara rubutun da ke ƙasa zuwa {target}, ta yadda zai yi daidai kuma ya fito a zahiri, ba kalma-da-kalma kawai ba.",
            "Za ka iya taimaka mini in fassara abin da ke ƙasa zuwa {target}?",
            "Juya wannan zuwa {target}:",
            "Wannan rubutu yana cikin {source}. Ina buƙatar fassararsa zuwa {target}; rubuta fassarar kawai, kada ka ƙara bayani.",
            "Daga {source} zuwa {target}:",
            "A fassara min rubutun nan da ke biye zuwa {target}, kuma a bar sunayen mutane da na wurare yadda suke.",
        ],
        "post": [
            "Fassara rubutun da ke sama zuwa {target}.",
            "Don Allah ka fassara abin da na rubuta a sama zuwa {target}.",
            "Za ka iya fassara wannan duka zuwa {target}?",
            "Ka mayar da rubutun da ke sama zuwa {target}, ba tare da ƙarin bayani ba.",
            "Fassara shi zuwa {target}.",
        ],
        "any": [
            "Fassara wannan zuwa {target}.",
            "Ina buƙatar fassarar wannan zuwa {target}.",
        ],
    },
    "ibo": {
        "pre": [
            "Sụgharịa ederede a gaa n'asụsụ {target}.",
            "Biko, sụgharịa paragraf dị n'okpuru a gaa n'asụsụ {target}.",
            "Achọrọ m ka ị sụgharịa ederede a gaa n'asụsụ {target}, ka ọ dị ka onye na-asụ ya si asụ ya, ọ bụghị naanị okwu-n'okwu.",
            "Ị nwere ike inyere m aka sụgharịa ihe dị n'okpuru a gaa n'asụsụ {target}?",
            "Gbanwee ihe a na-esonụ gaa n'asụsụ {target}:",
            "Ederede a bụ n'asụsụ {source}. Achọrọ m nsụgharị ya n'asụsụ {target}; dee naanị nsụgharị ahụ, agbakwunyela nkọwa ọ bụla.",
            "Site n'asụsụ {source} gaa n'asụsụ {target}:",
        ],
        "post": [
            "Sụgharịa ederede dị n'elu gaa n'asụsụ {target}.",
            "Biko, sụgharịa ihe m dere n'elu gaa n'asụsụ {target}.",
            "Ị nwere ike isụgharị ihe a niile gaa n'asụsụ {target}?",
            "Dee ya n'asụsụ {target}.",
        ],
        "any": [
            "Sụgharịa nke a gaa n'asụsụ {target}.",
            "Achọrọ m nsụgharị nke a n'asụsụ {target}.",
        ],
    },
    "twi": {
        "pre": [
            "Kyerɛ nsɛm a ɛwɔ ase yi ase wɔ {target} mu.",
            "Mesrɛ wo, kyerɛ saa nsɛm yi ase wɔ {target} mu.",
            "Fi {source} mu kɔ {target} mu:",
        ],
        "post": [
            "Kyerɛ nsɛm a ɛwɔ soro no ase wɔ {target} mu.",
            "Mesrɛ wo, kyerɛ nea metwerɛe wɔ atifi no ase wɔ {target} mu.",
        ],
        "any": [
            "Kyerɛ yei ase wɔ {target} mu.",
            "Kyerɛw yei wɔ {target} mu.",
        ],
    },
    "ewe": {
        "pre": [
            "Ɖe nya siwo le ete lae gɔme ɖe {target} me.",
            "Taflatse, ɖe nuŋɔŋlɔ sia gɔme ɖe {target} me.",
        ],
        "post": [
            "Ɖe nya siwo le dziŋ gɔme ɖe {target} me.",
        ],
        "any": [
            "Ɖe esia gɔme ɖe {target} me.",
        ],
    },
}

# English instructions are additionally composed from optional context leads and trailing constraints,
# which multiplies the structural variety well beyond the hand-written list.
ENG_LEAD_INS = [
    "I'm preparing material for a bilingual audience.",
    "This is for a school project.",
    "A friend asked me to help with this.",
    "I'm writing to a relative who prefers another language.",
    "We're localizing content for a community radio station.",
    "I came across this text and want to share it.",
    "This will go into a handout for a workshop.",
]
ENG_TAILS = [
    "Output only the translation.",
    "Keep proper names unchanged.",
    "Use a natural, conversational tone.",
    "Keep the formal register of the original.",
    "Do not add explanations.",
    "Preserve the original punctuation.",
    "If a term has no direct equivalent, keep the original term.",
    "Prioritise fidelity over style.",
]


@dataclass(frozen=True)
class Instruction:
    text: str
    position: str            # "prepend" | "append"
    language: str            # instruction language
    confidence: str


def instruction_languages() -> list[str]:
    return [l for l in LANGS if TEMPLATES.get(l)]


def _name(lang: str, mention_in: str, rng: random.Random) -> str:
    if lang == "eng" and mention_in in LOCALIZED_ENGLISH and rng.random() < 0.7:
        return LOCALIZED_ENGLISH[mention_in]
    return rng.choice(DISPLAY_NAMES[lang])


def make_instruction(
    instruction_lang: str, source: str, target: str, rng: random.Random, position: str | None = None,
) -> Instruction:
    bank = TEMPLATES[instruction_lang]
    position = position or rng.choice(["prepend", "append"])
    pool = bank["pre" if position == "prepend" else "post"] + bank["any"]
    template = rng.choice(pool)
    text = template.format(
        target=_name(target, instruction_lang, rng), source=_name(source, instruction_lang, rng),
    )
    if instruction_lang == "eng":
        short = len(template.split()) <= 11 and "only" not in template
        if short and rng.random() < 0.35:
            tail = rng.choice(ENG_TAILS)
            text = f"{text} {tail}"
        if position == "prepend" and rng.random() < 0.2:
            text = f"{rng.choice(ENG_LEAD_INS)} {text}"
    return Instruction(text, position, instruction_lang, CONFIDENCE.get(instruction_lang, "low"))


def compose_prompt(instruction: Instruction, text: str, rng: random.Random) -> str:
    if instruction.position == "prepend":
        sep = rng.choice(["\n\n", "\n", "\n\n", " "] if not instruction.text.endswith(":") else ["\n", "\n\n", " "])
        return f"{instruction.text}{sep}{text}"
    return f"{text}{rng.choice(['\n\n', '\n', ' '])}{instruction.text}"
