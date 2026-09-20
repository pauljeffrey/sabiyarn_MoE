"""Language registry: the model's language tag plus each benchmark's code for that language.

A benchmark code of None means the benchmark does not cover the language (Efik and Urhobo have
no standard benchmark in any of these tasks; use --custom-translation for your own data).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Lang:
    code: str  # the model's language code (matches the <xxx> tag)
    name: str
    flores: Optional[str] = None  # FLORES(+)-200 code
    sib: Optional[str] = None  # SIB-200 (topic) config
    news: Optional[str] = None  # MasakhaNEWS (topic) config
    senti: Optional[str] = None  # AfriSenti (sentiment) config
    ner: Optional[str] = None  # MasakhaNER 2.0 config
    mmlu: Optional[str] = None  # AfriMMLU config
    mafand: Optional[str] = None  # MAFAND-MT en-<x> config suffix (English-pivot pairs only)

    @property
    def tag(self) -> str:
        return f"<{self.code}>"


LANGS: dict[str, Lang] = {
    l.code: l
    for l in [
        Lang("eng", "English", flores="eng_Latn", sib="eng_Latn", news="eng", mmlu="eng"),
        Lang("yor", "Yoruba", "yor_Latn", "yor_Latn", "yor", "yor", "yor", "yor", "yor"),
        Lang("hau", "Hausa", "hau_Latn", "hau_Latn", "hau", "hau", "hau", "hau", "hau"),
        Lang("ibo", "Igbo", "ibo_Latn", "ibo_Latn", "ibo", "ibo", "ibo", "ibo", "ibo"),
        Lang("pcm", "Nigerian Pidgin", None, None, "pcm", "pcm", "pcm", None, "pcm"),
        Lang("twi", "Twi", "twi_Latn", "twi_Latn", None, "twi", "twi", "twi", "twi"),
        Lang("aka", "Akan", "aka_Latn", "aka_Latn"),
        Lang("ewe", "Ewe", "ewe_Latn", "ewe_Latn", None, None, "ewe", "ewe", None),
        Lang("fon", "Fon", "fon_Latn", "fon_Latn", None, None, "fon", None, None),
        Lang("fuv", "Fulfulde", "fuv_Latn", "fuv_Latn"),
        Lang("ful", "Fulah"),  # no standard benchmark distinct from fuv
        Lang("efi", "Efik"),  # no standard benchmark
        Lang("urh", "Urhobo"),  # no standard benchmark
    ]
}


def parse_langs(spec: str) -> list[str]:
    if spec == "all":
        return [c for c in LANGS if c != "eng"]
    codes = [c.strip() for c in spec.split(",") if c.strip()]
    unknown = [c for c in codes if c not in LANGS]
    if unknown:
        raise SystemExit(f"unknown language code(s) {unknown}; known: {list(LANGS)}")
    return codes
