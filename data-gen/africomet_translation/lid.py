"""GlotLID (fastText) language identification used to verify web-scraped sentences.

Measured on HPLT 2.0 "Yoruba": only ~46% of lines the corpus itself labels yor_Latn are Yoruba according to GlotLID
(the rest are Budu / Gun / Gokana Bible text and similar), so web sources are only used through this filter.
Each worker process holds its own ~1.6 GB copy of the model, so keep the number of LID workers small.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

# Accepted GlotLID labels per language. Twi and Akan are one continuum; GlotLID separates them.
EXPECTED = {
    "yor": {"yor_Latn"}, "ibo": {"ibo_Latn"}, "hau": {"hau_Latn"},
    # GlotLID labels Nigerian Pidgin as wes_Latn (Cameroonian Pidgin) about as often as pcm_Latn, and calls Ibibio "efi".
    "pcm": {"pcm_Latn", "wes_Latn"}, "efi": {"efi_Latn", "ibb_Latn"}, "ibb": {"efi_Latn", "ibb_Latn"},
    "fon": {"fon_Latn"}, "ewe": {"ewe_Latn"}, "urh": {"urh_Latn"},
    "twi": {"twi_Latn", "aka_Latn"}, "eng": {"eng_Latn"},
    # languages added for the pretrain collection
    "kin": {"kin_Latn"}, "sna": {"sna_Latn"}, "zul": {"zul_Latn"}, "swh": {"swh_Latn"}, "orm": {"gaz_Latn", "orm_Latn"},
    "wol": {"wol_Latn"}, "xho": {"xho_Latn"}, "tsn": {"tsn_Latn"}, "nya": {"nya_Latn"}, "afr": {"afr_Latn"},
    "som": {"som_Latn"}, "sot": {"sot_Latn"},
}

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import fasttext
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("cis-lmu/glotlid", "model_v3.bin", token=os.environ.get("HF_TOKEN"))
        _MODEL = fasttext.load_model(path)
    return _MODEL


def top_labels(sentences: list[str]) -> list[str]:
    """Top-1 GlotLID label for each sentence (batched; this build exposes labels only, ~3k sentences/s)."""
    if not sentences:
        return []
    raw = _model().f.multilinePredict([s.replace("\n", " ") for s in sentences], 1, 0.0, "replace")
    return [r[0].replace("__label__", "") if r else "und" for r in raw]


_PCM_MARK = re.compile(r"\b(dey|wey|wetin|abeg|sabi|pikin|oga|dia|dem|na)\b", re.I)


def verify(sentences: list[str], lang: str) -> list[bool]:
    ok = EXPECTED[lang]
    labels = top_labels(sentences)
    if lang == "pcm":      # Pidgin is often labelled English; keep those only when they contain Pidgin function words
        return [l in ok or (l == "eng_Latn" and bool(_PCM_MARK.search(s))) for s, l in zip(sentences, labels)]
    return [l in ok for l in labels]
