"""Persona / style sampling to keep generated conversations from all
sounding like the same person. Each generator samples one persona per
example and injects it into the meta-prompt sent to the generation model.
"""

from __future__ import annotations

import random

AGE_OCCUPATION_BANK: list[str] = [
    "a 19-year-old university student",
    "a 34-year-old market trader",
    "a 45-year-old primary school teacher",
    "a 28-year-old commercial okada/keke rider",
    "a 52-year-old smallholder farmer",
    "a 23-year-old mobile phone repairer",
    "a 60-year-old retired civil servant",
    "a 31-year-old nurse at a local clinic",
    "a 17-year-old secondary school student",
    "a 39-year-old tailor/fashion designer",
    "a 26-year-old small business owner running a provisions store",
    "a 41-year-old commercial driver",
    "a 22-year-old NYSC corps member",
    "a 48-year-old pastor/imam's assistant handling church/mosque admin",
    "a 35-year-old salon/barbershop owner",
    "a 29-year-old delivery rider for an e-commerce app",
    "a 55-year-old retired trader now farming part-time",
    "a 20-year-old apprentice electrician",
]

TONE_BANK: list[str] = [
    "busy and writes short, direct messages",
    "polite and a bit formal, uses full sentences",
    "casual and relaxed, sometimes uses local slang",
    "anxious/urgent about the situation and writes quickly",
    "curious and asks follow-up questions",
    "skeptical and wants to double-check the answer",
    "friendly and chatty, adds small talk before the real question",
    "terse, almost like typing on a low-end phone with limited data",
    "frustrated because a previous attempt at something failed",
    "calm and methodical, lists things out clearly",
]

QUIRK_BANK: list[str] = [
    "occasionally includes a minor typo or dropped word, but stays understandable",
    "sometimes code-switches a single word into English mid-sentence",
    "uses common everyday phrasing rather than textbook-formal language",
    "avoids punctuation at the end of sentences, like real chat messages",
    "writes in full, carefully punctuated sentences",
    "uses a local greeting or expression naturally where it fits",
]


def sample_persona(rng: random.Random) -> str:
    age_occ = rng.choice(AGE_OCCUPATION_BANK)
    tone = rng.choice(TONE_BANK)
    quirk = rng.choice(QUIRK_BANK)
    return (
        f"Simulate the user as {age_occ} who is {tone}. Their writing style "
        f"{quirk}. Keep this persona consistent across all of their turns "
        f"in the conversation."
    )
