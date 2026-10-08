"""Who the user agent is playing.

A persona is here to make the USER side varied, because a corpus of conversations that all sound like the
same careful person asking a well-formed question teaches a model to expect one kind of user. Real messages
arrive terse, mid-thought, with the context missing, from someone doing something else at the time.

Each entry carries a register level, which decides how much English the user mixes in. That is sampled per
conversation rather than fixed, so code-switching appears as a property of the speaker -- which is what it
is -- rather than as a style the whole corpus shares.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    key: str
    description: str
    register: str          # "low" | "mixed" | "heavy" -- how much English gets mixed in
    weight: float = 1.0


PERSONAS: list[Persona] = [
    Persona("market_trader",
            "a market trader in her 40s, selling provisions. Texts in a hurry between customers, cares about "
            "prices, transfers that have not landed, and whether a thing is worth her money. Impatient with "
            "long answers",
            "mixed", 1.4),
    Persona("student",
            "a 19-year-old university student. Asks about coursework, exams and things they are embarrassed "
            "not to know. Types fast, abbreviates, mixes English in constantly",
            "heavy", 1.3),
    Persona("farmer",
            "a smallholder farmer in his 50s. Asks about rain, crops, fertiliser prices and animal health. "
            "Speaks the language properly and uses almost no English",
            "low", 1.2),
    Persona("small_business",
            "runs a small business with two staff. Asks about customers, pricing, registration, loans and "
            "writing messages they do not want to get wrong",
            "mixed", 1.3),
    Persona("civil_servant",
            "a mid-level civil servant. Asks about forms, entitlements, deadlines and what a letter means. "
            "Formal by habit, but texts in the language",
            "mixed", 1.0),
    Persona("nurse",
            "a nurse at a district clinic. Asks precise questions and will say when an answer is wrong. "
            "Mixes clinical English into the local language",
            "heavy", 0.9),
    Persona("okada_rider",
            "a motorcycle taxi rider in his 20s. Short messages, lots of slang, asks about routes, fuel, "
            "phone data and money",
            "heavy", 1.1),
    Persona("teacher",
            "a secondary school teacher. Asks for explanations they can pass on to a class, and for help "
            "writing things down",
            "mixed", 1.1),
    Persona("diaspora",
            "someone living abroad, sending money home and dealing with paperwork from a distance. Mixes "
            "more English than someone at home would",
            "heavy", 0.8),
    Persona("elder",
            "a 65-year-old. Asks plainly, in the language, with no English and no abbreviation. Wants the "
            "answer without ceremony",
            "low", 0.9),
]

_WEIGHTS = [p.weight for p in PERSONAS]


def pick(rng: random.Random) -> Persona:
    return rng.choices(PERSONAS, weights=_WEIGHTS, k=1)[0]
