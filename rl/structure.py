"""Parse a SabiYarn assistant turn into its scaffold fields and score the scaffold against the expected one.

A turn is   <|input_lang|><src>[<think>..</think>]<task_plan>TOKENS</task_plan><|target_lang|><tgt><response>CONTENT</s>

The scaffold (everything before CONTENT) is short, closed-vocabulary and has exactly one right answer, so it is scored by exact
field matching, not by an overlap metric such as BLEU: with 1-token fields a BLEU/chrF score is either 0 or 1 anyway, and its
n-gram machinery would only add noise (and reward getting the *order* of plan tokens right, which we do not want). What we score:

    input_lang   exact match of the language tag                                      (0/1)
    target_lang  exact match of the language tag                                      (0/1)
    plan         F1 of the multisets of tokens inside <task_plan>: ORDER-AGNOSTIC; partial credit for a partly right plan,
                 and extra tokens cost precision exactly as missing ones cost recall
    structure    0.6 * order + 0.2 * clean + 0.2 * closed, where
                   order   ORDER MATTERS between the landmarks <|input_lang|> < <task_plan> < </task_plan> < <|target_lang|> <
                           <response> (graded: longest correctly-ordered subsequence / 5, halved when a landmark repeats) --
                           only the order of the tokens between <task_plan> and </task_plan> is free
                   clean   non-empty content with no scaffold tokens leaking into it
                   closed  the turn ends with </s>

`format_score` is their weighted mean. Weights default to 0.20 / 0.30 / 0.30 / 0.20: the target-language tag decides what the content
must be, and the plan names the task, so those two carry the most.

The CONTENT is not scored here; rl/rewards.py scores it (AfriCOMET, with a chrF++ anchor) and only when the scaffold is right
enough for the content to be meaningful (see StructuredTranslationReward).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_EDGE = re.compile(r"^(?:\s|<s>|<pad>)+|(?:\s|</s>|<pad>)+$")
_TOKEN = re.compile(r"<[^<>]+>")
_LEAK = re.compile(r"<\|(?:input_lang|target_lang|user|assistant|system)\|>|</?task_plan>|</?think>|<response>|<s>|</s>")
# `plan_order` defaults to 0: the verb order is MEASURED and reported but earns nothing until there is
# evidence about what it is worth. Folding it in silently would make a correct-set-wrong-order plan score
# like a wrong-set one, which is the opposite of "not catching all the tasks matters more than the order".
DEFAULT_WEIGHTS = {"input_lang": 0.20, "plan": 0.30, "target_lang": 0.30, "structure": 0.20,
                   "plan_order": 0.0}


@dataclass
class Parsed:
    input_lang: str = ""
    target_lang: str = ""
    plan: list[str] = field(default_factory=list)
    think: str = ""
    response: str = ""
    has_response_tag: bool = False
    ordered: bool = False          # input_lang < (think) < task_plan < target_lang < response, each exactly once
    order: float = 0.0             # graded version of `ordered` in [0, 1]
    closed: bool = False           # ends with </s>
    n_input: int = 0
    n_target: int = 0
    n_plan: int = 0
    n_response: int = 0
    leak: bool = False             # scaffold tokens inside the content


def parse(text: str) -> Parsed:
    raw = text or ""
    p = Parsed(closed=raw.rstrip().endswith("</s>"))
    t = _EDGE.sub("", raw)
    p.n_input, p.n_target = t.count("<|input_lang|>"), t.count("<|target_lang|>")
    p.n_plan, p.n_response = t.count("<task_plan>"), t.count("<response>")
    i_in, i_pl = t.find("<|input_lang|>"), t.find("<task_plan>")
    i_pe, i_tg, i_rs = t.find("</task_plan>"), t.find("<|target_lang|>"), t.find("<response>")
    m = re.match(r"<\|input_lang\|>(<[^<>]+>)", t[i_in:]) if i_in >= 0 else None
    p.input_lang = m.group(1) if m else ""
    m = re.match(r"<\|target_lang\|>(<[^<>]+>)", t[i_tg:]) if i_tg >= 0 else None
    p.target_lang = m.group(1) if m else ""
    if i_pl >= 0 and i_pe > i_pl:
        p.plan = _TOKEN.findall(t[i_pl + len("<task_plan>"): i_pe])
    th = re.search(r"<think>(.*?)</think>", t, re.S)
    p.think = th.group(1) if th else ""
    if i_rs >= 0:
        p.has_response_tag = True
        p.response = t[i_rs + len("<response>"):].strip()
    p.leak = bool(_LEAK.search(p.response))
    idx = [i_in, i_pl, i_pe, i_tg, i_rs]
    p.ordered = all(i >= 0 for i in idx) and idx == sorted(idx) and (p.n_input, p.n_plan, p.n_target, p.n_response) == (1, 1, 1, 1)
    p.order = _order_score(idx, (p.n_input, p.n_plan, p.n_target, p.n_response))
    return p


def _order_score(idx: list[int], counts: tuple[int, int, int, int]) -> float:
    """Longest increasing subsequence of the landmark positions that were found, over the 5 expected, halved on repeats."""
    found = [i for i in idx if i >= 0]
    if not found:
        return 0.0
    best, tails = 0, []
    for x in found:                       # patience sorting: LIS length
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x
    best = len(tails)
    score = best / 5.0
    return score * 0.5 if any(c > 1 for c in counts) else score


# RECALL IS WEIGHTED ABOVE PRECISION, by beta squared. Missing a task and adding one are not the same
# mistake: a plan that omits <|RAG|> describes a turn that does not retrieve, while a plan with one verb too
# many is merely verbose. beta=2 weights recall 4x, which is the formal statement of "not catching all the
# tasks matters more than the order". beta=1 recovers plain F1 if that is ever wanted.
PLAN_BETA = 2.0


def plan_f1(pred: list[str], exp: list[str], beta: float = PLAN_BETA) -> float:
    """Multiset F-beta of two token lists, IGNORING ORDER. Recall-weighted by default -- see PLAN_BETA.

    Order-agnostic on purpose: the task_plan is a set of things the turn does, and a correct set in an odd
    order is far better than a wrong set in a tidy one. Order is measured separately by plan_order() so it
    can be reported and weighted in its own right rather than smuggled into this number.
    """
    if not pred and not exp:
        return 1.0
    if not pred or not exp:
        return 0.0
    common = sum((Counter(pred) & Counter(exp)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(exp)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def plan_order(pred: list[str], exp: list[str]) -> float | None:
    """Kendall tau-b over the verbs the two plans SHARE, mapped to 0..1. None when order is undefined.

    A sensible order exists -- declare what kind of question it is, retrieve, then answer -- and it is worth
    measuring even though it is worth little next to getting the set right. Tau is the right statistic
    because it is defined on RANKINGS rather than positions, so verbs the prediction adds or omits around
    the shared ones do not distort it: it answers "of the verbs both plans contain, are they in the same
    relative order", which is exactly the question.

    Returns None rather than 0.0 when fewer than two verbs are shared, because order is undefined for one
    element and scoring that as failure would punish a correct one-verb plan.
    """
    exp_set = set(exp)
    seen: set[str] = set()
    common = [v for v in pred if v in exp_set and not (v in seen or seen.add(v))]
    if len(common) < 2:
        return None
    rank = {v: i for i, v in enumerate(dict.fromkeys(v for v in exp if v in set(common)))}
    seq = [rank[v] for v in common if v in rank]
    if len(seq) < 2:
        return None
    concordant = discordant = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] < seq[j]:
                concordant += 1
            elif seq[i] > seq[j]:
                discordant += 1
    total = concordant + discordant
    if not total:
        return None
    return ((concordant - discordant) / total + 1.0) / 2.0


def structure_of(text: str) -> str:
    """The canonical scaffold string of an assistant turn, with the prose stripped.

    This is what data-gen writes into each RL record's `expected_structure` column -- the column
    StructuredTranslationReward already reads -- so the reward has a per-sample reference for ANY task
    rather than falling back to the translation-shaped default.
    """
    p = parse(text or "")
    out = []
    # parse() already returns these WITH their angle brackets ("<eng>"), so they are concatenated rather
    # than re-wrapped -- wrapping produced "<|input_lang|><<eng>>".
    if p.input_lang:
        out.append(f"<|input_lang|>{p.input_lang}")
    if p.plan:
        out.append("<task_plan>" + "".join(p.plan) + "</task_plan>")
    if p.target_lang:
        out.append(f"<|target_lang|>{p.target_lang}")
    if p.has_response_tag:
        out.append("<response>")
    return "".join(out)


def format_score(completion: str, expected: str, weights: dict[str, float] | None = None, allow_tag_in_content: bool | None = None) -> dict:
    """-> {"score", "input_lang", "target_lang", "plan", "structure", "gate", "response"}; every component in [0, 1]."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    pred, exp = parse(completion), parse(expected)
    # A component the REFERENCE does not specify is dropped and the weights renormalise -- it is not scored
    # as a failure. This matters most for tool-calling turns, which legitimately have no target language:
    # scoring the absent component as 0 while it still carried weight 0.30 meant a PERFECT tool turn scored
    # 0.572 against its own structure, systematically under-rewarding roughly 60% of the corpus. Measured on
    # live records, where chosen replies scored as low as 0.572 against structures extracted from themselves.
    s_in = float(pred.input_lang == exp.input_lang) if exp.input_lang else None
    s_tg = float(pred.target_lang == exp.target_lang) if exp.target_lang else None
    s_pl = plan_f1(pred.plan, exp.plan)
    s_or = plan_order(pred.plan, exp.plan)        # None when fewer than two verbs are shared
    allow_tag = ("<tag>" in exp.plan) if allow_tag_in_content is None else allow_tag_in_content
    leak = pred.leak and not (allow_tag and not re.search(r"<\|(?:input_lang|target_lang)\|>|</?task_plan>|<response>", pred.response))
    clean = float(bool(pred.response) and not leak)
    s_st = 0.6 * pred.order + 0.2 * clean + 0.2 * float(pred.closed)
    terms = {"plan": s_pl, "structure": s_st}
    if s_in is not None:
        terms["input_lang"] = s_in
    if s_tg is not None:
        terms["target_lang"] = s_tg
    if s_or is not None and w.get("plan_order"):
        terms["plan_order"] = s_or
    denom = sum(w[k] for k in terms) or 1.0
    total = sum(w[k] * v for k, v in terms.items()) / denom
    # A turn whose reference has no target language is a TOOL CALL: there is no content to gate, so the gate
    # is closed and the reward comes from the format term alone, which is correct -- there is no translation
    # to score.
    gate = bool(s_tg and pred.has_response_tag and pred.response and not leak)
    # The order term is REPORTED but carries no weight in `score` by default: it is worth knowing and worth
    # little, and folding it in silently would make a correct-set-wrong-order plan look like a wrong-set one.
    # DEFAULT_WEIGHTS["plan_order"] can be raised to give it weight once there is evidence about what it is
    # worth; until then it is an observation, not a reward.
    return {"score": total, "input_lang": s_in, "target_lang": s_tg, "plan": s_pl, "structure": s_st,
            "plan_order": s_or, "gate": gate, "response": pred.response,
            "scored_terms": sorted(terms)}


def shape_reward(r: float, threshold: float = 0.55, negative_scale: float = 2.0) -> float:
    """Piecewise-linear, order-preserving reshaping of a [0, 1] reward around `threshold`.

    r >= threshold  ->  (r - threshold) / (1 - threshold)          in [0, 1]    (positive)
    r <  threshold  ->  -negative_scale * (threshold - r) / threshold  in [-negative_scale, 0)  (large negative)

    Within a group it does not change which completion is best; it moves the zero point to `threshold` and widens the gap between
    an acceptable and an unacceptable completion, so a bad sample is pushed down hard instead of merely ranking a little lower.
    The trainer still standardises advantages inside each group, which rescales this but keeps its ordering and sign structure.
    """
    if r >= threshold:
        return (r - threshold) / max(1e-9, 1.0 - threshold)
    return -negative_scale * (threshold - r) / max(1e-9, threshold)


def expected_translation_structure(src: str, tgt: str) -> str:
    return f"<|input_lang|><{src}><task_plan><translate></task_plan><|target_lang|><{tgt}><response>"
