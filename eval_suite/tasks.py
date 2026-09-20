"""Task runners. Each returns (metrics dict, per-example prediction rows)."""

from __future__ import annotations

from typing import Optional

from eval_suite import data, metrics, prompts
from eval_suite.langs import LANGS, Lang


def _rank(runner, style: str, contexts: list[str], labels: list[str], prefix: str) -> list[str]:
    conts = [runner.encode_continuation(prefix + l) for l in labels]
    preds = []
    for text in contexts:
        scores = runner.score(runner.encode(text, style), conts)
        preds.append(labels[max(range(len(labels)), key=scores.__getitem__)])
    return preds


def run_translation(runner, lang: Lang, style: str, limit, source: str, token, max_new_tokens: int,
                    num_beams: int, repetition_penalty: float, custom_path: Optional[str] = None,
                    africomet=None):
    pairs, src_name = data.load_translation(lang, limit, source, token, custom_path)
    eng = LANGS["eng"]
    out_metrics: dict = {"source": src_name}
    rows: list[dict] = []
    for direction, (s_lang, t_lang, s_key, t_key) in {
        f"eng->{lang.code}": (eng, lang, "eng", "xx"),
        f"{lang.code}->eng": (lang, eng, "xx", "eng"),
    }.items():
        ctxs = [runner.encode(prompts.translation(style, runner.tok, s_lang, t_lang, p[s_key]), style) for p in pairs]
        hyps = runner.generate(ctxs, max_new_tokens, num_beams, repetition_penalty)
        refs = [p[t_key] for p in pairs]
        m = metrics.chrf_bleu(hyps, refs)
        if africomet is not None:
            m["africomet"] = africomet([p[s_key] for p in pairs], hyps, refs)
        m["n"] = len(pairs)
        out_metrics[direction] = m
        rows += [{"direction": direction, "src": p[s_key], "ref": r, "hyp": h} for p, r, h in zip(pairs, refs, hyps)]
    return out_metrics, rows


def run_classification(runner, kind: str, lang: Lang, style: str, limit, topic_source: str, token):
    if kind == "topic":
        items, labels = data.load_topic(lang, topic_source, limit, token)
    else:
        items, labels = data.load_sentiment(lang, limit, token)
    contexts, prefix = [], ""
    for it in items:
        ctx, prefix = prompts.classification(style, runner.tok, kind, it["text"], labels)
        contexts.append(ctx)
    preds = _rank(runner, style, contexts, labels, prefix)
    gold = [it["label"] for it in items]
    m = metrics.classification_metrics(gold, preds, labels)
    return m, [{"text": it["text"], "gold": g, "pred": p} for it, g, p in zip(items, gold, preds)]


def parse_tags(text: str, n_tokens: int, valid: set[str]) -> list[str]:
    tags = [t if t in valid else "O" for t in text.split()][:n_tokens]
    return tags + ["O"] * (n_tokens - len(tags))


def run_ner(runner, lang: Lang, style: str, limit, token, max_new_tokens: int, num_beams: int, repetition_penalty: float):
    if style != "tag":
        raise data.DataUnavailable("NER is only defined for the tag (pretraining) prompt style")
    items, tag_names = data.load_ner(lang, limit, token)
    ctxs = [runner.encode(prompts.ner(it["tokens"]), style) for it in items]
    outs = runner.generate(ctxs, max_new_tokens, num_beams, repetition_penalty)
    valid = set(tag_names)
    preds = [parse_tags(o, len(it["tokens"]), valid) for o, it in zip(outs, items)]
    m = metrics.entity_f1([it["tags"] for it in items], preds)
    rows = [{"tokens": it["tokens"], "gold": it["tags"], "pred": p, "raw": o} for it, p, o in zip(items, preds, outs)]
    return m, rows


def run_mmlu(runner, lang: Lang, style: str, limit, token, shots: int):
    test, pool, name = data.load_mmlu(lang, limit, token)
    conts = [runner.encode_continuation(f" {l}") for l in "ABCD"]
    correct, rows = 0, []
    for r in test:
        demos = pool.get(r["subject"], [])[:shots]
        ctx = runner.encode(prompts.mmlu(r["subject"], r["question"], r["choices"], demos), "tag")
        scores = runner.score(ctx, conts)
        pred = max(range(4), key=scores.__getitem__)
        correct += pred == r["answer"]
        rows.append({"subject": r["subject"], "gold": "ABCD"[r["answer"]], "pred": "ABCD"[pred]})
    lo, hi = metrics.wilson_interval(correct, len(test))
    return {"accuracy": correct / len(test) if test else 0.0, "acc_ci95": [lo, hi], "chance": 0.25,
            "n": len(test), "shots": shots, "benchmark": name}, rows
