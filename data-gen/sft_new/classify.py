"""Supervised-task SFT rows built from human-annotated, commercially usable datasets (vetted: licence, date rule, no benchmark test
splits). Every turn is assembled with data-gen/assemble.py, so the scaffolding is exact.

  intent       <classify> + <intent>label          Injongo (8 languages), MASSIVE af/sw, WolBanking77 (Wolof)
  slots (JSON) <|data_extract|><|JSON|>            Injongo and MASSIVE slot annotations -> a JSON object, domain flag from keywords
  hate         <classify> + <|hate|>label           AfriHate (som orm swh xho zul kin)
  toxic        <classify> + <toxic>label            Swahili and Setswana offensive-language sets
  sentiment    <classify> + <sentiment>label        tabularisai Swahili (LLM text, gated on its judge-score columns)
  topic        <classify> + <topic>label            news sets: Swahili, Setswana, Kinyarwanda, Zulu
  ner          <NER> + <tag>                        NCHLT (afr tsn xho zul), word <tag> LABEL pairs, domain flag from keywords
  pos          <NER> + <tag>                        Universal Dependencies (afr wol pcm), word <tag> UPOS pairs
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sft_new import structured as S
from sft_new.domain import domain_flag

A = S.A
A.LABEL_TOKENS.update({"hate": "<|hate|>", "toxic": "<toxic>", "sentiment": "<sentiment>", "topic": "<topic>", "intent": "<intent>"})
A.VALID_LANGS |= {"swh"}
SCHEMA = pa.schema([("instruction", pa.string()), ("input", pa.string()), ("response", pa.string()), ("language", pa.string()),
                    ("task", pa.string()), ("messages", pa.string()), ("source_dataset", pa.string())])
CODE = {"swa": "swh", "sw": "swh", "kinyarwanda": "kin"}


def mk(task, instruction, text, response, lang, assistant, dataset):
    user = f"{instruction}\n\n{text}" if text else instruction
    return {"instruction": instruction, "input": text, "response": response, "language": lang, "task": task,
            "messages": json.dumps([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], ensure_ascii=False),
            "source_dataset": dataset}


def classify_row(task, label_key, instr_pool, text, label, lang, dataset, rng, options=None):
    instr = rng.choice(instr_pool)
    if options:
        instr += " Options: " + ", ".join(options) + "."
    try:
        reply = S.turn(lang, "eng", ["<classify>"], str(label), label=label_key)
    except Exception:
        return None
    return mk(task, instr, text, str(label), lang, reply, dataset)


# ------------------------------------------------------------------------------------------------ helpers
def lds(repo, config=None, split="train", **kw):
    from datasets import load_dataset
    return load_dataset(repo, config, split=split, **kw)


def convert_parquet(repo, path):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset", revision="refs/convert/parquet")


def class_names(parquet_path, column):
    meta = pq.read_schema(parquet_path).metadata or {}
    feats = json.loads(meta.get(b"huggingface", b"{}")).get("info", {}).get("features", {})
    f = feats.get(column, {})
    if "names" in f:
        return f["names"]
    inner = f.get("feature", {})
    return inner.get("names")


INTENT_INSTR = ["What is the intent of this message?", "Classify the intent of the text below.", "Identify what the user wants in this message.",
                "Which intent does this request express?"]
JSON_INSTR = ["Extract the entities from this text as JSON, using the slot names as keys and lists of values.",
              "Return a JSON object with the slots mentioned in the text (slot name -> values). Output only the JSON.",
              "Pull the key details out of this request as structured JSON."]
HATE_INSTR = ["Is this text hate speech, abusive, or normal?", "Classify this post as Hate, Abuse or Normal.", "Label the following text for hate speech."]
TOX_INSTR = ["Is this text offensive?", "Does the following text contain offensive language? Answer with the label only.", "Flag whether this text is offensive."]
SENT_INSTR = ["What is the sentiment of this text?", "Classify the sentiment of the text below.", "Give the sentiment label for this text."]
TOPIC_INSTR = ["What topic is this article about?", "Assign a topic label to this text.", "Which category does this news text belong to?"]
NER_INSTR = ["Tag the named entities in this sentence. Write each word followed by <tag> and its label (PERSON, ORGANIZATION, LOCATION, MISC or O).",
             "Do named-entity recognition on the sentence below, giving word <tag> LABEL pairs."]
POS_INSTR = ["Give the part-of-speech tag of every word in this sentence as word <tag> TAG pairs.",
             "POS-tag the following sentence. Write each word followed by <tag> and its tag."]


def slots_json(text, spans):
    b = text.encode("utf-8")
    out = collections.OrderedDict()
    for sp in spans or []:
        try:
            val = b[sp["start_byte"]: sp["limit_byte"]].decode("utf-8")
        except Exception:
            continue
        out.setdefault(sp["label"], []).append(val)
    return out


# ------------------------------------------------------------------------------------------------ sources
def injongo(rng, stats):
    langs = {"kin": "kin", "orm": "orm", "sna": "sna", "sot": "sot", "swa": "swh", "wol": "wol", "xho": "xho", "zul": "zul"}
    for cfg, code in langs.items():
        for sp in ("train", "validation"):                                  # test stays untouched
            for r in lds("masakhane/InjongoIntent", cfg, sp):
                t = (r["text"] or "").strip()
                if len(t) < 6:
                    continue
                row = classify_row("intent", "intent", INTENT_INSTR, t, r["intent"], code, "masakhane/InjongoIntent", rng)
                if row:
                    stats["intent_injongo"] += 1; yield row
                sl = slots_json(t, r.get("spans"))
                if sl:
                    js = json.dumps(sl, ensure_ascii=False)
                    try:
                        reply = S.turn(code, "eng", ["<|data_extract|>", "<|JSON|>"], js, flag=domain_flag(t))
                    except Exception:
                        continue
                    stats["slots_injongo"] += 1
                    yield mk("structured_output", rng.choice(JSON_INSTR), t, js, code, reply, "masakhane/InjongoIntent")


def massive(rng, stats):
    for loc, code in (("af-ZA", "afr"), ("sw-KE", "swh")):
        for sp in ("train", "validation"):
            p = convert_parquet("AmazonScience/massive", f"{loc}/{sp}/0000.parquet")
            names = class_names(p, "intent")
            tb = pq.read_table(p, columns=["utt", "annot_utt", "intent"]).to_pylist()
            for r in tb:
                t = (r["utt"] or "").strip()
                if len(t) < 4:
                    continue
                lab = names[r["intent"]] if names else str(r["intent"])
                row = classify_row("intent", "intent", INTENT_INSTR, t, lab, code, "AmazonScience/massive", rng)
                if row:
                    stats["intent_massive"] += 1; yield row
                sl = collections.OrderedDict()
                for name, val in re.findall(r"\[([a-z_]+) : ([^\]]+)\]", r["annot_utt"] or ""):
                    sl.setdefault(name, []).append(val.strip())
                if sl:
                    js = json.dumps(sl, ensure_ascii=False)
                    try:
                        reply = S.turn(code, "eng", ["<|data_extract|>", "<|JSON|>"], js, flag=domain_flag(t))
                    except Exception:
                        continue
                    stats["slots_massive"] += 1
                    yield mk("structured_output", rng.choice(JSON_INSTR), t, js, code, reply, "AmazonScience/massive")


def wolbanking(rng, stats):
    for r in lds("karim155/WolBanking77", None, "train"):
        t = (r.get("input_wo") or "").strip()
        if len(t) >= 6:
            row = classify_row("intent", "intent", INTENT_INSTR, t, r["label"], "wol", "karim155/WolBanking77", rng)
            if row:
                stats["intent_wolbanking"] += 1; yield row


def afrihate(rng, stats):
    for cfg in ("som", "orm", "swa", "xho", "zul", "kin"):
        code = CODE.get(cfg, cfg)
        for sp in ("train", "validation"):
            for r in lds("afrihate/afrihate", cfg, sp):
                t = (r["tweet"] or "").strip()
                if len(t) >= 8 and r["label"] in ("Hate", "Abuse", "Normal"):
                    row = classify_row("hate", "hate", HATE_INSTR, t, r["label"], code, "afrihate/afrihate", rng, ["Hate", "Abuse", "Normal"])
                    if row:
                        stats["hate_afrihate"] += 1; yield row


def offensive(rng, stats):
    for sp in ("train", "validation"):
        for r in lds("metabloit/offensive-swahili-text", None, sp):
            t = (r["text"] or "").strip()
            lab = "offensive" if int(r["labels"]) == 1 else "not offensive"
            row = classify_row("toxic", "toxic", TOX_INSTR, t, lab, "swh", "metabloit/offensive-swahili-text", rng, ["offensive", "not offensive"]) if len(t) >= 8 else None
            if row:
                stats["toxic_swahili"] += 1; yield row
    import csv
    from huggingface_hub import hf_hub_download
    for fn in ("training_data-tag-free.csv", "testing_data-tag-free.csv"):
        try:
            p = hf_hub_download("mopatik/setswana-offensive-977", fn, repo_type="dataset")
        except Exception:
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            t = (r.get("TEXT") or "").strip()
            lab = "offensive" if str(r.get("TARGET")).strip().lower() in ("1", "offensive", "true") else "not offensive"
            row = classify_row("toxic", "toxic", TOX_INSTR, t, lab, "tsn", "mopatik/setswana-offensive-977", rng, ["offensive", "not offensive"]) if len(t) >= 8 else None
            if row:
                stats["toxic_setswana"] += 1; yield row


def sentiment(rng, stats):
    ds = lds("tabularisai/swahili-sentiment-dataset", None, "train")
    cols = [c for c in ds.column_names if c.lower() in ("overall_quality", "language_correctness", "sentiment_alignment", "cultural_relevance")]
    for r in ds:
        t = (r["text"] or "").strip()
        if len(t) < 8:
            continue
        try:
            ok = all(float(r[c]) >= 4 for c in cols)       # keep only rows the generator's own judge scored 4/5 or better
        except (TypeError, ValueError):
            ok = False
        if not ok:
            stats["sentiment_gated_out"] += 1
            continue
        lab = re.sub(r"^[\d.]+\s*-\s*", "", str(r["sentiment"])).strip()
        row = classify_row("sentiment", "sentiment", SENT_INSTR, t, lab, "swh", "tabularisai/swahili-sentiment-dataset", rng)
        if row:
            stats["sentiment_swahili"] += 1; yield row


def topics(rng, stats):
    ds = lds("community-datasets/swahili_news", None, "train")
    names = ds.features["label"].names
    for r in ds:
        t = (r["text"] or "").strip()
        if 60 <= len(t) <= 2500:
            row = classify_row("topic", "topic", TOPIC_INSTR, t, names[r["label"]], "swh", "community-datasets/swahili_news", rng, names)
            if row:
                stats["topic_swahili"] += 1; yield row
    for sp in ("train", "validation", "test"):
        try:
            d = lds("dsfsi/daily-news-dikgang", None, sp)
        except Exception:
            continue
        for r in d:
            t = ((r.get("title") or "") + "\n" + (r.get("text") or "")).strip()
            if 60 <= len(t) <= 3500 and r.get("orig_label"):
                row = classify_row("topic", "topic", TOPIC_INSTR, t, r["orig_label"], "tsn", "dsfsi/daily-news-dikgang", rng)
                if row:
                    stats["topic_setswana"] += 1; yield row
    p = convert_parquet("andreniyongabo/kinnews_kirnews", "kinnews_cleaned/train/0000.parquet")
    names = class_names(p, "label")
    for r in pq.read_table(p, columns=["label", "title", "content"]).to_pylist():
        t = ((r["title"] or "") + "\n" + (r["content"] or "")).strip()
        if 60 <= len(t) <= 3500:
            lab = names[r["label"]] if names else str(r["label"])
            row = classify_row("topic", "topic", TOPIC_INSTR, t, lab, "kin", "andreniyongabo/kinnews_kirnews", rng)
            if row:
                stats["topic_kinyarwanda"] += 1; yield row
    d = lds("mteb/IsiZuluNewsClassification", None, "train")
    for r in d:
        t = (r.get("text") or "").strip()
        lab = r.get("label_text") or r.get("label")
        if 40 <= len(t) <= 3500 and lab is not None:
            row = classify_row("topic", "topic", TOPIC_INSTR, t, lab, "zul", "mteb/IsiZuluNewsClassification", rng)
            if row:
                stats["topic_zulu"] += 1; yield row


def tagged(task, instr_pool, tokens, tags, lang, dataset, rng, stats, flag):
    pairs = " ".join(f"{w} <tag> {t}" for w, t in zip(tokens, tags))
    sent = " ".join(tokens)
    try:
        reply = S.turn(lang, "eng", ["<NER>"], pairs, flag=flag) if False else None
    except Exception:
        reply = None
    try:
        content = A.assistant_content(input_lang=lang, target_lang=lang, think=None, task_plan=([flag] if flag else []) + ["<NER>"],
                                      response=pairs, sequence_labels=True)
    except Exception:
        stats["assemble_rejected"] += 1
        return None
    return mk(task, rng.choice(instr_pool), sent, pairs, lang, content, dataset)


def ner(rng, stats):
    lab = {"PERS": "PERSON", "PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION", "MISC": "MISC", "O": "O"}
    for cfg, code in (("af", "afr"), ("tn", "tsn"), ("xh", "xho"), ("zu", "zul")):
        p = convert_parquet("nwu-ctext/nchlt", f"{cfg}/train/0000.parquet")
        names = class_names(p, "ner_tags")
        for r in pq.read_table(p, columns=["tokens", "ner_tags"]).to_pylist():
            toks = r["tokens"]
            if not (4 <= len(toks) <= 60):
                continue
            tags = [lab.get(re.sub(r"^[BI]-", "", names[i]), "MISC") for i in r["ner_tags"]]
            if all(t == "O" for t in tags):
                continue
            row = tagged("ner", NER_INSTR, toks, tags, code, "nwu-ctext/nchlt", rng, stats, domain_flag(" ".join(toks)))
            if row:
                stats["ner_nchlt"] += 1; yield row


def pos(rng, stats):
    for cfg, code in (("af_afribooms", "afr"), ("wo_wtb", "wol"), ("pcm_nsc", "pcm")):
        try:
            d = lds("universal-dependencies/universal_dependencies", cfg, "train")
        except Exception as e:
            print("pos", cfg, "failed", e, file=sys.stderr); continue
        f = d.features["upos"].feature
        names = getattr(f, "names", None)
        for r in d:
            toks = r["tokens"]
            if not (4 <= len(toks) <= 60):
                continue
            row = tagged("pos", POS_INSTR, toks, [names[i] if names else i for i in r["upos"]], code, "universal-dependencies/universal_dependencies", rng, stats, None)
            if row:
                stats["pos_ud"] += 1; yield row


SOURCES = {"injongo": injongo, "massive": massive, "wolbanking": wolbanking, "afrihate": afrihate, "offensive": offensive,
           "sentiment": sentiment, "topics": topics, "ner": ner, "pos": pos}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop each source after N rows")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(S.HERE))
    from africomet_translation.build import hf_token
    token = hf_token()
    import os
    os.environ["HF_TOKEN"] = token
    out = Path(a.out)
    o = S.Out(out, "structured-tasks", schema=SCHEMA)
    stats = collections.defaultdict(int)
    rng = random.Random("classify")
    for name, fn in SOURCES.items():
        if a.only and name not in a.only:
            continue
        n = 0
        try:
            for r in fn(rng, stats):
                o.add(r); n += 1
                if a.limit and n >= a.limit:
                    break
        except Exception as e:                                           # one broken source must not stop the others
            print(f"[{name}] stopped: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr, flush=True)
        print(f"[{name}] {n} rows", file=sys.stderr, flush=True)
    o.flush()
    (out / "stats").mkdir(parents=True, exist_ok=True)
    (out / "stats" / "structured-tasks.json").write_text(json.dumps(dict(stats), indent=1))
    print(dict(stats), o.total, "rows", file=sys.stderr)
    if a.push and o.total:
        S.push(out, "structured-tasks", token)


if __name__ == "__main__":
    main()
