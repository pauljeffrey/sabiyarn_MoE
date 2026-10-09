"""Structured SFT turns that use the corpus's full tag grammar, built only from real data (no LLM writes anything):

  <|input_lang|><src>[<think>..</think>]<task_plan><verb></task_plan><|target_lang|><tgt><response>[label]TEXT

Tasks
  identify   language identification from our own pretrain text; the <think> cites cue words that really are
             distinctive for that language in the corpus (log-odds vs the other languages), so it is a verified fact.
  diacritize restore tone marks / underdots / hooked letters; the input is the real text with them removed.
  chat/qa    CohereLabs/aya_dataset, human-written prompts and completions (Apache-2.0), new-language rows only.
  summarize  bltlab/lr-sum (CC-BY-4.0): VOA article -> reference summary, Kinyarwanda and Shona.

The turns are assembled by data-gen/assemble.py (`assistant_content`), the same builder the SFT generator uses, so a
turn that is malformed raises instead of being written.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]
REPO = "BeardedMonster/sft-new"
PRETRAIN = "BeardedMonster/pretrain-new"

NAMES = {"eng": "English", "afr": "Afrikaans", "zul": "Zulu", "xho": "Xhosa", "sna": "Shona", "nya": "Chewa", "sot": "Sesotho",
         "kin": "Kinyarwanda", "tsn": "Setswana", "orm": "Oromo", "wol": "Wolof", "som": "Somali", "swh": "Swahili", "hau": "Hausa",
         "ibo": "Igbo", "yor": "Yoruba", "twi": "Twi", "ewe": "Ewe", "fon": "Fon", "efi": "Efik", "ibb": "Ibibio", "urh": "Urhobo",
         "pcm": "Nigerian Pidgin"}
SCHEMA = pa.schema([("instruction", pa.string()), ("input", pa.string()), ("response", pa.string()),
                    ("source_lang", pa.string()), ("targ_lang", pa.string()), ("task", pa.string()),
                    ("messages", pa.string()), ("source_dataset", pa.string())])


def _assemble():
    """Load data-gen/assemble.py under its own name (africomet_translation has an unrelated assemble module)."""
    spec = importlib.util.spec_from_file_location("sft_assemble", HERE / "assemble.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["sft_assemble"] = m
    spec.loader.exec_module(m)
    m.VALID_LANGS |= set(NAMES) - {"eng"} | {"eng"}          # every tag below exists in the tokenizer
    return m


A = _assemble()
DOMAIN_FLAGS = ["<|is_technical|>", "<|is_legal|>", "<|is_medical|>", "<|is_scientific|>", "<|is_non_technical|>", "<|is_financial|>"]
A.VALID_VERBS |= set(DOMAIN_FLAGS)          # document-domain flags ride in the plan, before the task verb
A.LABEL_TOKENS.update({"score": "<score>", "correct": "<correct>", "reason": "<reason>"})      # all three exist in the tokenizer


def turn(src: str, tgt: str, verbs: list[str], response: str, think: str | None = None, label: str | None = None,
         flag: str | None = None) -> str:
    plan = ([flag] if flag else []) + verbs
    return A.assistant_content(input_lang=src, target_lang=tgt, think=think, task_plan=plan, response=response, label_token=label)


def row(task, instruction, text, response, src, tgt, assistant, dataset) -> dict:
    user = f"{instruction}\n\n{text}" if text else instruction
    return {"instruction": instruction, "input": text, "response": response, "source_lang": src, "targ_lang": tgt, "task": task,
            "messages": json.dumps([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], ensure_ascii=False),
            "source_dataset": dataset}


class Out:
    def __init__(self, out: Path, name: str, per=100_000, schema=None):
        self.schema = schema or SCHEMA
        self.dir = out / "data" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows, self.n, self.per, self.total = [], 0, per, 0

    def add(self, r):
        self.rows.append(r)
        self.total += 1
        if len(self.rows) >= self.per:
            self.flush()

    def flush(self):
        if self.rows:
            pq.write_table(pa.table({k: [r[k] for r in self.rows] for k in self.schema.names}, schema=self.schema),
                           self.dir / f"part-{self.n:05d}.parquet", compression="zstd")
            self.n += 1
            self.rows = []


def push(out: Path, name: str, token: str):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(out / "data" / name), path_in_repo=f"data/{name}",
                      commit_message=f"{name}")
    import shutil
    shutil.rmtree(out / "data" / name, ignore_errors=True)


# --------------------------------------------------------------------------------------------- pretrain text access
def pretrain_docs(lang: str, token: str, limit: int, skip: int = 0):
    """Yield documents of one language from the Hub copy of pretrain-new, reading only the text column."""
    from huggingface_hub import HfApi, HfFileSystem
    files = sorted(f.path for f in HfApi(token=token).list_repo_tree(PRETRAIN, path_in_repo=f"data/{lang}", repo_type="dataset")
                   if f.path.endswith(".parquet"))
    fs = HfFileSystem(token=token)
    seen = 0
    for p in files:
        with fs.open(f"datasets/{PRETRAIN}/{p}", "rb") as fh:
            for b in pq.ParquetFile(fh).iter_batches(batch_size=2000, columns=["text"]):
                for t in b.column(0).to_pylist():
                    seen += 1
                    if seen <= skip:
                        continue
                    yield t
                    if seen - skip >= limit:
                        return


_W = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.U)


def words(text: str) -> list[str]:
    return [w.lower() for w in _W.findall(text)]


# --------------------------------------------------------------------------------------------- identify
ID_INSTR = ["Which language is this text written in?", "Identify the language of the following text.",
            "What language is the passage below?", "Tell me which language this is.", "Name the language of this snippet.",
            "Language ID: what language is this?", "Which language am I looking at here?", "Detect the language of this text."]


def build_cues(langs: list[str], token: str, n_docs: int = 8000, top: int = 400) -> dict[str, list[str]]:
    counts = {}
    for l in langs:
        c = collections.Counter()
        for t in pretrain_docs(l, token, n_docs):
            c.update(words(t[:6000]))
        counts[l] = c
        print("cues: counted", l, len(c), file=sys.stderr, flush=True)
    total = collections.Counter()
    for c in counts.values():
        total.update(c)
    cues = {}
    for l, c in counts.items():
        n_l = sum(c.values()); n_o = sum(total.values()) - n_l
        scored = []
        for w, k in c.items():
            if k < 25 or len(w) < 3:
                continue
            o = total[w] - k
            ratio = math.log((k / n_l) / ((o + 0.5) / n_o))
            if ratio >= 2.5:
                scored.append((k, w))
        cues[l] = [w for _, w in sorted(scored, reverse=True)[:top]]
    return cues


def identify_rows(l: str, cues: dict[str, list[str]], token: str, n: int, rng: random.Random):
    from africomet_translation import lid
    from africomet_translation.text_utils import good_sentence, normalize, split_sentences
    cue_set = set(cues[l])
    cand = []
    for t in pretrain_docs(l, token, 20000):
        sents = [normalize(s) for s in split_sentences(t)]
        for i in range(0, max(0, len(sents) - 1), 3):
            for k in (1, 2):
                sn = " ".join(sents[i:i + k])
                nw = len(sn.split())
                if 8 <= nw <= 45 and all(good_sentence(x, min_words=3, max_words=60, require_terminal=False, lenient=True) for x in sents[i:i + k]):
                    found = []
                    for w in words(sn):
                        if w in cue_set and w not in found:
                            found.append(w)
                    if len(found) >= 2:
                        cand.append((sn, found)); break
        if len(cand) >= n * 3:
            break
    rng.shuffle(cand)
    cand = cand[: n * 2]
    ok = lid.verify([c[0] for c in cand], l) if cand else []
    kept = [c for c, g in zip(cand, ok) if g][:n]
    for sn, found in kept:
        cue = ", ".join(f"“{w}”" for w in found[:4])
        think = (f"The snippet is not English. Words like {cue} are characteristic of {NAMES[l]}, so this is {NAMES[l]}.")
        reply = turn(l, "eng", ["<identify>"], f"{NAMES[l]} ({l})", think=think, label="lang_id")
        yield row("identify", rng.choice(ID_INSTR), sn, f"{NAMES[l]} ({l})", l, "eng", reply, "BeardedMonster/pretrain-new")


# --------------------------------------------------------------------------------------------- diacritize
DI_INSTR = {"yor": ["Add the missing tone marks and underdots to this Yoruba text.", "Restore the diacritics in the Yoruba sentence below.",
                    "This Yoruba text has lost its tone marks and underdots. Write it out correctly."],
            "ibo": ["Add the missing tone marks and dot-below letters to this Igbo text.", "Restore the diacritics in this Igbo text."],
            "ewe": ["Add the missing tone marks to this Ewe text.", "Restore the diacritics in the Ewe text below."],
            "fon": ["Add the missing tone marks to this Fon text.", "Restore the diacritics in this Fon text."],
            "hau": ["Restore the hooked letters (ɓ, ɗ, ƙ) in this Hausa text.", "This Hausa text was typed without the hooked letters. Write it with the correct letters."]}
HAU_MAP = str.maketrans({"ɓ": "b", "ɗ": "d", "ƙ": "k", "Ɓ": "B", "Ɗ": "D", "Ƙ": "K"})


def strip_marks(t: str, lang: str) -> str:
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    if lang == "hau":
        t = t.translate(HAU_MAP)
    return unicodedata.normalize("NFC", t)


def diacritize_rows(l: str, token: str, n: int, rng: random.Random):
    from africomet_translation.text_utils import good_sentence, normalize, split_sentences
    got = 0
    for t in pretrain_docs(l, token, 60000):
        sents = [normalize(s) for s in split_sentences(t)]
        for i in range(0, max(0, len(sents)), 4):
            sn = " ".join(sents[i:i + rng.choice((1, 2))])
            nw = len(sn.split())
            if not 5 <= nw <= 60 or not good_sentence(sn, min_words=3, max_words=70, require_terminal=False, lenient=True):
                continue
            plain = strip_marks(sn, l)
            diffs = sum(a != b for a, b in zip(unicodedata.normalize("NFC", sn), plain)) + abs(len(sn) - len(plain))
            if diffs < 4 or plain == sn:
                continue
            reply = turn(l, l, ["<diacritize>"], sn)
            yield row("diacritize", rng.choice(DI_INSTR[l]), plain, sn, l, l, reply, "BeardedMonster/pretrain-new")
            got += 1
            if got >= n:
                return


# --------------------------------------------------------------------------------------------- aya_dataset
AYA_LANGS = {"zul", "xho", "sna", "nya", "sot", "tsn", "kin", "orm", "wol", "som", "swh", "afr"}


def aya_rows(token: str, rng: random.Random, stats):
    from huggingface_hub import hf_hub_download
    from africomet_translation import lid
    per_lang = collections.Counter()
    for sp in ("train", "test"):
        p = hf_hub_download("CohereLabs/aya_dataset", f"data/{sp}-00000-of-00001.parquet", repo_type="dataset", token=token)
        tb = pq.read_table(p, columns=["inputs", "targets", "language_code", "annotation_type"]).to_pylist()
        for r in tb:
            l = r["language_code"]
            if l not in AYA_LANGS:
                continue
            q, a = (r["inputs"] or "").strip(), (r["targets"] or "").strip()
            stats["seen"] += 1
            if per_lang[l] >= 3000:                       # Somali alone is half of aya's rows for these languages
                stats["capped"] += 1
                continue
            if len(q) < 8 or len(a) < 8 or len(a) > 3500 or len(q) > 3500:
                stats["short_or_long"] += 1
                continue
            if "<" in q + a and re.search(r"<\|?\w+\|?>", q + a):
                stats["markup"] += 1
                continue
            verb = "<qa>" if (q.rstrip().endswith("?") and len(q) < 300) else "<|chat|>"
            try:
                reply = turn(l, l, [verb], a)
            except Exception:
                stats["assemble_rejected"] += 1
                continue
            per_lang[l] += 1
            yield row("chat" if verb == "<|chat|>" else "qa", q, "", a, l, l, reply, "CohereLabs/aya_dataset")


# --------------------------------------------------------------------------------------------- translation scoring
SC_INSTR = ["Rate this translation from {a} into {b} on a scale of 1 to 10.", "How good is this {a} to {b} translation? Give a score from 1 to 10.",
            "Score the quality of the {b} translation of the {a} text below, from 1 (unusable) to 10 (excellent).",
            "Judge this translation from {a} to {b}. Reply with a 1-10 score."]
MO = {"afrikaans": "afr", "chewa": "nya", "ewe": "ewe", "fon": "fon", "hausa": "hau", "igbo": "ibo", "kinyarwanda": "kin", "oromo": "orm",
      "shona": "sna", "somali": "som", "southern_sotho": "sot", "swahili": "swh", "tswana": "tsn", "twi": "twi", "wolof": "wol",
      "xhosa": "xho", "yoruba": "yor", "zulu": "zul"}


def score_rows(token: str, rng: random.Random, stats, per_score: int = 300):
    """African-Languages-Lab/multi-open carries a 1-10 translation_quality_score. Rows are drawn evenly from the score levels
    that actually occur, so the model sees poor translations as well as good ones."""
    import shutil
    import tempfile
    from huggingface_hub import hf_hub_download
    for cfg, code in MO.items():
        cache = tempfile.mkdtemp()
        try:
            p = hf_hub_download("African-Languages-Lab/multi-open", f"data/english-{cfg}/train-00000.parquet", repo_type="dataset",
                                token=token, cache_dir=cache)
            pf = pq.ParquetFile(p)
            other = [c for c in pf.schema_arrow.names if c not in ("english", "translation_quality_score") and not c.endswith("_token_count")][0]
            res: dict[int, list] = collections.defaultdict(list)
            seen = collections.Counter()
            for b in pf.iter_batches(batch_size=20000, columns=["english", other, "translation_quality_score"]):
                d = b.to_pydict()
                for e, o, sc in zip(d["english"], d[other], d["translation_quality_score"]):
                    if sc is None or sc not in (1, 3, 4, 6, 7, 10) or not e or not o or not (15 <= len(e) <= 600 and 10 <= len(o) <= 800):
                        continue
                    seen[sc] += 1
                    if len(res[sc]) < per_score:
                        res[sc].append((e, o))
                    else:
                        j = rng.randint(0, seen[sc] - 1)
                        if j < per_score:
                            res[sc][j] = (e, o)
            for sc, items in res.items():
                for e, o in items:
                    if rng.random() < 0.5:
                        src, tgt, a, b_ = "eng", code, e, o
                    else:
                        src, tgt, a, b_ = code, "eng", o, e
                    instr = rng.choice(SC_INSTR).format(a=NAMES[src], b=NAMES[tgt])
                    ctx = f"Source ({NAMES[src]}): {a}\nTranslation ({NAMES[tgt]}): {b_}"
                    try:
                        reply = turn(src, "eng", ["<|analyze|>"], str(sc), label="score")
                    except Exception:
                        continue
                    stats[code] += 1
                    yield row("translation_score", instr, ctx, str(sc), src, "eng", reply, "African-Languages-Lab/multi-open")
        finally:
            shutil.rmtree(cache, ignore_errors=True)


# --------------------------------------------------------------------------------------------- Hausa spelling correction
FIX_INSTR = ["Correct the spelling mistakes in this Hausa text.", "This Hausa sentence has spelling errors. Rewrite it correctly.",
             "Fix the errors in the Hausa text below.", "Proofread this Hausa text and give the corrected version."]


def correct_rows(token: str, rng: random.Random, stats, cap: int = 30000):
    """ahmadmwali/hausa-writing-anomalies: `Clean` is real Hausa (Wikipedia-style) and `Noisy` is the same text with injected
    spelling errors. Space-tokenised punctuation is joined and English lines are dropped with GlotLID."""
    from huggingface_hub import HfApi, HfFileSystem
    from africomet_translation import lid
    files = sorted(f.path for f in HfApi(token=token).list_repo_tree("ahmadmwali/hausa-writing-anomalies", repo_type="dataset", recursive=True)
                   if f.path.endswith(".parquet") and "train" in f.path)
    fs = HfFileSystem(token=token)
    got = 0
    for path in files:
        with fs.open(f"datasets/ahmadmwali/hausa-writing-anomalies/{path}", "rb") as fh:
            for b in pq.ParquetFile(fh).iter_batches(batch_size=4000, columns=["Noisy", "Clean"]):
                d = b.to_pydict()
                pairs = []
                for n_, c_ in zip(d["Noisy"], d["Clean"]):
                    stats["seen"] += 1
                    n_, c_ = _fix_tok(n_ or ""), _fix_tok(c_ or "")
                    if n_ == c_ or not (6 <= len(c_.split()) <= 60) or abs(len(n_) - len(c_)) > 12:
                        stats["skipped"] += 1
                        continue
                    pairs.append((n_, c_))
                ok = lid.verify([c for _, c in pairs], "hau") if pairs else []
                for (n_, c_), g in zip(pairs, ok):
                    if not g:
                        stats["lid_out"] += 1
                        continue
                    try:
                        reply = turn("hau", "hau", ["<clean>"], c_, label="correct")
                    except Exception:                      # e.g. a stray markup tag in the source text
                        stats["assemble_rejected"] += 1
                        continue
                    yield row("correct", rng.choice(FIX_INSTR), n_, c_, "hau", "hau", reply, "ahmadmwali/hausa-writing-anomalies")
                    got += 1
                    if got >= cap:
                        return


def _fix_tok(t: str) -> str:
    t = re.sub(r"\s+([.,;:!?%)\]])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------------------------- lr-sum
def lrsum_rows(token: str, rng: random.Random, stats):
    from huggingface_hub import HfApi, hf_hub_download
    files = [f.path for f in HfApi(token=token).list_repo_tree("bltlab/lr-sum", repo_type="dataset", recursive=True)
             if f.path.endswith((".parquet", ".jsonl", ".json")) and any(k in f.path for k in ("kin", "sna"))]
    HEAD_INSTR = ["Write a headline for this article.", "Give this article a short headline.", "What would be a good title for this article?"]
    INSTR = ["Summarize the following article.", "Write a short summary of this text.", "Give me a brief summary of the article below.",
             "Condense this article into a few sentences."]
    for path in files:
        l = "kin" if "kin" in path else "sna"
        p = hf_hub_download("bltlab/lr-sum", path, repo_type="dataset", token=token)
        if p.endswith(".parquet"):
            rows = pq.read_table(p, columns=["text", "summary", "title"]).to_pylist()
        else:
            rows = [json.loads(x) for x in open(p, encoding="utf-8")] if p.endswith(".jsonl") else json.load(open(p, encoding="utf-8"))
        for r in rows:
            t, s = (r.get("text") or "").strip(), re.sub(r"<[^>]+>", "", r.get("summary") or "").strip()
            stats["seen"] += 1
            if not (400 <= len(t) <= 9000) or not (40 <= len(s) <= 900):
                stats["len"] += 1
                continue
            t = "\n".join(x for x in t.split("\n") if "Shakira izindi nkuru" not in x and "Saba amakuru muri email" not in x)
            from sft_new.domain import domain_flag
            flag = domain_flag(t)
            stats["flag" + flag] += 1
            try:
                reply = turn(l, l, ["<summarize>"], s, label="summary", flag=flag)
            except Exception:
                stats["assemble_rejected"] += 1
                continue
            yield row("summarize", rng.choice(INSTR), t, s, l, l, reply, "bltlab/lr-sum")
            title = re.sub(r"\s+", " ", (r.get("title") or "")).strip()
            if 15 <= len(title) <= 220:
                try:
                    reply = turn(l, l, ["<summarize>"], title, label="headline", flag=flag)
                except Exception:
                    continue
                yield row("headline", rng.choice(HEAD_INSTR), t, title, l, l, reply, "bltlab/lr-sum")


def relabel_swim_ir(token: str):
    """The swim-ir rows answer a question with the retrieved passage, which is what <|RAG|> names. Rewrite the plan of the pushed rows
    from <qa> to <|RAG|><qa> in place (same files, same rows)."""
    import shutil
    import tempfile
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    for code in ("swh", "yor"):
        name = f"instruct-swim-ir-{code}"
        files = sorted(f.path for f in api.list_repo_tree(REPO, path_in_repo=f"data/{name}", repo_type="dataset") if f.path.endswith(".parquet"))
        tmp = Path(tempfile.mkdtemp())
        for p in files:
            local = hf_hub_download(REPO, p, repo_type="dataset", token=token, cache_dir=str(tmp / "c"))
            tb = pq.read_table(local)
            msgs = [m.replace("<task_plan><qa></task_plan>", "<task_plan><|RAG|><qa></task_plan>") for m in tb["messages"].to_pylist()]
            out = tmp / "o" / name
            out.mkdir(parents=True, exist_ok=True)
            idx = tb.schema.get_field_index("messages")
            pq.write_table(tb.set_column(idx, "messages", pa.array(msgs)), out / Path(p).name, compression="zstd")
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(tmp / "o" / name), path_in_repo=f"data/{name}",
                          commit_message=f"{name}: plan is <|RAG|><qa>")
        shutil.rmtree(tmp, ignore_errors=True)
        print("relabelled", name, len(files), "files", file=sys.stderr, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["identify", "diacritize", "aya", "lrsum", "score", "correct", "rag"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--per-lang", type=int, default=3000)
    ap.add_argument("--langs", nargs="*")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(HERE))
    from africomet_translation.build import hf_token
    token = hf_token()
    os.environ["HF_TOKEN"] = token
    out = Path(a.out)
    name = f"structured-{a.task}"
    o = Out(out, name)
    stats = collections.defaultdict(int)
    rng = random.Random(f"structured-{a.task}")
    if a.task == "identify":
        langs = a.langs or list(NAMES.keys() - {"eng"})
        cues = build_cues(langs, token)
        for l in langs:
            k = 0
            for r in identify_rows(l, cues, token, a.per_lang, rng):
                o.add(r); k += 1
            stats[l] = k
            print("identify", l, k, file=sys.stderr, flush=True)
    elif a.task == "diacritize":
        for l in (a.langs or list(DI_INSTR)):
            k = 0
            for r in diacritize_rows(l, token, a.per_lang * 3, rng):
                o.add(r); k += 1
            stats[l] = k
            print("diacritize", l, k, file=sys.stderr, flush=True)
    elif a.task == "aya":
        for r in aya_rows(token, rng, stats):
            o.add(r); stats["kept"] += 1
    elif a.task == "rag":
        relabel_swim_ir(token)
        return
    elif a.task == "score":
        for r in score_rows(token, rng, stats):
            o.add(r); stats["kept"] += 1
    elif a.task == "correct":
        for r in correct_rows(token, rng, stats):
            o.add(r); stats["kept"] += 1
    else:
        for r in lrsum_rows(token, rng, stats):
            o.add(r); stats["kept"] += 1
    o.flush()
    (out / "stats").mkdir(exist_ok=True, parents=True)
    (out / "stats" / f"{name}.json").write_text(json.dumps(dict(stats), indent=1))
    print(name, dict(stats), o.total, "rows", file=sys.stderr)
    if a.push and o.total:
        push(out, name, token)


if __name__ == "__main__":
    main()
