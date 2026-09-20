"""Prompt builders. Two styles:

  tag  -- the multitask pretraining format (see training/constant_tokens.py): the model was pretrained
          on `<translate> {src} <tgt> {tgt}`, `<classify> {text} <topic> : {label}`,
          `<classify> {text} <sentiment> : {label}`, `<NER> {tokens} <tag> : {BIO tags}`.
          Use this for pretrained (base) checkpoints, zero-shot.
  chat -- the SFT chat template with an English instruction, for instruction-tuned checkpoints.

Every builder returns (context_text, continuation_prefix) where relevant: classification is scored by
the likelihood of each label as the continuation, not by parsing free-form generations.
"""

from __future__ import annotations

from eval_suite.langs import Lang

SYSTEM = "You are a helpful multilingual assistant for West African language speakers."


def _chat(tokenizer, user: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def translation(style: str, tokenizer, src: Lang, tgt: Lang, text: str) -> str:
    if style == "tag":
        return f"<translate> {text} {tgt.tag}"
    return _chat(tokenizer, f"Translate the following {src.name} text into {tgt.name}:\n\n{text}")


def classification(style: str, tokenizer, kind: str, text: str, labels: list[str]) -> tuple[str, str]:
    """kind: 'topic' | 'sentiment'. Returns (context, continuation_prefix); each label is scored as
    continuation_prefix + label."""
    if style == "tag":
        return f"<classify> {text} <{kind}> :", " "
    what = "topic" if kind == "topic" else "sentiment"
    return _chat(tokenizer, f"What is the {what} of the following text? Answer with one of: {', '.join(labels)}.\n\n{text}"), ""


def ner(tokens: list[str]) -> str:
    return f"<NER> {' '.join(tokens)} <tag> :"


def mmlu(subject: str, question: str, choices: list[str], shots: list[dict]) -> str:
    """Standard MMLU prompt (Hendrycks et al.): k worked examples, then the question, 'Answer:'."""

    def block(q, ch, ans=None):
        lines = [q.strip()] + [f"{l}. {c}" for l, c in zip("ABCD", ch)]
        lines.append("Answer:" + (f" {'ABCD'[ans]}" if ans is not None else ""))
        return "\n".join(lines)

    head = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    parts = [block(s["question"], s["choices"], s["answer"]) for s in shots] + [block(question, choices)]
    return head + "\n\n".join(parts)
