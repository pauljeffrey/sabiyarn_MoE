#!/usr/bin/env python3
"""Runnable examples. Needs torch + transformers for the real model; the last
example needs neither and is what CI exercises.

    python -m agent.example --repo Aletheia-ng/SabiYarn_MoE-280M
    python -m agent.example --dry-run        # scripted model, no GPU
"""

from __future__ import annotations

import argparse
import json
from typing import Literal

from . import (
    Agent, ConfirmNeeded, Delta, Failed, Final, HFBackend, ScriptedLLM, Think, ToolCalled,
    ToolReturned, tool,
)
from .protocol import EOS, render_tool_call

# --- tools ------------------------------------------------------------------
# Names and descriptions mirror data-gen/seeds/build_seeds.py's catalogue, so a
# model trained on that corpus sees tools it recognises.


@tool
def get_weather_forecast(city: str, days: int = 1) -> dict:
    """Weather forecast for a Nigerian city. Returns temperature in Celsius."""
    return {"city": city, "days": days, "high_c": 31, "low_c": 24, "summary": "humid, scattered showers"}


@tool
def calculate(expression: str) -> dict:
    """Evaluate an arithmetic expression, e.g. '12 * (3 + 4)'."""
    # Arithmetic only: eval on model-supplied text would otherwise be a hole
    # straight into the host. ast.literal_eval cannot do operators, so the
    # expression is walked explicitly.
    import ast
    import operator

    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
    }

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return {"expression": expression, "result": walk(ast.parse(expression, mode="eval"))}


@tool
def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Search the local document store. Returns a list of passages, not an answer."""
    # Stand-in for a real retriever: swap the body for your vector store and the
    # agent loop does not change. The docstring's "not an answer" phrasing
    # matters -- it is what teaches the model to read results before replying.
    corpus = [
        {"title": "Lagos transport policy 2026", "text": "The new policy adds 120 buses on the Ikorodu corridor."},
        {"title": "Kano power schedule", "text": "Band A feeders receive 20 hours of supply daily."},
        {"title": "Cassava prices", "text": "A tonne of cassava sells for N220,000 in Oyo state."},
    ]
    terms = {t for t in query.lower().split() if len(t) > 2}
    scored = [
        (len(terms & set((d["title"] + " " + d["text"]).lower().split())), d) for d in corpus
    ]
    return [d for score, d in sorted(scored, key=lambda s: s[0], reverse=True)[:top_k] if score]


@tool(requires_confirmation=True)
def send_message(to: str, body: str, channel: Literal["sms", "whatsapp"] = "sms") -> dict:
    """Send a message to a contact. Requires confirmation."""
    return {"sent": True, "to": to, "channel": channel, "chars": len(body)}


TOOLS = [get_weather_forecast, calculate, search_documents, send_message]


# --- examples ---------------------------------------------------------------
def demo_streaming(bot: Agent, prompt: str) -> None:
    print(f"\n--- streaming: {prompt}")
    for ev in bot.run(prompt):
        if isinstance(ev, Delta):
            print(ev.text, end="", flush=True)
        elif isinstance(ev, Think):
            print(f"\n  [think] {ev.text}")
        elif isinstance(ev, ToolCalled):
            print(f"\n  [call] {ev.name}({json.dumps(ev.arguments, ensure_ascii=False)})")
        elif isinstance(ev, ToolReturned):
            print(f"  [result] {ev.error or ev.value}")
        elif isinstance(ev, ConfirmNeeded):
            answer = input(f"\n  run {ev.call.name}({ev.call.raw_arguments})? [y/N] ")
            ev.approve() if answer.strip().lower().startswith("y") else ev.decline()
        elif isinstance(ev, Final):
            print(f"\n  [done in {ev.rounds} round(s)]")
        elif isinstance(ev, Failed):
            print(f"\n  [failed] {ev.reason}")


def demo_structured(bot: Agent) -> None:
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "high_c": {"type": "number"},
            "advice": {"type": "string"},
        },
        "required": ["city", "high_c", "advice"],
        "additionalProperties": False,
    }
    print("\n--- structured output")
    try:
        print(bot.structured("Summarise today's Lagos forecast as JSON.", schema))
    except Exception as exc:
        print(f"  [failed] {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="Aletheia-ng/SabiYarn_MoE-280M", help="HF model repo or local checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="use a scripted model; no torch needed")
    ap.add_argument("--prompt", default="What is the weather in Lagos today?")
    args = ap.parse_args()

    if args.dry_run:
        llm = ScriptedLLM(
            replies=[
                # Step 1 ends at </tool_call> with NO </s>: the turn continues
                # after the tool result (chat template v2).
                "<|input_lang|><eng><think>I need live weather data.</think>"
                "<task_plan><|RAG|></task_plan>"
                + render_tool_call("get_weather_forecast", {"city": "Lagos"}),
                # Step 2 is the continuation of the SAME turn, and closes it.
                "<|input_lang|><eng><task_plan><|explain|></task_plan><|target_lang|><eng>"
                "<response>It should reach 31C in Lagos today, humid with scattered showers.</response>" + EOS,
            ]
        )
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.repo, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.repo, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        llm = HFBackend(model, tok)

    bot = Agent(llm, tools=TOOLS, identity="You are an AI assistant. Your name is Ọmọlúàbí.")
    demo_streaming(bot, args.prompt)
    if not args.dry_run:
        demo_streaming(bot, "What does the document store say about the Lagos transport policy?")
        demo_structured(bot)


if __name__ == "__main__":
    main()
