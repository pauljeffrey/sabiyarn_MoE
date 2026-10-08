"""The three-agent loop: user agent -> assistant (with tools) -> refiner, turn by turn.

WHY THIS EXISTS. The single-call generator asks one request to produce a whole conversation AND a ranked pair
AND the markers AND a discriminator. Measured on 120 live requests, 70.8% survive, and the pairs it does
produce have a structural weakness no amount of yield fixes: both candidates come from one forward pass by a
model that knows which is which, so the `rejected` one is cartoonishly bad. One real example from a kept
sample's reasoning block: "The tool returned an error, but I will tell the user it was successful to be
helpful." Nothing is learnt from avoiding that.

Here every call does ONE thing -- write a user message, answer it, rewrite the answer -- so the schema
surface per call is a single turn, and the pair is a real first attempt against a real improvement of it.
One pair per TURN rather than one per conversation.

COSTS, from the model in scratchpad/cost_model.py, anchored on measured token counts:
  single call      $0.698 / 1k pairs    ($141 for the 202k-pair corpus)
  this loop        $0.561 / 1k pairs    ($113) with chunked retrieval and eviction
  this loop        $1.232 / 1k pairs    ($249) if whole documents stay in the agent's history
The 2x swing between the last two lines is why `_RETRIEVED` is evicted after the turn that cited it, and why
the retriever hands back passages rather than articles. The passages are still written into the published
record -- see retrieval.py for why that is not a contradiction.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from agents import agent_prompts as AP
from agents import personas
from agents.retrieval import Passage, Retriever, render
from assemble import VALID_VERBS, AssemblyError, build_messages, structure_of

# The reply may not grow. The refiner's job is to change how a reply SOUNDS, and a preference dataset whose
# chosen side is reliably longer teaches "longer is better" instead. 1.15 leaves room for a rewrite that
# genuinely needs a few more characters in a language with longer words, and nothing more.
MAX_LENGTH_RATIO = 1.15


@dataclass
class Turn:
    """One exchange. `retrieved` is what the tools returned, kept for the record after eviction."""
    user: str
    assistant: dict                       # the base assistant's fields: think/task_plan/response/tool_call
    refined: Optional[dict] = None        # the refiner's version -- the `chosen` side
    tool_calls: list[dict] = field(default_factory=list)
    retrieved: list[Passage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    lang: str
    turns: list[Turn] = field(default_factory=list)
    persona: str = ""
    register: str = ""
    stats: dict = field(default_factory=dict)


def _json_obj(text: str) -> Optional[dict]:
    """One JSON object out of a model reply, tolerating a fence and surrounding prose."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except ValueError:
        s, e = t.find("{"), t.rfind("}")
        if 0 <= s < e:
            try:
                v = json.loads(t[s:e + 1])
                return v if isinstance(v, dict) else None
            except ValueError:
                return None
        return None


_RESTATE = re.compile(
    r"^(?:the\s+user|the\s+question|they|user)\s+"
    r"(?:is\s+asking|is\s+requesting|has\s+asked|asked|asks|wants?\s+to\s+know|wants?|needs?|"
    r"would\s+like|is\s+looking\s+for)\b[^.!?]*[.!?]", re.IGNORECASE)


def _tidy_think(t: Any) -> str:
    """Drop a leading restatement of the question, and drop the block entirely when that is ALL it was.

    Measured on the single-call path: 14 of 21 think blocks opened with a restatement, and those clauses were
    39% of all think text. Measured on this loop: 3 of 4 think blocks were nothing BUT a restatement -- "The
    user is asking for app recommendations for daily savings in Hausa." -- with no decision after it. A block
    like that is pure spend, and the contract already says to omit <think> where nothing needs deciding, so
    omitting it here is enforcing the instruction rather than overriding it.
    """
    s = str(t or "").strip()
    if not s:
        return ""
    stripped = _RESTATE.sub("", s, count=1).lstrip()
    if stripped == s:
        return s
    # two words is enough for a real decision ("Look it up.", "Needs the 2025 figure.")
    return stripped if len(stripped.split()) >= 2 else ""



# LaTeX leaks out of a model that has read a lot of papers: a live Hausa reply rendered a hierarchy as
# "AI $\\rightarrow$ Machine Learning". The arrow is right and the markup is not -- it would be trained into
# a model whose users have never seen a dollar-delimited equation. Repaired rather than rejected, because the
# content is sound.
_MATHY = [
    (re.compile(r"\$\s*\\(?:rightarrow|to|Rightarrow)\s*\$"), " \u2192 "),
    (re.compile(r"\$\s*\\(?:leftarrow|gets)\s*\$"), " \u2190 "),
    (re.compile(r"\$\s*\\times\s*\$"), " \u00d7 "),
    (re.compile(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}"), r"\1"),
    (re.compile(r"\$+([^$]{1,40}?)\$+"), r"\1"),
]


def _clean_response(s: str) -> str:
    for pat, rep in _MATHY:
        s = pat.sub(rep, s)
    return re.sub(r"[ \t]{2,}", " ", s).strip()


class Loop:
    """Drives one conversation. `ask` is any callable (prompt_messages, max_tokens) -> text."""

    def __init__(self, ask, retriever: Retriever, *, lang: str, lang_name: str, today: str,
                 tools: Optional[list[dict]] = None, plan: Optional[list[str]] = None,
                 subject: str = "", rng: Optional[random.Random] = None, mode: str = "rl",
                 tags: Optional[list[str]] = None):
        self.ask = ask
        self.retriever = retriever
        self.lang, self.lang_name, self.today = lang, lang_name, today
        self.tools = tools or []
        self.plan = plan or []
        self.subject = subject
        self.rng = rng or random.Random(0)
        self.mode = mode
        self.tags = list(tags or [])
        self.persona = personas.pick(self.rng)
        self.stats: dict[str, int] = {}

    # ---- prompts

    def _tools_block(self) -> str:
        if not self.tools:
            return ("TOOLS: none. If you do not know something and cannot look it up, say so plainly.")
        lines = [f"  - {t['name']}({', '.join((t.get('arguments') or {}).keys()) or '...'}): "
                 f"{t.get('description', '')}" for t in self.tools]
        return ("TOOLS you may call, one per turn, by returning a `tool_call`:\n" + "\n".join(lines) +
                "\nCall one only when you actually need it. After the result comes back, answer from it.")

    def _plan_hint(self) -> str:
        return ", ".join(f'"{v}"' for v in self.plan) if self.plan else '"<|chat|>"'

    def _history(self, upto: Optional[int] = None, *, for_user_agent: bool = False) -> list[dict]:
        """The conversation so far, WITHOUT retrieved passages.

        Eviction happens here, by construction: nothing in this list carries a tool result body. What is left
        behind is a one-line note of what was retrieved, which is what a later turn actually refers back to --
        the assistant's own answer, not the source text. A turn that needs the document again calls the tool
        again, which is also what a real RAG assistant does.
        """
        out: list[dict] = []
        for t in self.turns[:upto]:
            out.append({"role": "user", "content": t.user})
            best = t.refined or t.assistant
            body = str(best.get("response") or "")
            if t.retrieved and not for_user_agent:
                ids = ", ".join(p.id for p in t.retrieved[:3])
                body = f"[retrieved {len(t.retrieved)} passages: {ids}]\n{body}"
            out.append({"role": "assistant", "content": body})
        return out

    # ---- the three agents

    def user_turn(self, index: int) -> str:
        sysmsg = AP.USER_AGENT.format(
            lang_name=self.lang_name, persona=self.persona.description,
            register_hint=AP.register_hint(self.lang_name, self.persona.register),
            turn_instruction=AP.FIRST_TURN if index == 0 else AP.LATER_TURN,
            subject=self.subject or "anything this person would actually ask about")
        msgs = [{"role": "system", "content": sysmsg}] + self._history(for_user_agent=True)
        if index:
            msgs.append({"role": "user", "content": "Write the next message."})
        else:
            msgs.append({"role": "user", "content": "Write the first message."})
        txt = (self.ask(msgs, 160) or "").strip()
        # a model asked for one line sometimes returns a labelled line or a quoted one
        txt = re.sub(r'^(?:user|message)\s*[:\-]\s*', "", txt, flags=re.I).strip().strip('"').strip()
        return txt.split("\n")[0].strip() if txt.count("\n") > 2 else txt

    def assistant_turn(self, user_msg: str, *, sft: bool = False) -> tuple[dict, list[dict], list[Passage]]:
        """Answer, resolving at most one tool call. Returns (fields, calls_made, passages_retrieved)."""
        template = AP.SFT_AI_AGENT if sft else AP.AI_AGENT
        sysmsg = template.format(lang_name=self.lang_name, today=self.today,
                                 tools_block=self._tools_block(), plan_hint=self._plan_hint(),
                                 task_directive=AP.task_directive(self.tags),
                                 **({"flavour_hint": AP.flavour_for(self.lang)} if sft else {}))
        base = [{"role": "system", "content": sysmsg}] + self._history()
        base.append({"role": "user", "content": user_msg})

        calls, passages = [], []
        obj = _json_obj(self.ask(base, 700)) or {}
        if obj.get("tool_call"):
            call = obj["tool_call"]
            calls.append(call)
            result, passages = self._run_tool(call, user_msg)
            # the result is visible for THIS turn only
            base.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
            base.append({"role": "user", "content": f"Tool result:\n{result}\n\nNow answer from it."})
            obj2 = _json_obj(self.ask(base, 700)) or {}
            if obj2.get("response"):
                obj2.setdefault("task_plan", obj.get("task_plan") or [])
                obj = obj2
            else:
                self.stats["tool_second_call_empty"] = self.stats.get("tool_second_call_empty", 0) + 1
        obj["think"] = _tidy_think(obj.get("think"))
        if obj.get("response"):
            obj["response"] = _clean_response(obj["response"])
        return obj, calls, passages

    def _run_tool(self, call: dict, user_msg: str) -> tuple[str, list[Passage]]:
        """Execute a tool. Retrieval is real; everything else is a plausible stub.

        Retrieval being real is the point: the passages come from the document pool, so the answer can be
        checked against them afterwards and "cited the wrong passage" is a scoreable event rather than an
        invisible one.
        """
        name = str(call.get("name") or "")
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if name in ("search_documents", "search_internet", "fetch_page", "lookup_legal_info",
                    "lookup_health_guidance", "lookup_crop_guidance"):
            q = str(args.get("query") or args.get("q") or user_msg)
            hits = self.retriever.search(q, k=4, lang="")
            self.stats["retrievals"] = self.stats.get("retrievals", 0) + 1
            return render(hits), hits
        if name == "get_current_datetime":
            return json.dumps({"datetime": f"{self.today}T09:14:00+01:00"})
        return json.dumps({"status": "ok", "note": f"{name} executed", "arguments": args},
                          ensure_ascii=False)

    def refine(self, fields: dict, user_msg: str, findings: str = "") -> Optional[dict]:
        """Rewrite for nativeness, same length or shorter. Returns None when the rewrite is unusable."""
        original = str(fields.get("response") or "")
        if not original:
            return None
        sysmsg = AP.REFINER.format(lang_name=self.lang_name, flavour_hint=AP.flavour_for(self.lang),
                                   fix_block=AP.FIX_BLOCK.format(findings=findings) if findings else "")
        msgs = [{"role": "system", "content": sysmsg}] + self._history()
        msgs.append({"role": "user", "content":
                     f"The user said:\n{user_msg}\n\nThe assistant replied with:\n"
                     f"{json.dumps(fields, ensure_ascii=False)}\n\nRewrite it."})
        obj = _json_obj(self.ask(msgs, 700)) or {}
        new = str(obj.get("response") or "")
        if not new:
            self.stats["refine_empty"] = self.stats.get("refine_empty", 0) + 1
            return None
        # THE LENGTH GUARD. Without it the refiner mostly makes replies longer and the pair encodes verbosity
        # rather than nativeness -- see agent_prompts.REFINER for why that is the failure to avoid.
        if len(new) > MAX_LENGTH_RATIO * len(original):
            self.stats["refine_too_long"] = self.stats.get("refine_too_long", 0) + 1
            return None
        if new.strip() == original.strip():
            self.stats["refine_identical"] = self.stats.get("refine_identical", 0) + 1
            return None
        # THE PLAN COMES FROM THE ORIGINAL, ALWAYS. Asked for a task_plan, the refiner returns a list of the
        # edits it is about to make -- "Replace 'manhajoji' with 'apps'", "Insert natural particles like
        # 'dai'" -- because that IS a plan, just not this one. It cost 11 of 17 pairs on a live run before
        # the field was removed from its contract and overridden here. The plan is substance; the refiner
        # only has licence over register.
        obj["task_plan"] = list(fields.get("task_plan") or [])
        # And the THINK comes from the original too, for the same reason as the plan. Invited to "keep or
        # shorten" it, the refiner wrote one from scratch on every turn -- including the turns where the base
        # agent had correctly omitted it because nothing needed deciding. 17 of 17 chosen replies carried a
        # <think> block against 5 of 18 base replies, which is the refiner padding the side we reward.
        obj["think"] = _tidy_think(fields.get("think"))
        if fields.get("tool_call"):
            obj["tool_call"] = fields["tool_call"]
        return obj

    # ---- driver

    @property
    def turns(self) -> list[Turn]:
        return getattr(self, "_turns", [])

    def run(self, n_turns: int = 4, *, findings_for=None) -> Conversation:
        """Drive `n_turns` exchanges. `findings_for(fields, turn)` may return verifier findings as a string,
        which the refiner is asked to fix in addition to the register."""
        self._turns: list[Turn] = []
        sft = self.mode == "sft"
        for i in range(n_turns):
            user = self.user_turn(i)
            if not user:
                self.stats["user_turn_empty"] = self.stats.get("user_turn_empty", 0) + 1
                break
            fields, calls, passages = self.assistant_turn(user, sft=sft)
            if not (fields.get("response") or fields.get("tool_call")):
                self.stats["assistant_turn_empty"] = self.stats.get("assistant_turn_empty", 0) + 1
                break
            turn = Turn(user=user, assistant=fields, tool_calls=calls, retrieved=passages)
            if not sft:
                findings = findings_for(fields, turn) if findings_for else ""
                turn.refined = self.refine(fields, user, findings)
                if turn.refined is None:
                    self.stats["turn_without_pair"] = self.stats.get("turn_without_pair", 0) + 1
            self._turns.append(turn)
        return Conversation(lang=self.lang, turns=self._turns, persona=self.persona.key,
                            register=self.persona.register, stats=dict(self.stats))


# --------------------------------------------------------------------------- record building


def _valid_plan(plan: Any, default_plan: Optional[list[str]] = None,
                fallback: str = "<|chat|>") -> list[str]:
    """Only REAL tokens survive, with the task's canonical plan standing in when nothing does.

    An agent asked for a plan sometimes answers in prose ("Simplify the phrasing") or with the call it is
    about to make ("search_internet(query='...')"), and a plan the tokenizer does not recognise fails
    assembly and costs the whole turn. Every path that builds a turn goes through here, because the one that
    did not -- the tool-call turn -- is exactly where the next leak appeared.
    """
    keep = [v for v in (plan or []) if isinstance(v, str) and v in VALID_VERBS]
    if keep:
        return keep
    keep = [v for v in (default_plan or []) if isinstance(v, str) and v in VALID_VERBS]
    return keep or [fallback]


def _assistant_turns(fields: dict, lang: str, target: str,
                     default_plan: Optional[list[str]] = None) -> list[dict]:
    """The structured turns for one assistant reply, in the shape assemble.build_messages wants.

    `task_plan` is REQUIRED by the assembler, and an agent that forgets it would otherwise cost the whole
    turn. The task's canonical plan is already known from the row, so falling back to it is deterministic
    rather than a guess -- the same reasoning as the per-tool plan inference in postprocess_gen.
    """
    t: dict[str, Any] = {"role": "assistant", "input_lang": lang}
    if fields.get("think"):
        t["think"] = fields["think"]
    t["task_plan"] = _valid_plan(fields.get("task_plan"), default_plan)
    if fields.get("tool_call"):
        t["tool_call"] = fields["tool_call"]
    else:
        t["target_lang"] = target
        t["response"] = fields.get("response") or ""
    return [t]


def records_from(conv: Conversation, *, row: dict, lang: str, target_lang: str, model: str,
                 subject: str = "", tasks: Optional[list[str]] = None,
                 tools: Optional[list[str]] = None,
                 default_plan: Optional[list[str]] = None) -> list[dict]:
    """One record PER PAIRED TURN, in the schema the existing postprocess and trainers already read.

    The published prefix carries the tool call AND the retrieved passages as a tool result, even though the
    generator evicted them -- otherwise an SFT record would show an answer with no visible source and teach
    the model to answer from nowhere.
    """
    out: list[dict] = []
    prefix: list[dict] = []
    default_plan = list(default_plan or row.get("task_plan") or [])
    for i, turn in enumerate(conv.turns):
        prefix.append({"role": "user", "content": turn.user})
        base_turns = _assistant_turns(turn.assistant, lang, target_lang, default_plan)
        if turn.tool_calls:
            # call turn, then the result it got back, then the answer
            call_turn = {"role": "assistant", "input_lang": lang,
                         "task_plan": _valid_plan(turn.assistant.get("task_plan"), default_plan,
                                                  "<|plan|>"),
                         "tool_call": turn.tool_calls[0]}
            if turn.assistant.get("think"):
                call_turn["think"] = turn.assistant["think"]
            result = render(turn.retrieved) if turn.retrieved else json.dumps({"status": "ok"})
            prefix.append(call_turn)
            prefix.append({"role": "tool", "name": turn.tool_calls[0].get("name") or "tool",
                           "content": result})
            # The answer turn is the same fields WITHOUT the call -- the call already happened above. If the
            # post-tool request came back empty there is no answer to strip to, so the turn stays a call and
            # yields no pair rather than an unassemblable fragment.
            if base_turns[0].get("response"):
                base_turns = [{k: v for k, v in base_turns[0].items() if k != "tool_call"}]
            else:
                conv.stats["turn_had_no_answer_after_tool"] = \
                    conv.stats.get("turn_had_no_answer_after_tool", 0) + 1
                prefix.extend([])
                continue

        if turn.refined is None:
            prefix.extend(base_turns)
            continue

        try:
            prompt_messages = build_messages(prefix)
            chosen = build_messages(_assistant_turns(turn.refined, lang, target_lang,
                                                     default_plan))[0]["content"]
            rejected = build_messages(base_turns)[0]["content"]
        except AssemblyError as exc:
            # COUNTED, not swallowed. A pair lost here is indistinguishable from a pair never generated
            # unless the reason is recorded, and that is how 4 of 4 turns of a good conversation went
            # missing on the first live run with an empty stats dict to show for it.
            conv.stats[f"assembly_failed:{exc}"] = conv.stats.get(f"assembly_failed:{exc}", 0) + 1
            prefix.extend(base_turns)
            continue

        rec = {
            "id": f"{row.get('custom_id') or row.get('id')}__t{i + 1}",
            "lang": lang, "model": model,
            "domain": row.get("domain") or "", "subtopic": subject,
            "tasks": list(tasks or []), "tools": list(tools or []),
            "persona": conv.persona, "register": conv.register,
            "prompt_messages": prompt_messages,
            "ranking": ["best", "worst"],
            "response_1": chosen, "response_2": rejected,
            "expected_structure": structure_of(chosen),
            "retrieved_context": [
                {"id": p.id, "title": p.title, "section": p.section, "score": p.score, "text": p.text}
                for p in turn.retrieved],
            "turn_index": i + 1,
        }
        out.append(rec)
        # later turns continue from the BETTER reply
        prefix.extend(_assistant_turns(turn.refined, lang, target_lang, default_plan))
    return out


def sft_record(conv: Conversation, *, row: dict, lang: str, target_lang: str, model: str,
               subject: str = "", tasks: Optional[list[str]] = None,
               tools: Optional[list[str]] = None,
               default_plan: Optional[list[str]] = None) -> Optional[dict]:
    """One SFT record for the whole conversation: no pair, one good reply per user message."""
    turns: list[dict] = []
    retrieved: list[dict] = []
    default_plan = list(default_plan or row.get("task_plan") or [])
    for turn in conv.turns:
        turns.append({"role": "user", "content": turn.user})
        if turn.tool_calls:
            call_turn = {"role": "assistant", "input_lang": lang,
                         "task_plan": _valid_plan(turn.assistant.get("task_plan"), default_plan,
                                                  "<|plan|>"),
                         "tool_call": turn.tool_calls[0]}
            if turn.assistant.get("think"):
                call_turn["think"] = turn.assistant["think"]
            turns.append(call_turn)
            turns.append({"role": "tool", "name": turn.tool_calls[0].get("name") or "tool",
                          "content": render(turn.retrieved) if turn.retrieved else
                          json.dumps({"status": "ok"})})
            retrieved += [{"id": p.id, "title": p.title, "section": p.section, "text": p.text}
                          for p in turn.retrieved]
            turns.append({k: v for k, v in
                          _assistant_turns(turn.assistant, lang, target_lang, default_plan)[0].items()
                          if k != "tool_call"})
        else:
            turns.extend(_assistant_turns(turn.assistant, lang, target_lang, default_plan))
    if len(turns) < 2:
        return None
    try:
        msgs = build_messages(turns)
    except AssemblyError:
        return None
    return {"id": row.get("custom_id") or row.get("id"), "lang": lang, "model": model,
            "domain": row.get("domain") or "", "subtopic": subject,
            "tasks": list(tasks or []), "tools": list(tools or []),
            "persona": conv.persona, "register": conv.register,
            "messages": msgs, "retrieved_context": retrieved, "n_turns": len(conv.turns)}
