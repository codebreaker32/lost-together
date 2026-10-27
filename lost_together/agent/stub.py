"""A deterministic stand-in for the VLM, for tests and offline development. It is NOT a result.

It reads the same messages the real model gets (the machine-readable lines in the first user message
and the JSON tool results), plays a sensible look-again script, and answers in the strict JSON format:
  turn 1: look_closer on the most threatening ambiguous candidate + query_map (parallel calls)
  turn 2: compare_frames on that candidate if a previous frame exists
  final : escape / regroup / look_again / keep_going from rescored confidence, motion and map risk
`fault` injects failures so the loop's fallback paths can be tested:
  invalid_json | timeout | error | bad_tool | bad_candidate | endless_tools | prose
"""
from __future__ import annotations

import json

from .llm import LLMError, LLMTimeout
from .tools import threat_weight

CAND_LABEL = "CANDIDATES_JSON:"
MAP_LABEL = "MAP_SUMMARY_JSON:"
PREV_LABEL = "PREVIOUS_FRAME:"


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def _labelled_json(text: str, label: str):
    i = text.find(label)
    if i < 0:
        return None
    rest = text[i + len(label):].strip()
    try:
        return json.JSONDecoder().raw_decode(rest)[0]
    except json.JSONDecodeError:
        return None


def _call(i: int, name: str, args: dict) -> dict:
    return {"id": f"call_{i}_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class StubModel:
    def __init__(self, fault: str | None = None, escape_evidence: float = 0.9):
        self.fault = fault
        self.escape_evidence = escape_evidence
        self.n = 0

    def __call__(self, messages: list[dict], tools: list[dict] | None) -> dict:
        self.n += 1
        if self.fault == "timeout":
            raise LLMTimeout("stub: simulated timeout")
        if self.fault == "error":
            raise LLMError("stub: simulated server error")
        first_user = next((_text(m["content"]) for m in messages if m["role"] == "user"), "")
        cands = _labelled_json(first_user, CAND_LABEL) or []
        has_prev = f"{PREV_LABEL} yes" in first_user
        names = {}
        results: dict[str, dict] = {}
        for m in messages:
            for tc in m.get("tool_calls") or []:
                names[tc["id"]] = tc["function"]["name"]
            if m["role"] == "tool":
                try:
                    results[names.get(m["tool_call_id"], "?")] = json.loads(_text(m["content"]))
                except json.JSONDecodeError:
                    results[names.get(m["tool_call_id"], "?")] = {}
        tool_turns = sum(1 for m in messages if m["role"] == "assistant" and m.get("tool_calls"))

        if self.fault == "endless_tools" and tools:
            return {"role": "assistant", "content": None, "tool_calls": [_call(self.n, "query_map", {})]}
        if self.fault == "bad_tool" and tools and tool_turns == 0:
            return {"role": "assistant", "content": None, "tool_calls": [_call(self.n, "rm_rf", {"path": "/"})]}

        threats = [c for c in cands if threat_weight(c["cls"]) > 0]
        top = max(threats, key=lambda c: threat_weight(c["cls"]) * c["score"], default=None)
        if tools and top is not None:
            if tool_turns == 0:
                return {"role": "assistant", "content": None,
                        "tool_calls": [_call(1, "look_closer", {"candidate_id": top["id"]}), _call(1, "query_map", {})]}
            if tool_turns == 1 and has_prev:
                return {"role": "assistant", "content": None,
                        "tool_calls": [_call(2, "compare_frames", {"candidate_id": top["id"]})]}
        if tools and top is None and tool_turns == 0:
            return {"role": "assistant", "content": None, "tool_calls": [_call(1, "query_map", {})]}

        # final answer
        risk = float(results.get("query_map", {}).get("risk_near_hiker", 0.0))
        score, cls = (top["score"], top["cls"]) if top else (0.0, "none")
        lc = results.get("look_closer", {})
        if "after" in lc:
            score, cls = lc["after"]["score"], lc["after"]["cls"]
        moving = results.get("compare_frames", {}).get("moving")
        threat = threat_weight(cls) * score if top else 0.0
        if moving is False:
            threat *= 0.3                              # static shape: likely a rock or stump
        elif moving is True:
            threat = min(1.0, threat + 0.4)
        evidence = risk + threat
        if evidence >= self.escape_evidence:
            decision = "escape"
        elif evidence >= 0.45:
            decision = "regroup"
        elif top is not None and 0.25 <= score <= 0.75 and not lc:
            decision = "look_again"
        else:
            decision = "keep_going"
        why = (f"{cls} conf {score:.2f}" + (f", moving={moving}" if moving is not None else "")
               + f", map risk {risk:.2f} -> evidence {evidence:.2f}")
        out = {"decision": decision, "candidate_id": top["id"] if top else None, "why": why}
        if self.fault == "invalid_json":
            return {"role": "assistant", "content": "{decision: escape, because bear"}
        if self.fault == "prose":
            return {"role": "assistant", "content": "I think we should run! " + json.dumps(out)}
        if self.fault == "bad_candidate":
            out["candidate_id"] = 99
        return {"role": "assistant", "content": json.dumps(out)}

