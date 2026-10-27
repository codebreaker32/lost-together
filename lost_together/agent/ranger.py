"""Human control: the ranger console and the autonomy dial.

  act     the agent's decision is executed directly (the ranger is notified and can audit it later)
  advise  an `escape` order (the evacuation broadcast) becomes a pending approval; a ranger approves it,
          or overrides it with another action. Other decisions pass through.

Approvers: CLIApprover (interactive console with a timeout), AutoApprover (batch runs), ScriptedApprover
(tests / replays of recorded ranger choices). Every review is appended to the decision's trace and,
optionally, to a jsonl ranger log, so approvals and overrides are auditable.
"""
from __future__ import annotations

import json
import select
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, TextIO

from .loop import DECISIONS, Decision

GATED = ("escape",)                    # decisions that need approval under ADVISE


class Autonomy(str, Enum):
    ADVISE = "advise"
    ACT = "act"


@dataclass
class Approval:
    approved: bool
    by: str
    final_action: str
    note: str = ""
    wait_s: float = 0.0


Approver = Callable[[Decision], Approval]


class AutoApprover:
    """Batch mode: approve (or reject) every pending order, and say so in the trace."""

    def __init__(self, approve: bool = True, fallback_action: str = "regroup"):
        self.approve, self.fallback_action = approve, fallback_action

    def __call__(self, d: Decision) -> Approval:
        if self.approve:
            return Approval(True, "auto", d.decision, "auto-approved (batch mode)")
        return Approval(False, "auto", self.fallback_action, "auto-rejected (batch mode)")


class ScriptedApprover:
    """Recorded ranger choices: {point_id: action}; an action equal to the decision means approve."""

    def __init__(self, script: dict[str, str], default: str | None = None):
        self.script, self.default = script, default

    def __call__(self, d: Decision) -> Approval:
        action = self.script.get(d.point_id, self.default or d.decision)
        return Approval(action == d.decision, "scripted", action,
                        "approved" if action == d.decision else f"overridden to {action}")


class CLIApprover:
    """Interactive ranger console. On timeout it applies `on_timeout` (default: approve, because a missed
    evacuation is the costlier failure) and records that nobody answered."""

    def __init__(self, timeout_s: float = 30.0, on_timeout: str = "approve",
                 stdin: TextIO | None = None, stdout: TextIO | None = None):
        self.timeout_s, self.on_timeout = timeout_s, on_timeout
        self.stdin, self.stdout = stdin or sys.stdin, stdout or sys.stdout

    def _readline(self) -> str | None:
        try:
            ready, _, _ = select.select([self.stdin], [], [], self.timeout_s)
        except (ValueError, OSError, TypeError):           # not a real file (e.g. StringIO in tests)
            ready = [self.stdin]
        return self.stdin.readline() if ready else None

    def __call__(self, d: Decision) -> Approval:
        out = self.stdout
        out.write(f"\n=== RANGER CONSOLE: pending {d.decision.upper()} order for {d.point_id} ===\n"
                  f"  agent: {d.story()}\n  why:   {d.why}\n"
                  f"  [a]pprove  [r]eject -> regroup  or type an action {list(DECISIONS)}  "
                  f"({self.timeout_s:.0f}s, then {self.on_timeout}): ")
        out.flush()
        t0 = time.perf_counter()
        line = self._readline()
        wait = time.perf_counter() - t0
        if line is None:
            ok = self.on_timeout == "approve"
            return Approval(ok, "timeout", d.decision if ok else "regroup",
                            f"no ranger response in {self.timeout_s:.0f}s; {self.on_timeout}", wait)
        ans = line.strip().lower()
        if ans in ("", "a", "y", "yes", "approve"):
            return Approval(True, "ranger", d.decision, "approved", wait)
        if ans in ("r", "n", "no", "reject"):
            return Approval(False, "ranger", "regroup", "rejected", wait)
        if ans in DECISIONS:
            return Approval(ans == d.decision, "ranger", ans, f"overridden to {ans}", wait)
        return Approval(False, "ranger", "look_again", f"unrecognised answer {ans[:20]!r}; holding with look_again", wait)


def make_approver(kind: str) -> Approver:
    return {"cli": CLIApprover, "auto": AutoApprover, "reject": lambda: AutoApprover(approve=False)}[kind]()


class Ranger:
    def __init__(self, autonomy: Autonomy = Autonomy.ADVISE, approver: Approver | None = None,
                 log_path: str | Path | None = None):
        self.autonomy = Autonomy(autonomy)
        self.approver = approver or AutoApprover()
        self.log_path = Path(log_path) if log_path else None
        self.pending: list[Decision] = []

    def review(self, d: Decision) -> Decision:
        """Sets d.action (what is executed) and d.approval, and appends a 'ranger' trace step."""
        if d.decision not in GATED:
            d.action = d.decision
            step = {"outcome": "pass-through", "autonomy": self.autonomy.value}
        elif self.autonomy is Autonomy.ACT:
            d.action = d.decision
            step = {"outcome": "executed (act mode, ranger notified)", "autonomy": "act"}
        else:
            self.pending.append(d)
            ap = self.approver(d)
            self.pending.remove(d)
            d.action = ap.final_action if ap.final_action in DECISIONS else "regroup"
            d.approval = {"approved": ap.approved, "by": ap.by, "final_action": d.action, "note": ap.note,
                          "wait_s": round(ap.wait_s, 3)}
            step = {"outcome": "approved" if ap.approved else f"overridden -> {d.action}", "autonomy": "advise",
                    "by": ap.by, "note": ap.note}
        d.trace.append({"i": len(d.trace), "kind": "ranger", **step})
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "point_id": d.point_id, "policy": d.policy,
                                    "decision": d.decision, "action": d.action, **step}) + "\n")
        return d
