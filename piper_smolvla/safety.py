"""What stands between the policy's action and the arm.

Every action the policy proposes passes through SafetyFilter before it reaches the driver. The rules are the
ACT engine's (act/engine/nervus_act/runner.py in NervusOS, same author), fixed on the real arm in the Piper-ACT
project, plus a tracking-error stop taken from piper-dual-arm-smolvla-towel-folding (Apache-2.0):

* Targets are clamped to the range the demonstrations covered, plus a margin (degrees; millimetres for the
  gripper), and to the joint limits the driver reports.
* Step limit on the COMMAND, not on the measured position: the gripper answers with 0.5-0.8 s of dead time,
  and limiting from the measurement pinned it half-open. The limit is a multiple of the demonstrations' 99th
  percentile per-step change (three times that for the gripper).
* A lead limit caps how far a joint command may run ahead of the measured arm, so a blocked or slow arm is
  never handed a target far away.
* The grasp latch: once the fingers have stopped on an object a little wider than the command, the command
  stays at "measured minus squeeze", ignoring single-step flickers to open or close fully; letting go takes
  `gripper_release_steps` open requests in a row. A few mm of squeeze gives grip force (commanding exactly the
  contact width asks for zero force).
* Tracking stop (towel-folding repo): if the measured arm is further than `tracking_stop_deg` from the last
  command, something is wrong (a collision, the arm being pushed, a fault), and the run stops.
* A non-finite action stops the run.

The facts the limits scale with (envelope, per-step 99th percentiles, start pose) are computed from the
training data when a policy is trained (data.facts) and published with it in nervus_policy.json. Without them,
the runner's fallbacks apply: the driver's joint limits as the envelope and 4 degrees as every joint's p99.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .clients import JOINTS

FALLBACK_STEP_P99 = 4.0          # deg per step, the ACT runner's fallback
FALLBACK_GRIPPER_RANGE = (0.0, 70.0)
FALLBACK_JOINT_LIMITS = {"joint1": (-154.0, 154.0), "joint2": (0.0, 195.0), "joint3": (-175.0, 0.0),
                         "joint4": (-106.0, 106.0), "joint5": (-75.0, 75.0), "joint6": (-100.0, 100.0),
                         "gripper": (0.0, 95.0)}


class UnsafeAction(RuntimeError):
    """The run must stop: the action, or the arm's response to the last one, is not safe to continue from."""


@dataclass
class SafetyParams:
    step_limit_x: float = 1.5         # max per-step change of the command, x the demos' p99 (0 = off)
    lead_limit_x: float = 4.0         # max joint command lead over the measured arm, x the demo p99 (0 = off)
    envelope_margin: float = 5.0      # targets clamped to the demonstrated range +/- this (deg; mm for the gripper)
    tracking_stop_deg: float = 30.0   # stop if a joint is further than this from the last command (0 = off)
    gripper_squeeze_mm: float = 1.5   # commanded this much tighter than the measured width while holding
    gripper_release_steps: int = 8    # while holding, open requests needed in a row to let go (0 = off)


def load_facts(path: str | Path | None) -> dict | None:
    """nervus_policy.json from a policy directory (or the file itself); None if there is none."""
    if not path:
        return None
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / "nervus_policy.json"
    try:
        meta = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    facts = {k: meta.get(k) for k in ("envelope", "step_p99", "start_pose", "cameras", "fps", "task", "tasks")}
    return facts if facts.get("envelope") and facts.get("step_p99") else None


# Adapted from izhangfly/NervusOS-Cambridge/act/engine/nervus_act/runner.py (Runner._load_facts fallbacks).
def complete_facts(facts: dict | None, joint_limits: dict | None) -> dict:
    """Facts with every gap filled the way the ACT runner fills them."""
    lim = dict(FALLBACK_JOINT_LIMITS)
    lim.update(joint_limits or {})
    out = dict(facts or {})
    env = dict(out.get("envelope") or {})
    for j in JOINTS[:6]:
        env.setdefault(j, list(lim[j]))
    env.setdefault("gripper", list(FALLBACK_GRIPPER_RANGE))
    out["envelope"] = env
    p99 = dict(out.get("step_p99") or {})
    for j in JOINTS:
        p99.setdefault(j, FALLBACK_STEP_P99)
    out["step_p99"] = p99
    out["joint_limits"] = lim
    out["from_training_data"] = bool(facts and facts.get("envelope"))
    return out


class SafetyFilter:
    def __init__(self, facts: dict, params: SafetyParams | None = None):
        self.facts = facts
        self.p = params or SafetyParams()
        self.last: dict | None = None      # the last command that was sent
        self.grip: dict = {}
        self.counts = {"steps": 0, "envelope": 0, "joint_limit": 0, "step": 0, "lead": 0, "grip_hold": 0,
                       "grip_release": 0}

    def reset(self):
        self.last, self.grip = None, {}

    def gripper_max(self) -> float:
        return float(self.facts["envelope"].get("gripper", FALLBACK_GRIPPER_RANGE)[1] or FALLBACK_GRIPPER_RANGE[1])

    # Adapted from izhangfly/NervusOS-Cambridge/act/engine/nervus_act/runner.py (Runner._clamp).
    def clamp(self, target: dict, current: dict | None, last: dict | None) -> dict:
        p, env, p99, lim = self.p, self.facts["envelope"], self.facts["step_p99"], self.facts["joint_limits"]
        out = {}
        for j in JOINTS:
            v = float(target[j])
            lo, hi = env[j]
            c = min(hi + p.envelope_margin, max(lo - p.envelope_margin, v))
            if c != v:
                self.counts["envelope"] += 1
            v = c
            jl = lim.get(j)
            if jl:
                c = min(jl[1], max(jl[0], v))
                if c != v:
                    self.counts["joint_limit"] += 1
                v = c
            if p.step_limit_x > 0 and last is not None:
                step = max(0.2, p99[j]) * p.step_limit_x * (3.0 if j == "gripper" else 1.0)
                c = min(last[j] + step, max(last[j] - step, v))
                if c != v:
                    self.counts["step"] += 1
                v = c
            if j != "gripper" and p.lead_limit_x > 0 and current is not None:
                lead = max(0.2, p99[j]) * p.lead_limit_x
                c = min(current[j] + lead, max(current[j] - lead, v))
                if c != v:
                    self.counts["lead"] += 1
                v = c
            out[j] = v
        out["gripper"] = max(0.0, out["gripper"])
        return out

    # Adapted from izhangfly/NervusOS-Cambridge/act/engine/nervus_act/runner.py (Runner._gripper).
    def gripper(self, plan_g: float, meas_g: float) -> tuple[float, str]:
        """Millimetres; the thresholds scale with the gripper's demonstrated range, so a different gripper or
        object works the same way."""
        p, gmax, g = self.p, self.gripper_max(), self.grip
        sq = p.gripper_squeeze_mm
        last = g.get("last")
        hist = g.setdefault("hist", deque(maxlen=6))
        hist.append(meas_g)
        stable = len(hist) == hist.maxlen and max(hist) - min(hist) <= 0.7
        holding = (last is not None and stable and 0.25 * gmax <= meas_g <= 0.92 * gmax
                   and 1.0 <= meas_g - last <= 8.0)
        if holding and p.gripper_release_steps > 0:
            g["open_count"] = g.get("open_count", 0) + 1 if plan_g >= meas_g + 5.5 else 0
            if g["open_count"] < p.gripper_release_steps:
                cmd = max(0.0, meas_g - sq)
                g["last"] = cmd
                self.counts["grip_hold"] += 1
                return cmd, "hold"
            g["open_count"] = 0
            g["last"] = plan_g
            self.counts["grip_release"] += 1
            return plan_g, "release"
        g["open_count"] = 0
        cmd = plan_g if plan_g >= 0.95 * gmax else max(0.0, plan_g - sq)
        g["last"] = cmd
        return cmd, "follow"

    # ---- one step ----
    def __call__(self, raw: dict, measured: dict) -> tuple[dict, str]:
        """(the command to send, the gripper mode). Raises UnsafeAction when the run must stop."""
        for j in JOINTS:
            v = raw.get(j)
            if v is None or not math.isfinite(float(v)):
                raise UnsafeAction(f"the policy proposed a non-finite {j}: {v!r}")
        # Adapted from dk2472780158-ctrl/piper-dual-arm-smolvla-towel-folding/smolvla_piper_towel/scripts/
        # smolvla_piper_execute_safe.py (tracking-error safety stop; limit_command steps from the previous command).
        if self.p.tracking_stop_deg > 0 and self.last is not None:
            worst = max(JOINTS[:6], key=lambda j: abs(measured[j] - self.last[j]))
            err = abs(measured[worst] - self.last[worst])
            if err > self.p.tracking_stop_deg:
                raise UnsafeAction(f"{worst} is {err:.0f} deg from its last command (limit "
                                   f"{self.p.tracking_stop_deg:g}): blocked, pushed or faulted")
        # The first command of a run is step-limited from where the arm is (the runner let it jump by the lead).
        tgt = self.clamp(raw, measured, self.last if self.last is not None else measured)
        tgt["gripper"], mode = self.gripper(tgt["gripper"], measured["gripper"])
        self.last = tgt
        self.counts["steps"] += 1
        return tgt, mode

    def hold_target(self, measured: dict) -> dict:
        """Freeze where the arm IS, with a light grip from the measured width (runner: end of a run)."""
        hold = dict(measured)
        if hold["gripper"] < 0.95 * self.gripper_max():
            hold["gripper"] = max(0.0, hold["gripper"] - self.p.gripper_squeeze_mm)
        return hold


PREVIEW_GOOD, PREVIEW_WEAK = 20.0, 8.0   # degrees the model plans to move within one chunk (ACT runner)


# Adapted from izhangfly/NervusOS-Cambridge/act/engine/nervus_act/runner.py (Runner.preview verdicts).
def preview_verdict(planned_deg: float) -> str:
    """The ACT runner's scene check: a policy that plans to sit still is seeing a scene unlike its
    demonstrations (different light, a moved camera)."""
    if planned_deg >= PREVIEW_GOOD:
        return "plans to move"
    if planned_deg >= PREVIEW_WEAK:
        return "weak: may hesitate"
    return "would likely sit still: the scene or light differs from the demonstrations"
