"""Smooth, planned motion for the PiPER: a jerk-limited joint trajectory follower and a small planner.

The driver's "smooth" Cartesian mode streams joints (JointCtrl, firmware MOVE J) from this module
at CONTROL_HZ instead of handing Cartesian targets to the firmware's own IK (EndPoseCtrl, MOVE P).

  Follower        Ruckig (pantor/ruckig, MIT) online trajectory generation: one tick per control
                  period, retaining position, velocity and acceleration across every target change,
                  so a goal arriving at 10-20 Hz just moves the target and the motion stays
                  continuous. A longer path is followed through a one-DoF jerk-limited profile
                  along it, which the six-DoF Ruckig tracks. The Ruckig handling (keep q/v/a, brake through the velocity interface, never
                  replay missed ticks) is adapted from SUNNYsyy2005/bimanual-vla
                  bimanual_vla/deployment/servo.py (RuckigJointPlanner, ServoLoop).
  Planner         For a Cartesian goal: the straight line if it is feasible (IK along it, within
                  joint limits, no joint jumps, away from singularities, links above the table);
                  else a joint-space move to the goal's nearest IK solution (which backs up, turns
                  and extends by itself), checked by sampling FK; if that leaves the workspace box
                  or dips below the table, via points (raise, retract, tuck) are tried.
  CartesianJog    The panel's held-key jog: a Cartesian carrot that advances at the jog speed,
                  leashed to the commanded pose, solved by IK every tick.

Units: degrees and millimetres, like the driver.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import ruckig

import kinematics as K

CONTROL_HZ = 100          # JointCtrl stream rate (the vendor demos stream at 200 Hz, piper_ros at 200,
                          # bimanual-vla at 100, PiperXR at 50); 3 CAN frames per tick
# Per-joint limits at speed 100 %. Velocity scales with the speed setting; acceleration and jerk
# with max(speed, 30 %) so slow moves still start and stop crisply. The firmware's own MOVE J cap
# is the speed setting times its rated 3 rad/s (172 deg/s), so it stays above these at every speed
# and acts as a backstop; its default acceleration limit is 5 rad/s^2 (286 deg/s^2).
EDGE_DEG = 0.3            # paths are planned this far inside the joint limits
JOINT_VMAX = np.array([90.0, 90.0, 90.0, 120.0, 120.0, 120.0])        # deg/s
JOINT_AMAX = np.array([200.0, 200.0, 200.0, 300.0, 300.0, 300.0])     # deg/s^2
JOINT_JMAX = np.array([1000.0, 1000.0, 1000.0, 1500.0, 1500.0, 1500.0])  # deg/s^3
ACC_FLOOR = 0.3
# Stops (deadman, Stop, guard trips) brake with these whatever the speed setting.
BRAKE_AMAX = JOINT_AMAX
BRAKE_JMAX = JOINT_JMAX * 2.0
# Flange speed caps at speed 100 % (scaled like the joint velocities). Lines and jogs keep to
# them; joint-space plans may go TCP_JOINT_FACTOR times faster. At the default speed 15 the line
# cap is 30 mm/s, close to the 27 mm/s the firmware's streamed MOVE P gave.
TCP_MM_S = 200.0
TCP_DEG_S = 120.0
TCP_JOINT_FACTOR = 1.5

# Path following (Follower's profiled mode). The profile along a path uses this much of the
# binding joint's velocity limit, and of its acceleration and jerk limits (the rest is headroom for
# the path's curvature and for the tracker to catch up).
PROFILE_V_FRAC = 0.9
PROFILE_A_FRAC = 0.3
PROFILE_J_FRAC = 0.3
BRISK_A_FRAC = 0.7        # the jog's trail (set_path(brisk=True)): a quicker start, still within
BRISK_J_FRAC = 0.5        # 0.2 mm of the jog line in simulation
CURVE_A_FRAC = 0.25       # ... and the path speed is held where its curvature alone would ask more
                          # than this share of a joint's acceleration limit
SETTLE_DEG = 0.02
CORNER_DV_S = 0.03        # a via point is passed slowly enough that turning there changes each
                          # joint's velocity by at most 30 ms of its acceleration limit
TRACK_KP = 8.0            # 1/s: the tracker's position gain (16-20 oscillates with these limits)

# Planner.
LINE_STEP_MM = 5.0        # straight-line samples at most this far apart ...
LINE_STEP_DEG = 3.0       # ... and turning at most this much
LINE_MAX_STEP_DEG = 8.0   # a joint moving more than this between samples is a jump (IK flip)
MANIP_MIN = 0.02          # kinematics.manipulability: ~J5 3 deg, elbow straight, wrist on the J1 axis
JOINT_SAMPLE_DEG = 1.0    # joint-space paths are sampled this finely for the FK check
TABLE_Z_MM = 0.0          # the plane the arm stands on
LINK_CLEARANCE_MM = 30.0  # elbow and wrist centre stay this far above it
RAISE_MM = 80.0           # via point: lift this much above the higher end
RETRACT_MM = 200.0        # via point: pull the TCP in to this reach to turn
ORI_SLACK_DEG = 3.0       # a goal whose exact orientation is out of reach may give this much
REPLAN_FAILED_S = 0.5     # a goal that couldn't be reached is re-planned at most this often
DIRECT_MAX_DEG = 5.0      # a plan this short (largest joint change) is a streamed goal (teleop,
                          # LLM Arm steps): the trajectory heads straight for its end, which keeps
                          # 10-20 Hz goals continuous; longer plans are followed along their path

READY_POSE = [0.0, 45.0, -60.0, 0.0, 45.0, 0.0]


def motion_limits(speed: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(velocity, acceleration, jerk) per joint, deg-based, for a speed setting in percent."""
    s = min(1.0, max(0.01, float(speed) / 100.0))
    f = max(s, ACC_FLOOR)
    return JOINT_VMAX * s, JOINT_AMAX * f, JOINT_JMAX * f


def norm_inf(v: np.ndarray) -> float:
    return float(np.max(np.abs(v)))


def densify(waypoints: Sequence[np.ndarray], step_deg: float = JOINT_SAMPLE_DEG
            ) -> Tuple[np.ndarray, List[int]]:
    """A joint-space polyline sampled at most step_deg apart (any joint), and the sample index of
    each inner waypoint (the corners)."""
    out, corners = [np.asarray(waypoints[0], dtype=float)], []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = max(1, int(math.ceil(norm_inf(b - a) / step_deg)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
        corners.append(len(out) - 1)
    return np.array(out), corners[:-1]


# ---------------------------------------------------------------- follower
class Follower:
    """Jerk-limited joint trajectory follower: call step() once per control period.

    It follows a joint path one of two ways:
      direct    a path of one or two samples (a teleop increment, a slider, a stop): Ruckig's
                target is the end, at rest. A target that moves every 50 ms only moves Ruckig's
                target, and Ruckig re-plans from the current position, velocity and
                acceleration, so the motion doesn't stop between targets.
      profiled  a longer path (a Cartesian line, a joint-space plan, a jog trail): a one-DoF
                jerk-limited Ruckig profile along the path's length gives a reference point each
                tick, moving at the fastest speed the joint limits allow, and the six-DoF Ruckig
                tracks that reference (position, velocity, acceleration). So the path is followed
                closely at speed, and the command still keeps the velocity, acceleration and jerk
                limits whatever the reference does. Each via point ends one profile and starts the
                next, and is passed at a crawl, so corners are not cut.
    The state kept (q, v, a) is the COMMANDED trajectory, never the measured one: feeding measured
    velocity back each tick would break Ruckig's bounds (bimanual-vla's rule). Measured joints only
    anchor it when it starts (reset) and guard it from outside (the driver's tracking check)."""

    def __init__(self, hz: float = CONTROL_HZ):
        self.dt = 1.0 / hz
        self.otg = ruckig.Ruckig(K.DOF, self.dt)
        self.inp = ruckig.InputParameter(K.DOF)
        self.out = ruckig.OutputParameter(K.DOF)
        self.otg1 = ruckig.Ruckig(1)
        self.lock = threading.RLock()
        self.q: Optional[np.ndarray] = None
        self.v = np.zeros(K.DOF)
        self.a = np.zeros(K.DOF)
        self.path: Optional[np.ndarray] = None
        self.scale = 1.0
        self.stopping = False
        self.path_id = 0
        self.fault: Optional[str] = None
        self._prof: Optional[dict] = None
        self._last_ref = None        # the reference this tick followed: (q, velocity, acceleration)
        self.lo, self.hi = K.LIMITS_DEG[:, 0].copy(), K.LIMITS_DEG[:, 1].copy()
        self.vcap: Optional[np.ndarray] = None   # deg/s per joint at 100 %: the arm's own maxima (driver sets it)

    # -- state -------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.q is not None

    def moving(self) -> bool:
        return self.q is not None and (norm_inf(self.v) > 1e-3 or norm_inf(self.a) > 1e-2)

    def idle(self) -> bool:
        """At rest with nothing left to do (no path, or its end reached)."""
        with self.lock:
            if self.q is None:
                return True
            return not self.moving() and (self.path is None or norm_inf(self.path[-1] - self.q) < 1e-6)

    def reset(self, q: Sequence[float]) -> None:
        """Anchor at q (measured joints), at rest, holding there."""
        with self.lock:
            self.q = np.asarray(q, dtype=float).copy()
            self.v = np.zeros(K.DOF)
            self.a = np.zeros(K.DOF)
            # An arm resting outside the limits (J2 slightly below 0 when folded) may move inward.
            self.lo = np.minimum(K.LIMITS_DEG[:, 0], self.q)
            self.hi = np.maximum(K.LIMITS_DEG[:, 1], self.q)
            self._set_path(np.array([self.q]), [], 1.0)
            self.stopping = False
            self.fault = None
            self._load_state()

    def deactivate(self) -> None:
        with self.lock:
            self.q = None
            self.path = None
            self._prof = None
            self.stopping = False

    def anchor(self, measured: Sequence[float]) -> np.ndarray:
        """The commanded joints, anchoring at the measured ones first if idle."""
        with self.lock:
            if self.q is None:
                self.reset(measured)
            return self.q.copy()

    def snapshot(self) -> dict:
        with self.lock:
            P = self._prof
            progress = None
            if P is not None and P.get("ref") is not None:
                progress = round(float(P["ref"] / max(P["s"][-1], 1e-9)), 3)
            return {"active": self.q is not None, "moving": self.moving(), "stopping": self.stopping,
                    "q": None if self.q is None else [round(float(x), 3) for x in self.q],
                    "v": [round(float(x), 2) for x in self.v],
                    "path_len": 0 if self.path is None else len(self.path),
                    "profiled": P is not None, "progress": progress,
                    "path_id": self.path_id, "fault": self.fault}

    # -- targets -----------------------------------------------------------------
    def _set_path(self, path: np.ndarray, corners: Sequence[int], scale: float, brisk: bool = False) -> None:
        path = np.asarray(path, dtype=float).reshape(-1, K.DOF)
        # Plan strictly inside the limits, so tracking can't overshoot them and fault (which held the arm
        # until the next new target). A new path also clears an old fault.
        lo, hi = np.minimum(self.lo + EDGE_DEG, self.hi), np.maximum(self.hi - EDGE_DEG, self.lo)
        if self.q is not None:              # never move a resting arm just to get inside the margin
            lo, hi = np.minimum(lo, self.q), np.maximum(hi, self.q)
        path = np.clip(path, lo, hi)
        self.fault = None
        if len(path) > 1:                                   # drop repeated samples
            keep = np.r_[True, np.max(np.abs(np.diff(path, axis=0)), axis=1) > 1e-9]
            newidx = np.cumsum(keep) - 1
            corners = [int(newidx[c]) for c in corners if 0 < c < len(path)]
            path = path[keep]
        corners = sorted({c for c in corners if 0 < c < len(path) - 1})
        self.path = path
        self.scale = float(min(1.0, max(0.05, scale)))
        self.path_id += 1
        self._prof = None
        if len(path) >= 3:
            seg = np.diff(path, axis=0)
            ds = np.max(np.abs(seg), axis=1)
            sk = np.r_[0.0, np.cumsum(ds)]
            u = seg / ds[:, None]
            # Cubic Hermite through the samples (Catmull-Rom tangents), so the reference's velocity
            # has no kinks; one-sided tangents at via points and the ends keep corners exact.
            m = np.empty_like(path)
            m[0], m[-1] = u[0], u[-1]
            m[1:-1] = (path[2:] - path[:-2]) / (sk[2:] - sk[:-2])[:, None]
            m_start, m_end = m[:-1].copy(), m[1:].copy()
            for c in corners:
                m_end[c - 1], m_start[c] = u[c - 1], u[c]
            bounds = [0] + corners + [len(path) - 1]
            # Curvature: how fast the tangent turns per unit length (deg/deg), per joint; at path
            # speed v the turn alone needs an acceleration of curv * v^2.
            curv = np.max(np.abs(m_end - m_start) / ds[:, None], axis=0)
            self._prof = {"s": sk, "u": u, "m0": m_start, "m1": m_end, "curv": curv,
                          "um": np.max(np.abs(np.r_[u, m]), axis=0),
                          "segs": list(zip(bounds[:-1], bounds[1:])), "m": 0,
                          "traj": None, "t": 0.0, "key": None, "ref": None, "fresh": True, "done": False,
                          "brisk": bool(brisk)}

    def set_path(self, path: np.ndarray, corners: Sequence[int] = (), scale: float = 1.0,
                 brisk: bool = False, anchor: Optional[Sequence[float]] = None) -> int:
        """Follow a joint path (N x 6, deg) from wherever the trajectory is now. Returns its id.
        brisk: a path that is replaced every tick and only gently curved (the jog's trail) may
        use more of the acceleration and jerk limits. anchor: measured joints to start from if
        the follower was deactivated meanwhile (else that raises RuntimeError)."""
        with self.lock:
            if self.q is None:
                if anchor is None:
                    raise RuntimeError("follower is not anchored")
                self.reset(anchor)
            self._set_path(path, corners, scale, brisk)
            self.stopping = False
            return self.path_id

    def set_target(self, q: Sequence[float], scale: float = 1.0,
                   anchor: Optional[Sequence[float]] = None) -> int:
        """Go to one joint target (a path of one point)."""
        return self.set_path(np.array([np.asarray(q, dtype=float)]), (), scale, anchor=anchor)

    def stop(self) -> Optional[np.ndarray]:
        """Brake to a standstill as fast as the brake limits allow. Returns where it will stop."""
        with self.lock:
            if self.q is None:
                return None
            if not self.moving():
                self._set_path(np.array([self.q]), [], 1.0)
                return self.q.copy()
            stop_q = self._brake_point()
            self._set_path(np.array([stop_q]), [], 1.0)
            self.stopping = True
            return stop_q.copy()

    def _brake_point(self) -> np.ndarray:
        """Where a brake (velocity -> 0) from the current state ends."""
        inp = ruckig.InputParameter(K.DOF)
        inp.current_position = self.q.tolist()
        inp.current_velocity = self.v.tolist()
        inp.current_acceleration = self.a.tolist()
        inp.control_interface = ruckig.ControlInterface.Velocity
        inp.synchronization = ruckig.Synchronization.Phase
        inp.target_velocity = [0.0] * K.DOF
        inp.target_acceleration = [0.0] * K.DOF
        inp.max_velocity = (JOINT_VMAX * 2.0).tolist()
        inp.max_acceleration = BRAKE_AMAX.tolist()
        inp.max_jerk = BRAKE_JMAX.tolist()
        traj = ruckig.Trajectory(K.DOF)
        res = ruckig.Ruckig(K.DOF).calculate(inp, traj)
        if int(res) < 0:
            return self.q.copy()
        pos, _, _ = traj.at_time(traj.duration)
        return np.clip(np.asarray(pos, dtype=float), self.lo, self.hi)

    # -- the profile along a path ---------------------------------------------------
    def _interp(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Path point and its tangent dq/ds at arc length s (cubic Hermite)."""
        P = self._prof
        k = int(np.clip(np.searchsorted(P["s"], s, side="right") - 1, 0, len(P["u"]) - 1))
        h = P["s"][k + 1] - P["s"][k]
        t = min(max((s - P["s"][k]) / h, 0.0), 1.0)
        p0, p1, m0, m1 = self.path[k], self.path[k + 1], P["m0"][k], P["m1"][k]
        t2, t3 = t * t, t * t * t
        q = (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * h * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * h * m1
        dq = (6 * t2 - 6 * t) / h * (p0 - p1) + (3 * t2 - 4 * t + 1) * m0 + (3 * t2 - 2 * t) * m1
        return q, dq

    def _corner_speed(self, b: int, lims) -> float:
        """Path speed at via point b: slow enough that the turn's velocity change is small."""
        P = self._prof
        du = np.abs(P["u"][b] - P["u"][b - 1])
        _, amax, _ = self._tracker_limits
        return float(min(lims[0], np.min(CORNER_DV_S * amax / np.maximum(du, 1e-6))))

    def _plan_segment(self, s0: float, sd0: float, sdd0: float, lims) -> None:
        P = self._prof
        v_s, a_s, j_s = lims
        a, b = P["segs"][P["m"]]
        s_end = float(P["s"][b])
        v_end = 0.0 if P["m"] == len(P["segs"]) - 1 else self._corner_speed(b, lims)
        inp = ruckig.InputParameter(1)
        inp.current_position = [float(s0)]
        inp.current_velocity = [float(min(max(sd0, 0.0), v_s))]
        inp.current_acceleration = [float(min(max(sdd0, -a_s), a_s))]
        inp.target_position = [s_end]
        inp.target_velocity = [min(v_end, v_s)]
        inp.target_acceleration = [0.0]
        inp.max_velocity = [v_s]
        inp.max_acceleration = [a_s]
        inp.max_jerk = [j_s]
        traj = ruckig.Trajectory(1)
        try:
            res = self.otg1.calculate(inp, traj)
        except Exception:
            res = ruckig.Result.Error
        P["traj"] = traj if int(res) >= 0 else None
        P["t"] = 0.0
        P["lo"], P["hi"] = float(P["s"][a]), s_end

    def _profile_start(self, lims) -> None:
        """Start the profile on a new path. If the last tick was following a reference, carry that
        reference over (its point, projected onto the new path, and its path speed and
        acceleration), so a path replaced every tick (the jog trail, teleop goals) keeps its pace
        smoothly; otherwise start from the commanded joints and velocity."""
        P = self._prof
        carry = self._last_ref
        q = carry[0] if carry is not None else self.q
        k = int(np.argmin(np.max(np.abs(self.path - q), axis=1)))
        best = None
        for j in (k - 1, k):
            if 0 <= j < len(P["u"]):
                u = P["u"][j]
                sj = P["s"][j] + float(np.dot(q - self.path[j], u) / np.dot(u, u))
                sj = min(max(sj, P["s"][j]), P["s"][j + 1])
                d = norm_inf(self.path[j] + u * (sj - P["s"][j]) - q)
                if best is None or d < best[0]:
                    best = (d, sj, j)
        _, s0, j = best
        _, u = self._interp(s0)
        uu = float(np.dot(u, u))
        if carry is not None:
            sd0 = float(np.dot(carry[1], u)) / uu
            sdd0 = float(np.dot(carry[2], u)) / uu if sd0 > 0 else 0.0
        else:
            sd0, sdd0 = float(np.dot(self.v, u)) / uu, 0.0
        P["m"] = next((m for m, (a, b) in enumerate(P["segs"]) if s0 < P["s"][b] - 1e-9), len(P["segs"]) - 1)
        self._plan_segment(s0, sd0, sdd0, lims)

    def _profile_reference(self, vmax: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """The reference (position, velocity, acceleration) for the end of this tick."""
        P = self._prof
        um = np.maximum(P["um"], 1e-6)
        _, amax, jmax = self._tracker_limits
        v_curve = float(np.min(np.sqrt(CURVE_A_FRAC * amax / np.maximum(P["curv"], 1e-9))))
        af, jf = (BRISK_A_FRAC, BRISK_J_FRAC) if P["brisk"] else (PROFILE_A_FRAC, PROFILE_J_FRAC)
        lims = (min(float(np.min(vmax / um)) * PROFILE_V_FRAC, v_curve),
                float(np.min(amax / um)) * af, float(np.min(jmax / um)) * jf)
        key = tuple(round(x, 6) for x in lims)
        if P["fresh"]:
            self._profile_start(lims)
            P["fresh"], P["key"] = False, key
        elif key != P["key"] and P["traj"] is not None:      # the speed changed: re-plan from here
            s, sd, sdd = (x[0] for x in P["traj"].at_time(min(P["t"], P["traj"].duration)))
            self._plan_segment(s, sd, sdd, lims)
            P["key"] = key
        if P["traj"] is None:
            return None
        P["t"] += self.dt
        while P["t"] > P["traj"].duration and P["m"] < len(P["segs"]) - 1:
            carry = P["t"] - P["traj"].duration
            s_e, sd_e, _ = (x[0] for x in P["traj"].at_time(P["traj"].duration))
            P["m"] += 1
            self._plan_segment(s_e, sd_e, 0.0, lims)
            if P["traj"] is None:
                return None
            P["t"] = carry
        if P["t"] >= P["traj"].duration and P["m"] == len(P["segs"]) - 1:
            # The reference is at the end: servo onto it, then hand over to an exact stop.
            P["ref"] = float(P["s"][-1])
            if norm_inf(self.path[-1] - self.q) < SETTLE_DEG or SETTLE_DEG <= 0:
                P["done"] = True
                return None
            self._last_ref = (self.path[-1], np.zeros(K.DOF), np.zeros(K.DOF))
            return self._last_ref
        s, sd, sdd = (x[0] for x in P["traj"].at_time(min(P["t"], P["traj"].duration)))
        s = min(max(s, P["lo"]), P["hi"])
        P["ref"] = s
        q_ref, u = self._interp(s)
        self._last_ref = (q_ref, u * sd, u * sdd)
        return self._last_ref

    # -- the tick ----------------------------------------------------------------
    def _load_state(self) -> None:
        self.inp.current_position = self.q.tolist()
        self.inp.current_velocity = self.v.tolist()
        self.inp.current_acceleration = self.a.tolist()

    def hold_tick(self) -> Optional[np.ndarray]:
        """One tick with the trajectory's clock stopped: the command stays where it is (the arm is
        behind and catches up), and the next step() resumes from the same state."""
        with self.lock:
            return None if self.q is None else self.q.copy()

    def step(self, speed: float) -> Optional[np.ndarray]:
        """Advance one control period. Returns the joint command (deg), or None if not anchored."""
        with self.lock:
            if self.q is None:
                return None
            vmax, amax, jmax = motion_limits(speed)
            if self.vcap is not None:      # the arm's own joint speeds at this MOVE J percentage
                vmax = np.minimum(vmax, self.vcap * min(1.0, max(0.01, float(speed) / 100.0)))
            self._tracker_limits = (vmax, amax, jmax)
            ref = None
            if self.stopping:
                vmax, amax, jmax = np.maximum(vmax, JOINT_VMAX * 2.0), BRAKE_AMAX, BRAKE_JMAX
            elif self._prof is not None and not self._prof["done"]:
                ref = self._profile_reference(vmax * self.scale)
            if ref is None:
                self._last_ref = None
            if ref is not None:
                # Track the reference as a velocity servo: its velocity plus TRACK_KP times the
                # position error, reached through Ruckig's velocity interface, so the command keeps
                # the velocity, acceleration and jerk limits. (Asking Ruckig to hit the moving
                # reference's exact state every tick is not stable: it detours.)
                q_ref, v_ref, a_ref = ref
                self.inp.control_interface = ruckig.ControlInterface.Velocity
                self.inp.synchronization = ruckig.Synchronization.No
                self.inp.target_velocity = np.clip(v_ref + TRACK_KP * (np.clip(q_ref, self.lo, self.hi) - self.q),
                                                   -vmax, vmax).tolist()
                self.inp.target_acceleration = np.clip(a_ref, -amax, amax).tolist()
            else:
                if not self.stopping:
                    vmax = vmax * self.scale
                self.inp.control_interface = ruckig.ControlInterface.Position
                self.inp.synchronization = ruckig.Synchronization.Phase
                self.inp.target_position = self.path[-1].tolist()
                self.inp.target_velocity = [0.0] * K.DOF
                self.inp.target_acceleration = [0.0] * K.DOF
            self.inp.max_velocity = vmax.tolist()
            self.inp.max_acceleration = amax.tolist()
            self.inp.max_jerk = jmax.tolist()
            try:
                res = self.otg.update(self.inp, self.out)
            except Exception as e:             # invalid input: hold where we are
                res, self.fault = ruckig.Result.Error, f"ruckig: {e}"
            if int(res) < 0:
                self.fault = self.fault or f"ruckig returned {res}"
                self.v = np.zeros(K.DOF)
                self.a = np.zeros(K.DOF)
                self._set_path(np.array([self.q]), [], 1.0)
                self._load_state()
                return self.q.copy()
            q = np.asarray(self.out.new_position, dtype=float)
            if np.any(q < self.lo - 0.5) or np.any(q > self.hi + 0.5):
                self.fault = "trajectory left the joint limits; held"
                q = np.clip(q, self.lo, self.hi)
                self.q, self.v, self.a = q, np.zeros(K.DOF), np.zeros(K.DOF)
                self._set_path(np.array([q]), [], 1.0)
                self._load_state()
                return q.copy()
            self.q = q
            self.v = np.asarray(self.out.new_velocity, dtype=float)
            self.a = np.asarray(self.out.new_acceleration, dtype=float)
            self.out.pass_to_input(self.inp)
            if self.stopping and not self.moving():
                self.stopping = False
            return q.copy()


# ---------------------------------------------------------------- planner
@dataclass
class Workspace:
    """Where the flange may go (a box, mm) and the table the arm stands on."""
    box: dict
    table_z: float = TABLE_Z_MM
    clearance: float = LINK_CLEARANCE_MM

    def outside(self, p: np.ndarray) -> float:
        """How far (mm) a point is outside the box; 0 inside."""
        d = 0.0
        for i, key in enumerate(("x", "y", "z")):
            lo, hi = self.box[key]
            d = max(d, lo - p[i], p[i] - hi)
        return max(0.0, d)

    def link_low(self, pts: dict) -> float:
        """Height (mm) of the lower of elbow and wrist centre above the clearance plane."""
        return min(pts["elbow"][2], pts["wrist"][2]) - (self.table_z + self.clearance)


@dataclass
class Plan:
    kind: str                       # line | joint | via | partial | hold | blocked | unreachable
    path: Optional[np.ndarray] = None
    corners: List[int] = field(default_factory=list)
    scale: float = 1.0
    reason: str = ""
    goal: Optional[List[float]] = None
    vias: List[List[float]] = field(default_factory=list)
    ms: float = 0.0

    @property
    def moves(self) -> bool:
        return self.path is not None and len(self.path) > 0

    @property
    def direct(self) -> bool:
        """Short enough to head straight for the end (see DIRECT_MAX_DEG)."""
        return self.moves and not self.corners and norm_inf(self.path[-1] - self.path[0]) <= DIRECT_MAX_DEG

    def install(self, follower: "Follower", anchor: Optional[Sequence[float]] = None) -> int:
        """Hand the plan to a follower; returns the follower's path id."""
        if self.direct:
            return follower.set_target(self.path[-1], self.scale, anchor=anchor)
        return follower.set_path(self.path, self.corners, self.scale, anchor=anchor)

    def summary(self) -> dict:
        return {"kind": self.kind, "reason": self.reason, "samples": 0 if self.path is None else len(self.path),
                "vias": [[round(v, 1) for v in p] for p in self.vias], "speed_scale": round(self.scale, 3),
                "goal": None if self.goal is None else [round(v, 2) for v in self.goal],
                "ms": round(self.ms, 1), "time": time.time()}


class Planner:
    def __init__(self, kin: K.PiperKinematics, workspace: Workspace):
        self.kin = kin
        self.ws = workspace

    # -- checks ------------------------------------------------------------------
    def _tcp_scale(self, poses: Sequence[Tuple[np.ndarray, np.ndarray]], qs: np.ndarray, factor: float) -> float:
        """Velocity scale that keeps the flange under the speed caps along a path, if the joints
        ran at their own limits. Speed-independent: both scale with the speed setting."""
        worst = 0.0
        for k in range(1, len(qs)):
            dq = np.abs(qs[k] - qs[k - 1])
            t = float(np.max(dq / JOINT_VMAX))
            if t < 1e-9:
                continue
            dp = float(np.linalg.norm(poses[k][0] - poses[k - 1][0]))
            dth = K.angle_deg(poses[k - 1][1], poses[k][1])
            worst = max(worst, dp / t / (TCP_MM_S * factor), dth / t / (TCP_DEG_S * factor))
        return 1.0 if worst <= 1.0 else 1.0 / worst

    def check_joint_path(self, path: np.ndarray) -> Tuple[bool, str, float]:
        """Sample a joint path's FK: the flange inside the box and the elbow and wrist above the
        table, never worse than where the path starts. Returns (ok, why not, speed scale)."""
        poses, pts0 = [], self.kin.points(path[0])
        box_slack = self.ws.outside(pts0["flange"]) + 1.0
        low_slack = min(0.0, self.ws.link_low(pts0)) - 1.0
        for q in path:
            pts = self.kin.points(q)
            f = pts["flange"]
            if self.ws.outside(f) > box_slack:
                return False, (f"the TCP would leave the workspace box at ({f[0]:.0f}, {f[1]:.0f}, "
                               f"{f[2]:.0f}) mm"), 1.0
            if self.ws.link_low(pts) < low_slack:
                z = min(pts["elbow"][2], pts["wrist"][2])
                return False, f"the arm would dip to {z:.0f} mm, below the table clearance", 1.0
            poses.append((f, pts["R"]))
        return True, "", self._tcp_scale(poses, path, TCP_JOINT_FACTOR)

    def line(self, q0: np.ndarray, p0: np.ndarray, R0: np.ndarray, pg: np.ndarray, Rg: np.ndarray
             ) -> Tuple[Optional[np.ndarray], str, np.ndarray, float]:
        """IK along the straight line. Returns (path or None, why not, the feasible prefix, scale)."""
        dist, turn = float(np.linalg.norm(pg - p0)), K.angle_deg(R0, Rg)
        n = max(1, int(math.ceil(max(dist / LINE_STEP_MM, turn / LINE_STEP_DEG))))
        m_start = self.kin.manipulability(q0)
        m_min = min(MANIP_MIN, 0.9 * m_start)            # moving away from a singularity is fine
        low_slack = min(0.0, self.ws.link_low(self.kin.points(q0))) - 1.0
        qs, poses, q = [np.asarray(q0, dtype=float)], [(p0, R0)], np.asarray(q0, dtype=float)
        why = ""
        for k in range(1, n + 1):
            w = k / n
            p, R = K.interpolate_pose(p0, R0, pg, Rg, w)
            qk = self.kin.ik_rt(p, R, q, max_seed_dist_deg=LINE_MAX_STEP_DEG)
            if qk is None or k == n:
                # DLS stalled, or the end of the line (DLS stops within 0.5 mm): the exact
                # closed-form solution on the same branch.
                qa = self.kin.nearest_ik(K.rt_to_pose(p, R), q)
                if qa is not None and norm_inf(qa - q) <= LINE_MAX_STEP_DEG:
                    qk = qa
            where = f"{w * dist:.0f} mm along the line" if dist >= 1 else f"{w * turn:.0f} deg into the turn"
            if qk is None:
                why = f"no IK solution without a joint jump {where}"
                break
            if self.kin.manipulability(qk) < m_min:
                why = f"near a singularity {where} (J5 {qk[4]:.0f} deg, J3 {qk[2]:.0f} deg)"
                break
            if self.ws.link_low(self.kin.points(qk)) < low_slack:
                why = f"the arm would dip below the table clearance {where}"
                break
            qs.append(qk)
            poses.append((p, R))
            q = qk
        prefix = np.array(qs)
        if why:
            return None, why, prefix, self._tcp_scale(poses, prefix, 1.0)
        return prefix, "", prefix, self._tcp_scale(poses, prefix, 1.0)

    # -- goal IK -----------------------------------------------------------------
    def _cost(self, q: np.ndarray, q0: np.ndarray) -> float:
        """Roughly the time to get there, plus penalties near limits and singularities."""
        d = np.abs(q - q0) / JOINT_VMAX
        c = float(np.max(d)) + 0.2 * float(np.sum(d))
        lo, hi = K.joint_bounds(0.0)
        if np.any(q - lo < 5.0) or np.any(hi - q < 5.0):
            c += 0.5
        if self.kin.manipulability(q) < MANIP_MIN:
            c += 1.0
        return c

    def solve(self, q0: np.ndarray, pg: np.ndarray, Rg: np.ndarray) -> Optional[np.ndarray]:
        """The goal's IK solution nearest the current joints: every closed-form solution inside
        the limits, ranked by _cost (time to get there, near-limit and singularity penalties).
        If the exact orientation is out of reach, DLS lets it give ORI_SLACK_DEG."""
        q0 = np.asarray(q0, dtype=float)
        cands = self.kin.ik_all_rt(pg, Rg, q0)
        if not cands:
            wc = pg - K.FLANGE_MM * Rg[:, 2]
            az = math.degrees(math.atan2(wc[1], wc[0]))
            for s in (q0, np.r_[az, q0[1:]], np.r_[az, READY_POSE[1:]]):
                sol = self.kin.ik_rt(pg, Rg, s, max_seed_dist_deg=0.0, ori_slack_deg=ORI_SLACK_DEG)
                if sol is not None:
                    cands.append(sol)
        if not cands:
            return None
        return min(cands, key=lambda q: self._cost(q, q0))

    # -- plans -------------------------------------------------------------------
    def plan_pose(self, q0: Sequence[float], goal: Sequence[float]) -> Plan:
        """Plan from joints q0 to an end pose [x, y, z mm, rx, ry, rz deg]."""
        t0 = time.perf_counter()
        plan = self._plan_pose(np.asarray(q0, dtype=float), [float(v) for v in goal])
        plan.goal = [float(v) for v in goal]
        plan.ms = (time.perf_counter() - t0) * 1000.0
        return plan

    def _plan_pose(self, q0: np.ndarray, goal: List[float]) -> Plan:
        p0, R0 = self.kin.fk(q0)
        pg, Rg = K.pose_to_rt(goal)
        if float(np.linalg.norm(pg - p0)) < 0.2 and K.angle_deg(R0, Rg) < 0.1:
            return Plan("hold", np.array([q0]), reason="already there")
        path, why_line, prefix, scale = self.line(q0, p0, R0, pg, Rg)
        if path is not None:
            return Plan("line", path, scale=scale)
        qg = self.solve(q0, pg, Rg)
        if qg is None:
            return self._partial(prefix, p0, scale, f"{why_line}; the goal itself is out of reach")
        jp, _ = densify([q0, qg])
        ok, why_joint, jscale = self.check_joint_path(jp)
        if ok:
            return Plan("joint", jp, scale=jscale, reason=f"straight line: {why_line}")
        for name, vias in self._via_candidates(q0, p0, R0, qg, pg, Rg):
            vp, corners = densify([q0] + [q for q, _ in vias] + [qg])
            ok, _, vscale = self.check_joint_path(vp)
            if ok:
                return Plan("via", vp, corners, vscale, vias=[p for _, p in vias],
                            reason=f"straight line: {why_line}; joint space: {why_joint}; via {name}")
        return self._partial(prefix, p0, scale, f"{why_line}; joint space: {why_joint}; no via point helps",
                             kind="blocked")

    def _partial(self, prefix: np.ndarray, p0: np.ndarray, scale: float, why: str, kind: str = "unreachable") -> Plan:
        """As far along the straight line as it is feasible, or nothing."""
        if len(prefix) > 1:
            pe, _ = self.kin.fk(prefix[-1])
            if float(np.linalg.norm(pe - p0)) >= 2.0 or len(prefix) > 2:
                return Plan("partial", prefix, scale=scale, reason=why)
        return Plan(kind, None, reason=why)

    def _via_candidates(self, q0, p0, R0, qg, pg, Rg):
        """Via points to try, quickest first: lift, retract-turn-extend, and the READY tuck."""
        box = self.ws.box
        z_top = box["z"][1] - 20.0
        zr = min(z_top, max(p0[2], pg[2]) + RAISE_MM)
        az0, azg = math.atan2(p0[1], p0[0]), math.atan2(pg[1], pg[0])
        reach = min(math.hypot(p0[0], p0[1]), math.hypot(pg[0], pg[1]))

        def at(r, az, z):
            return np.array([r * math.cos(az), r * math.sin(az), z])

        cart = [("raise", [(np.r_[p0[:2], zr], R0)]),
                ("raise at both ends", [(np.r_[p0[:2], zr], R0), (np.r_[pg[:2], zr], Rg)])]
        for r in sorted({max(RETRACT_MM * 0.8, 0.7 * reach), RETRACT_MM}):
            for z in sorted({max(p0[2], pg[2]), zr}):
                cart.append((f"retract to {r:.0f} mm, turn, extend",
                             [(at(r, az0, z), R0), (at(r, azg, z), Rg)]))
        out = []
        for name, poses in cart:
            qs, q, ok = [], q0, True
            for p, R in poses:
                if self.ws.outside(p) > 0:
                    ok = False
                    break
                qv = self.kin.ik_rt(p, R, q, max_seed_dist_deg=0.0, ori_slack_deg=30.0)
                if qv is None:
                    ok = False
                    break
                qs.append((qv, K.rt_to_pose(p, R)))
                q = qv
            if ok:
                out.append((name, qs))
        tuck0 = np.array([q0[0]] + READY_POSE[1:])
        tuckg = np.array([qg[0]] + READY_POSE[1:])
        out.append(("the READY tuck", [(tuck0, self.kin.fk_pose(tuck0)), (tuckg, self.kin.fk_pose(tuckg))]))

        def duration(item):
            wps = [q0] + [q for q, _ in item[1]] + [qg]
            return sum(float(np.max(np.abs(b - a) / JOINT_VMAX)) for a, b in zip(wps[:-1], wps[1:]))
        return sorted(out, key=duration)

    def plan_joints(self, q0: Sequence[float], qg: Sequence[float]) -> Plan:
        """A joint-space move (HOME), with the same FK check and via points as a Cartesian goal.
        If nothing passes, the plain joint-space move is returned (what MOVE J always did)."""
        t0 = time.perf_counter()
        q0, qg = np.asarray(q0, dtype=float), np.asarray(qg, dtype=float)
        jp, _ = densify([q0, qg])
        ok, why, scale = self.check_joint_path(jp)
        plan = Plan("joint", jp, scale=1.0)
        if not ok:
            p0, R0 = self.kin.fk(q0)
            pg, Rg = self.kin.fk(qg)
            for name, vias in self._via_candidates(q0, p0, R0, qg, pg, Rg):
                vp, corners = densify([q0] + [q for q, _ in vias] + [qg])
                if self.check_joint_path(vp)[0]:
                    plan = Plan("via", vp, corners, 1.0, vias=[p for _, p in vias], reason=f"{why}; via {name}")
                    break
            else:
                plan.reason = f"{why} (no via point helps; moving in joint space anyway)"
        plan.goal = self.kin.fk_pose(qg)
        plan.ms = (time.perf_counter() - t0) * 1000.0
        return plan


# ---------------------------------------------------------------- jog
JOG_MM_S = TCP_MM_S       # at speed 100 %; 30 mm/s at the default 15
JOG_DEG_S = 90.0
JOG_LEAD_S = 0.4          # the carrot runs at most this long ahead of the commanded pose ...
JOG_LEAD_MM = 10.0        # ... but at least this far (mm) ...
JOG_LEAD_DEG = 5.0        # ... and this far (deg); releasing the key brakes, so the lead never overshoots


class CartesianJog:
    """Held-key Cartesian jog: a carrot pose that advances along the held axes at the jog speed,
    stays within a short leash of the commanded pose and inside the box, and is solved by IK
    (seeded at the last solution) every tick. The solutions form a trail the follower follows as a
    path, so the TCP stays on the jog line instead of cutting across to the carrot. Rotations add
    to the end pose's Euler angles, as the firmware jog does."""

    def __init__(self, kin: K.PiperKinematics, workspace: Workspace):
        self.kin, self.ws = kin, workspace
        self.carrot: Optional[np.ndarray] = None
        self.q: Optional[np.ndarray] = None
        self.trail: List[np.ndarray] = []
        self.blocked = ""

    def reset(self) -> None:
        self.carrot, self.q, self.trail, self.blocked = None, None, [], ""

    def step(self, q_cmd: np.ndarray, dirs: Sequence[float], speed: float, dt: float) -> np.ndarray:
        """The joint path for this tick: the carrot's trail of IK solutions along the jog line."""
        q_cmd = np.asarray(q_cmd, dtype=float)
        if self.carrot is None:
            self.trail = [q_cmd.copy()]
        self._advance(q_cmd, dirs, speed, dt)
        # Drop what the arm has passed, keeping one sample behind it so the trajectory's reference
        # (a little ahead of the commanded joints) always lies on the trail.
        k = int(np.argmin([norm_inf(t - q_cmd) for t in self.trail]))
        if k > 1:
            self.trail = self.trail[k - 1:]
        return np.array(self.trail)

    def _advance(self, q_cmd: np.ndarray, dirs: Sequence[float], speed: float, dt: float) -> None:
        s = min(1.0, max(0.01, speed / 100.0))
        lead_mm = max(JOG_LEAD_MM, JOG_LEAD_S * JOG_MM_S * s)
        lead_deg = max(JOG_LEAD_DEG, JOG_LEAD_S * JOG_DEG_S * s)
        actual = np.array(self.kin.fk_pose(q_cmd))
        if self.carrot is None:
            self.carrot, self.q = actual.copy(), np.asarray(q_cmd, dtype=float).copy()
        new = self.carrot.copy()
        R_act = K.rot_sxyz(*actual[3:])
        for ax in range(6):
            if not dirs[ax]:
                continue
            sign = 1.0 if dirs[ax] > 0 else -1.0
            if ax < 3:
                v = new[ax] + sign * JOG_MM_S * s * dt
                if abs(v - actual[ax]) > lead_mm and (v - actual[ax]) * sign > 0:
                    continue
                lo, hi = self.ws.box["xyz"[ax]]
                if (v > hi and sign > 0) or (v < lo and sign < 0):
                    continue
                new[ax] = v
            else:
                trial = new.copy()
                trial[ax] += sign * JOG_DEG_S * s * dt
                if K.angle_deg(R_act, K.rot_sxyz(*trial[3:])) > lead_deg and \
                        K.angle_deg(R_act, K.rot_sxyz(*trial[3:])) > K.angle_deg(R_act, K.rot_sxyz(*new[3:])):
                    continue
                new = trial
        if np.allclose(new, self.carrot):
            return
        qn = self.kin.ik(new.tolist(), self.q, max_seed_dist_deg=LINE_MAX_STEP_DEG)
        if qn is None:
            self.blocked = "no IK solution that way (joint limit or workspace edge)"
            return
        m_new = self.kin.manipulability(qn)
        if m_new < MANIP_MIN and m_new < self.kin.manipulability(self.q):
            self.blocked = f"singularity ahead (J5 {qn[4]:.0f} deg)"
            return
        if self.ws.link_low(self.kin.points(qn)) < min(0.0, self.ws.link_low(self.kin.points(self.q))):
            self.blocked = "the arm would dip below the table clearance"
            return
        self.blocked = ""
        self.carrot, self.q = new, qn
        self.trail.append(qn)
