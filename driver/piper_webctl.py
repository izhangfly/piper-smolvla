#!/usr/bin/env python3
"""
PiPER arm driver: the NervusOS hardware layer for the AgileX PiPER.

It is the only program that touches the CAN bus (can0). Everything else talks to
it over a small HTTP API on loopback:

    Robot Arm app        sliders, Cartesian jog, gripper      (client "panel")
    nervus.piperd        Manipulator@1 Provider; ServoToPose  (client "piperd")
    ACT engine           joint streaming for learned policies (client "act")
    browser panel        http://127.0.0.1:8770                (client "panel")

On the Ubuntu host it runs as nervus-piper.service, starts and stops together
with NervusOS, and listens on loopback only, so arm control is never exposed to
the network. To use the browser panel from another machine, tunnel it:

    ssh -L 8770:localhost:8770 <ubuntu-host>

Safety design
  * Motors start in whatever state the arm is in; nothing moves until a client
    enables them and sends a target. Stopping this service never disables the
    motors: a disabled PiPER drops under gravity.
  * Every motion source has a deadman. The panel and Robot Arm app send a
    heartbeat (1.5 s); servo targets expire after 0.4 s and joint streams after
    0.5 s. When it expires the arm holds where it is.
  * One controller at a time. A client that moves the arm owns it until its
    deadman lapses or it releases it; others get HTTP 409 until then. Stop and
    E-stop are always accepted from anyone.
  * Targets are clamped server-side: joints to the manual's limits, Cartesian
    targets to a conservative workspace box, servo steps to a short leash.
  * /api/state reports how old the arm's feedback is. Feedback that is not
    flowing is reported as stale, never as a valid pose: the SDK returns its
    cached values forever, and cached zeros look like a plausible pose.

Cartesian control (--cartesian smooth, the default)
  The driver does its own IK and streams joints (JointCtrl, MOVE J) at 100 Hz from
  a jerk-limited trajectory (Ruckig), so the firmware's IK is out of the loop.
  Each Cartesian goal is planned (motion.py): the straight line if it's feasible,
  else a joint-space move to the nearest IK solution, with via points when that
  would leave the workspace box or dip below the table. Goals arriving at 10-20 Hz
  only move the trajectory's target, so motion stays continuous. Joint moves
  (sliders, HOME) go through the same trajectory. ACT streaming (/api/stream) is
  unchanged. --cartesian firmware is the previous behaviour: EndPoseCtrl (MOVE P)
  at 20 Hz to a target leashed a few millimetres ahead of the measured pose.

Hard-won arm facts (see README.md): joint zero is a kinematic singularity where
only Z can move, so HOME goes to the READY pose instead; streamed MOVE P is
fastest with the target about one control period ahead, so in firmware mode
Cartesian targets lead the measured pose by a few millimetres rather than
jumping to the goal.

Units: degrees and millimetres throughout (the arm speaks 0.001 deg / mm).

    python3 piper_webctl.py [--bind 127.0.0.1] [--port 8770] [--iface can0] [--speed 15]
                            [--cartesian smooth|firmware]
"""

import argparse
import json
import math
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    raise SystemExit("piper_sdk not importable; use the venv at /opt/nervus/piper/venv")

HERE = os.path.dirname(os.path.abspath(__file__))
# The smooth Cartesian mode needs numpy and ruckig (requirements.txt). Without them the
# driver still runs, in firmware mode, and says why in /api/state.
try:
    import sys as _sys
    if HERE not in _sys.path:
        _sys.path.insert(0, HERE)
    import numpy as np
    import kinematics as KIN
    import motion as MOTION
    SMOOTH_UNAVAILABLE = None
except ImportError as _e:          # pragma: no cover - depends on the venv
    np = KIN = MOTION = None
    SMOOTH_UNAVAILABLE = f"{_e}; pip install -r requirements.txt"
MDEG = 1000.0
GRIP_UNIT = 1000.0
JOINT_LIMITS = {1: (-154.0, 154.0), 2: (0.0, 195.0), 3: (-175.0, 0.0),
                4: (-106.0, 106.0), 5: (-75.0, 75.0), 6: (-100.0, 100.0)}
# One set of limits everywhere: the sliders, the clamps and the smooth follower. The follower plans within
# the stricter of these and piper_sdk's (kinematics.LIMITS_DEG); a slider allowed past those made the
# trajectory run out of bounds, fault and freeze (J3 to -175, J5 to 75: "stuck until I shake it").
if KIN is not None:
    JOINT_LIMITS = {j + 1: (max(lo, float(KIN.LIMITS_DEG[j][0])), min(hi, float(KIN.LIMITS_DEG[j][1])))
                    for j, (lo, hi) in enumerate(JOINT_LIMITS.values())}
# Commanded gripper opening. The jaws measure up to ~101 mm when pulled open by
# hand, but 95 mm is the widest opening that was commanded reliably.
GRIPPER_MM = (0.0, 95.0)
GRIPPER_EFFORT = (0, 5000)          # 0.001 N*m
GRIPPER_EFFORT_DEFAULT = 1000
# Elbow bent, wrist clear of the +-90 deg Euler singularity: all six Cartesian
# axes can jog from here (README section 4). Joint zero is NOT a usable home.
READY_POSE = [0.0, 45.0, -60.0, 0.0, 45.0, 0.0]
# Conservative Cartesian box for servo targets (mm). The arm enforces its true,
# joint-limited workspace on top of this.
WORKSPACE_MM = {"x": (60.0, 480.0), "y": (-320.0, 320.0), "z": (80.0, 600.0)}
# move_spd_rate_ctrl (CAN 0x151) is a percentage of the arm's RATED speed.
# Floor is 1, not 0: zero stalls motion mid-travel.
SPEED_MIN, SPEED_MAX = 1, 100
STREAM_HZ = 20
HEARTBEAT_TIMEOUT = 1.5     # panel / Robot Arm app
SERVO_DEADMAN = 0.4         # piperd ServoToPose
STREAM_DEADMAN = 0.5        # ACT joint streaming
FEEDBACK_STALE = 0.3        # joint feedback older than this is not a pose

# Leashed Cartesian control: the streamed TCP target is the MEASURED pose plus
# a short lead toward where we want to go, re-derived every tick. It can never
# run away from the arm, and a blocked axis doesn't freeze the others.
JOG_LEAD_MM = 6.0
JOG_LEAD_DEG = 5.0
SERVO_LEAD_MM = 6.0
SERVO_LEAD_DEG = 5.0
POSE_CLAMP_MM = 900.0
POSE_CLAMP_DEG = 185.0
JOINT_JOG_STEP = 0.6        # deg per tick while a wrist key is held (~12 deg/s)

# IK flip guard for the Cartesian modes. The arm solves IK itself, and near the
# wrist singularity (J5 near 0) a small TCP step can make it swing J4 and J6 half a
# turn in opposite directions while the TCP barely moves. The leash can't see
# that: it only limits the TCP. So the loop watches the joints too, over a short
# window, and holds when one joint swings far more than the TCP motion explains.
IK_GUARD_DEG = 45.0         # joint swing in the window that can trip (0 = off)
IK_GUARD_WINDOW = 0.5       # s
IK_GUARD_RATIO = 3.0        # trip if swing > ratio x (TCP turn deg + TCP travel mm / 2)
IK_GUARD_LATCH = 2.0        # s that servo goals are refused after a trip

# Smooth mode. The trajectory limits, rates and planner settings are in motion.py.
TRACK_ERR_DEG = 20.0        # measured joint this far from its command: something moved it; hold at once
TRACK_STALL_S = 1.0         # more than LAG_PAUSE_DEG behind for this long without closing the gap
                            # by TRACK_PROGRESS_DEG: blocked; hold
TRACK_PROGRESS_DEG = 1.0
LAG_PAUSE_DEG = 6.0         # the trajectory waits while the arm is this far behind it
FK_TOL_MM = 15.0            # our FK of the measured joints must match the arm's own pose this well
JOINT_JOG_DEG_S = 12.0      # a held wrist key (firmware mode: 0.6 deg per 20 Hz tick)
JJOG_LEAD_DEG = 3.0         # ... whose target stays this close to the commanded joint

CTRL_MODES = {0x00: "STANDBY", 0x01: "CAN_CTRL", 0x02: "TEACHING_MODE",
              0x03: "ETHERNET_CTRL", 0x04: "WIFI_CTRL", 0x05: "REMOTE_CTRL",
              0x06: "LINKAGE_TEACHING_INPUT", 0x07: "OFFLINE_TRAJECTORY"}
ARM_STATUS = {0x00: "NORMAL", 0x01: "EMERGENCY_STOP", 0x02: "NO_SOLUTION",
              0x03: "SINGULARITY_POINT", 0x04: "TARGET_POS_EXCEEDS_LIMIT",
              0x05: "JOINT_COMMUNICATION_EXCEPTION", 0x06: "JOINT_BRAKE_NOT_RELEASED",
              0x07: "COLLISION_DETECTED", 0x08: "TEACHING_OVERSPEED",
              0x09: "JOINT_STATUS_ABNORMAL", 0x0A: "OTHER_EXCEPTION",
              0x0B: "TEACHING_RECORD", 0x0C: "TEACHING_EXECUTION",
              0x0D: "TEACHING_PAUSE", 0x0E: "MAIN_CONTROL_NTC_OVERHEAT",
              0x0F: "RELEASE_RESISTOR_NTC_OVERHEAT"}
# Soft faults clear themselves ~1.3 s after you stop pushing; only hard faults
# need Recover (README section 4, fault taxonomy).
SOFT_FAULTS = {0x02, 0x03, 0x04}

# The panel and app logs are written from HTTP threads and the control loop, and
# log_call() is called while state is being changed. A plain Lock here once
# deadlocked the whole server (README section 5), so it is reentrant.
state_lock = threading.RLock()
io_lock = threading.Lock()
STATE = {
    "targets": [0.0] * 6,
    "mode": "joint",          # joint | cartesian | servo | stream
    "pose": [0.0] * 6,        # last streamed TCP target (mm, deg), for display
    "jog": [0.0] * 6,         # per-axis held-key lead: +lead / -lead / 0
    "jjog": [0.0] * 6,        # per-joint held-key step (deg/tick)
    "servo_goal": None,       # absolute TCP goal from the servo client
    "gripper_mm": 0.0,
    "gripper_effort": GRIPPER_EFFORT_DEFAULT,
    "enabled": False,
    "estopped": False,
    "last_heartbeat": 0.0,
    "speed": 15,
    "owner": None,            # {"client", "kind", "last"}
    "guard": None,            # last IK guard trip: {"time", "reason", "client"}
    "plan": None,             # smooth mode: the last Cartesian plan (motion.Plan.summary())
    "log": [],
}


class Busy(Exception):
    def __init__(self, owner):
        super().__init__(f"arm is controlled by {owner}")
        self.owner = owner


def log_call(fn, params):
    entry = {"t": time.strftime("%H:%M:%S"), "fn": fn, "params": params}
    if fn.startswith("("):
        # Guards, deadmen and trajectory faults also go to the journal (journalctl -u nervus-piper):
        # the in-memory log is lost when the driver restarts.
        print(f"{fn} {json.dumps(params)}", flush=True)
    with state_lock:
        STATE["log"].insert(0, entry)
        del STATE["log"][40:]


def clamp(j, deg):
    lo, hi = JOINT_LIMITS[j]
    return max(lo, min(hi, deg))


def rot_zyx(rx, ry, rz):
    """Rotation matrix of a PiPER end pose's RX, RY, RZ (deg): R = Rz Ry Rx."""
    a, b, c = (math.radians(v) for v in (rx, ry, rz))
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    return [[cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
            [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
            [-sb, cb * sa, cb * ca]]


def turn_deg(p, q):
    """Angle (deg) of the rotation between two end poses, whatever the Euler wrap."""
    r, s = rot_zyx(*p[3:6]), rot_zyx(*q[3:6])
    tr = sum(r[i][k] * s[i][k] for i in range(3) for k in range(3))   # trace(R^T S)
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


class FlipGuard:
    """Spots the arm's IK swinging joints that the TCP motion doesn't explain.

    Feed it (time, joints, pose) every control tick while a Cartesian mode streams;
    reset() it whenever streaming stops or the mode changes, so a joint move in
    between is never blamed on IK."""

    def __init__(self, limit_deg=IK_GUARD_DEG, window=IK_GUARD_WINDOW, ratio=IK_GUARD_RATIO):
        self.limit, self.window, self.ratio = limit_deg, window, ratio
        self.samples = []

    def reset(self):
        self.samples = []

    def update(self, t, joints, pose):
        """Returns why it tripped, or None."""
        if self.limit <= 0:
            return None
        self.samples.append((t, list(joints), list(pose)))
        while self.samples and t - self.samples[0][0] > self.window:
            self.samples.pop(0)
        t0, j0, p0 = self.samples[0]
        swing = [abs(joints[i] - j0[i]) for i in range(6)]
        worst = max(range(6), key=lambda i: swing[i])
        if swing[worst] < self.limit:
            return None
        travel = math.dist(pose[:3], p0[:3])
        turn = turn_deg(p0, pose)
        if swing[worst] <= self.ratio * (turn + travel / 2.0):
            return None
        return (f"J{worst + 1} swung {swing[worst]:.0f} deg in {t - t0:.2f} s while the TCP moved "
                f"{travel:.0f} mm and turned {turn:.0f} deg (IK flip near a singularity?)")


# The deadman a client's gripper commands ride on: the same as its motion source.
CLIENT_KIND = {"piperd": "servo", "act": "stream"}


def deadman_for(kind):
    return {"servo": SERVO_DEADMAN, "stream": STREAM_DEADMAN}.get(kind, HEARTBEAT_TIMEOUT)


def owner_fresh(now=None):
    """The current owner, or None if nobody holds the arm or their deadman lapsed."""
    now = now or time.time()
    o = STATE["owner"]
    if o and now - o["last"] < deadman_for(o["kind"]):
        return o
    return None


def claim(client, kind):
    """Take or refresh ownership for `client`. Raises Busy if someone else holds it."""
    with state_lock:
        now = time.time()
        o = owner_fresh(now)
        if o and o["client"] != client:
            raise Busy(o["client"])
        STATE["owner"] = {"client": client, "kind": kind, "last": now}
        if kind in ("panel",):
            STATE["last_heartbeat"] = now


class Arm:
    def __init__(self, iface, speed, cartesian="smooth", workspace=None, track_err_deg=TRACK_ERR_DEG):
        self.piper = C_PiperInterface_V2(iface)
        self.piper.ConnectPort()
        for _ in range(50):
            if self.piper.isOk():
                break
            time.sleep(0.1)
        if not self.piper.isOk():
            raise SystemExit(f"CAN thread unhealthy on {iface}")
        time.sleep(0.5)
        self.speed = speed
        self.move_mode = 0x01      # last MotionCtrl_2 move mode sent (0x00 P, 0x01 J)
        self.firmware = None
        self._last_grip = (None, None, 0.0)
        self.guard = FlipGuard()
        self._guard_mode = None    # the mode the guard's samples belong to
        self._stop = threading.Event()
        # Cartesian mode: "smooth" (our IK + planner + Ruckig, JointCtrl at 100 Hz) or
        # "firmware" (EndPoseCtrl leashed at 20 Hz, the firmware's IK).
        self.cartesian_note = None
        if cartesian == "smooth" and SMOOTH_UNAVAILABLE:
            self.cartesian_note = f"smooth mode unavailable ({SMOOTH_UNAVAILABLE}); using firmware"
            print(self.cartesian_note, flush=True)
            cartesian = "firmware"
        self.cartesian = cartesian
        self.smooth = cartesian == "smooth"
        self.track_err_deg = track_err_deg
        if self.smooth:
            self.kin = KIN.PiperKinematics(0x01)
            self.workspace = MOTION.Workspace(dict(workspace or WORKSPACE_MM))
            self.planner = MOTION.Planner(self.kin, self.workspace)
            self.jogger = MOTION.CartesianJog(self.kin, self.workspace)
            self.follower = MOTION.Follower(MOTION.CONTROL_HZ)
            self.plan_lock = threading.Lock()
            self._last_plan = None          # (plan, follower path id, time)
            self._applied_targets = None    # joint targets the follower was last given
            self.fk_error_mm = None
            self._dh_selected = False
            self.overruns = 0
            self._jog_note_t = 0.0
            threading.Thread(target=self._loop_smooth, daemon=True, name="control").start()
        else:
            threading.Thread(target=self._loop, daemon=True, name="control").start()

    # ---------------------------------------------------------------- feedback
    def feedback_age(self):
        try:
            ts = self.piper.GetArmJointMsgs().time_stamp
        except Exception:
            return math.inf
        return time.time() - ts if ts else math.inf

    def joints(self):
        j = self.piper.GetArmJointMsgs().joint_state
        return [getattr(j, f"joint_{n}") / MDEG for n in range(1, 7)]

    def end_pose(self):
        e = self.piper.GetArmEndPoseMsgs().end_pose
        return [e.X_axis / 1000.0, e.Y_axis / 1000.0, e.Z_axis / 1000.0,
                e.RX_axis / 1000.0, e.RY_axis / 1000.0, e.RZ_axis / 1000.0]

    def status(self):
        """Controller status, decoded. Values from a feedback stream that has stopped
        are not trusted (STANDBY doesn't broadcast; see README)."""
        out = {}
        try:
            msg = self.piper.GetArmStatus()
            s = msg.arm_status
            out["age_s"] = time.time() - msg.time_stamp if msg.time_stamp else math.inf
            out["ctrl_mode_code"] = int(s.ctrl_mode)
            out["ctrl_mode"] = CTRL_MODES.get(int(s.ctrl_mode), hex(int(s.ctrl_mode)))
            out["arm_status_code"] = int(s.arm_status)
            out["arm_status"] = ARM_STATUS.get(int(s.arm_status), hex(int(s.arm_status)))
            out["teach_status"] = int(s.teach_status)
            out["motion_status"] = int(s.motion_status)
            out["mode_feed"] = int(s.mode_feed)
            es = s.err_status
            out["joint_limit"] = [bool(getattr(es, f"joint_{n}_angle_limit", False)) for n in range(1, 7)]
            out["joint_comm"] = [bool(getattr(es, f"communication_status_joint_{n}", False)) for n in range(1, 7)]
        except Exception as e:
            out["error"] = str(e)
        return out

    def gripper_feedback(self):
        try:
            g = self.piper.GetArmGripperMsgs().gripper_state
            return g.grippers_angle / GRIP_UNIT, g.grippers_effort / 1000.0
        except Exception:
            return None, None

    def motors_enabled(self):
        try:
            return [bool(x) for x in self.piper.GetArmEnableStatus()]
        except Exception:
            return []

    def can_fps(self):
        try:
            return float(self.piper.GetCanFps())
        except Exception:
            return 0.0

    # ---------------------------------------------------------------- commands
    def motion_ctrl(self, move_mode):
        """MotionCtrl_2: CAN control + move mode (0x00 MOVE P, 0x01 MOVE J) + speed."""
        with io_lock:
            self.piper.MotionCtrl_2(0x01, move_mode, self.speed, 0x00)
        self.move_mode = move_mode

    def set_speed(self, v):
        """Change move_spd_rate_ctrl live. Keeps the CURRENT move mode: always
        sending MOVE J here used to drop a Cartesian jog into joint mode."""
        self.speed = int(v)
        self.motion_ctrl(self.move_mode)
        log_call("MotionCtrl_2", {"move_mode": hex(self.move_mode),
                                  "move_spd_rate_ctrl": self.speed, "can_id": "0x151"})

    def _stop_targets(self):
        """Smooth mode: brake the trajectory and return where it will stop (the measured joints
        if it isn't running)."""
        stop_q = self.follower.stop() if self.follower.active else None
        return [float(v) for v in stop_q] if stop_q is not None else self.joints()

    def enter_joint(self):
        if self.smooth:
            tgt = self._stop_targets()
            with state_lock:
                STATE["mode"] = "joint"
                STATE["targets"] = tgt
                self._applied_targets = list(tgt)
                STATE["jog"] = [0.0] * 6
                STATE["jjog"] = [0.0] * 6
                STATE["servo_goal"] = None
            self.jogger.reset()
            self.motion_ctrl(0x01)
            log_call("MotionCtrl_2", {"move_mode": "0x01 MOVE J", "cartesian": "smooth", "can_id": "0x151"})
            return
        with state_lock:
            STATE["mode"] = "joint"
            STATE["targets"] = self.joints()
            STATE["jog"] = [0.0] * 6
            STATE["jjog"] = [0.0] * 6
            STATE["servo_goal"] = None
        self.motion_ctrl(0x01)
        log_call("MotionCtrl_2", {"move_mode": "0x01 MOVE J", "can_id": "0x151"})

    def enter_cartesian(self, mode="cartesian"):
        if self.smooth:
            # Our own IK streams joints, so the arm stays in MOVE J. A servo client's plan
            # replaces whatever is running; the panel's jog starts from a standstill.
            if mode != "servo" and self.follower.active:
                self.follower.stop()
            with state_lock:
                STATE["mode"] = mode
                STATE["pose"] = self.end_pose()
                STATE["jog"] = [0.0] * 6
                STATE["jjog"] = [0.0] * 6
                if mode != "servo":
                    STATE["servo_goal"] = None
            self.jogger.reset()
            self.motion_ctrl(0x01)
            log_call("MotionCtrl_2", {"move_mode": "0x01 MOVE J", "mode": mode, "cartesian": "smooth",
                                      "can_id": "0x151"})
            return
        with state_lock:
            STATE["mode"] = mode
            STATE["pose"] = self.end_pose()
            STATE["jog"] = [0.0] * 6
            STATE["jjog"] = [0.0] * 6
            if mode != "servo":
                STATE["servo_goal"] = None
        self.motion_ctrl(0x00)
        log_call("MotionCtrl_2", {"move_mode": "0x00 MOVE P", "mode": mode, "can_id": "0x151"})

    def enter_stream(self):
        with state_lock:
            STATE["mode"] = "stream"
            STATE["jog"] = [0.0] * 6
            STATE["jjog"] = [0.0] * 6
            STATE["servo_goal"] = None
        if self.smooth:
            # The ACT engine streams joints itself; the trajectory stops sending.
            self.follower.deactivate()
            self.jogger.reset()
        self.motion_ctrl(0x01)
        log_call("MotionCtrl_2", {"move_mode": "0x01 MOVE J", "mode": "stream", "can_id": "0x151"})

    def enable(self):
        """README section 4: EnableArm, wait, then CAN control. This also pulls the
        arm out of a latched drag-teach mode. Never send EmergencyStop "resume"
        first: that leaves the motors disabled."""
        with io_lock:
            self.piper.EnableArm(7, 0x02)
        log_call("EnableArm", {"motor_num": 7, "enable_flag": "0x02", "can_id": "0x471"})
        time.sleep(1.2)
        self.motion_ctrl(0x01)
        log_call("MotionCtrl_2", {"ctrl_mode": "0x01 CAN", "move_mode": "0x01 MOVE J",
                                  "speed": self.speed, "can_id": "0x151"})
        if self.smooth:
            self.follower.deactivate()      # re-anchors at the measured joints on the next move
            self.jogger.reset()
            self.select_dh()
        with state_lock:
            STATE["enabled"] = True
            STATE["estopped"] = False
            STATE["mode"] = "joint"
            STATE["targets"] = self.joints()
            if self.smooth:
                self._applied_targets = list(STATE["targets"])
            STATE["pose"] = self.end_pose()
            STATE["servo_goal"] = None

    def disable(self):
        with state_lock:
            STATE["enabled"] = False
            STATE["owner"] = None
        if self.smooth:
            self.follower.deactivate()
        with io_lock:
            self.piper.DisableArm(7, 0x01)
        log_call("DisableArm", {"motor_num": 7, "enable_flag": "0x01", "can_id": "0x471"})

    def estop(self):
        with state_lock:
            STATE["enabled"] = False
            STATE["estopped"] = True
            STATE["owner"] = None
            STATE["servo_goal"] = None
        if self.smooth:
            self.follower.deactivate()      # nothing more is streamed
        with io_lock:
            self.piper.EmergencyStop(0x01)
        log_call("EmergencyStop", {"flag": "0x01", "can_id": "0x150"})

    def hold(self, hard=False):
        """Controlled stop: keep the motors on and hold. In smooth mode the trajectory brakes to
        a standstill at the brake limits (a fraction of a second); hard=True, or when nothing is
        streaming, holds the measured joints at once, as firmware mode always does."""
        try:
            cur = self.joints()
        except Exception:
            return
        if self.smooth and not hard and self.follower.active:
            tgt = self._stop_targets()
            with state_lock:
                STATE["targets"] = tgt
                self._applied_targets = list(tgt)
                STATE["jog"] = [0.0] * 6
                STATE["jjog"] = [0.0] * 6
                STATE["servo_goal"] = None
                STATE["mode"] = "joint"
            self.jogger.reset()
            log_call("hold", {"joints": [round(v, 1) for v in tgt], "note": "braking to a stop"})
            return
        if self.smooth:
            self.follower.deactivate()
            self.jogger.reset()
            with state_lock:
                self._applied_targets = list(cur)
        with state_lock:
            STATE["targets"] = cur
            STATE["jog"] = [0.0] * 6
            STATE["jjog"] = [0.0] * 6
            STATE["servo_goal"] = None
            STATE["mode"] = "joint"
            enabled = STATE["enabled"]
        if enabled and self.feedback_age() < FEEDBACK_STALE:
            self.motion_ctrl(0x01)
            with io_lock:
                self.piper.JointCtrl(*[int(round(v * MDEG)) for v in cur])
        log_call("hold", {"joints": [round(v, 1) for v in cur]})

    def recover(self):
        """Clear a latched collision/fault without a power cycle: ResetPiper (CAN
        0x150) clears errors and flags (the arm goes limp for a moment), then
        re-enable."""
        with state_lock:
            STATE["enabled"] = False
            STATE["estopped"] = False
            STATE["owner"] = None
        with io_lock:
            self.piper.ResetPiper()
        log_call("ResetPiper", {"note": "clear all errors + flags", "can_id": "0x150"})
        time.sleep(0.5)
        self.enable()

    def set_gripper(self, mm, effort=None, force=False):
        """GripperCtrl, only when the target changes (>= 0.3 mm), the effort changes,
        or every 2 s. Re-sending the same target at 15-20 Hz made a closed gripper
        buzz against the object it was holding."""
        mm = max(GRIPPER_MM[0], min(GRIPPER_MM[1], float(mm)))
        effort = int(max(GRIPPER_EFFORT[0], min(GRIPPER_EFFORT[1],
                     GRIPPER_EFFORT_DEFAULT if effort is None else effort)))
        last_mm, last_eff, last_t = self._last_grip
        now = time.time()
        with state_lock:
            STATE["gripper_mm"] = mm
            STATE["gripper_effort"] = effort
        if not force and last_mm is not None and abs(mm - last_mm) < 0.3 \
                and effort == last_eff and now - last_t < 2.0:
            return mm
        with io_lock:
            self.piper.GripperCtrl(int(mm * GRIP_UNIT), effort, 0x01, 0)
        self._last_grip = (mm, effort, now)
        log_call("GripperCtrl", {"mm": round(mm, 1), "effort": effort, "can_id": "0x159"})
        return mm

    def stream_joints(self, joints):
        """Send one joint target now (ACT streaming). The caller rate-limits."""
        tgt = [clamp(n + 1, float(v)) for n, v in enumerate(joints)]
        with state_lock:
            STATE["targets"] = tgt
        with io_lock:
            self.piper.JointCtrl(*[int(round(v * MDEG)) for v in tgt])
        return tgt

    # ---------------------------------------------------------------- loop
    def _leash(self, actual, goal, lead_mm, lead_deg):
        """target = actual + (goal - actual) clamped to the lead, per axis. Angles
        take the short way round."""
        out = []
        for ax in range(6):
            d = goal[ax] - actual[ax]
            if ax >= 3:
                d = (d + 180.0) % 360.0 - 180.0
            lead = lead_mm if ax < 3 else lead_deg
            v = actual[ax] + max(-lead, min(lead, d))
            if ax < 3:
                v = max(-POSE_CLAMP_MM, min(POSE_CLAMP_MM, v))
            else:
                v = (v + 180.0) % 360.0 - 180.0
            out.append(v)
        return out

    def read_motor_limits(self):
        """The arm's own per-joint angle limits and maximum speeds (feedback 0x473, read-only; the SDK
        asks for them in ConnectPort). Speeds are 0.01 rad/s from firmware V1.5-8 on."""
        try:
            for _ in range(20):
                m = self.piper.GetAllMotorAngleLimitMaxSpd().all_motor_angle_limit_max_spd
                rows = [m.motor[i] for i in range(1, 7)]
                if all(getattr(r, "max_joint_spd", 0) > 0 for r in rows):
                    return [{"min_deg": r.min_angle_limit / 10.0, "max_deg": r.max_angle_limit / 10.0,
                             "max_deg_s": round(r.max_joint_spd * 0.01 * 180.0 / math.pi, 1)} for r in rows]
                try:
                    self.piper.SearchAllMotorMaxAngleSpd()
                except Exception:
                    pass
                time.sleep(0.1)
        except Exception as e:
            print(f"motor limits unavailable: {e}", flush=True)
        return None

    def _trip(self, why, owner, hard=False):
        """A guard fired: hold where the arm is (motors stay on) and drop the controller,
        whose goals are refused for IK_GUARD_LATCH seconds. hard: hold the measured joints at
        once instead of braking the trajectory (the trajectory is no longer where the arm is)."""
        with state_lock:
            STATE["guard"] = {"time": time.time(), "reason": why,
                              "client": owner["client"] if owner else None}
            STATE["owner"] = None
        self.guard.reset()
        self.hold(hard=hard)
        log_call("(ik guard)" if not hard else "(tracking guard)", {"reason": why, "action": "held"})

    def _loop(self):
        period = 1.0 / STREAM_HZ
        streaming = False
        while not self._stop.is_set():
            time.sleep(period)
            now = time.time()
            with state_lock:
                enabled = STATE["enabled"]
                mode = STATE["mode"]
                tgt = list(STATE["targets"])
                jog = list(STATE["jog"])
                jjog = list(STATE["jjog"])
                goal = STATE["servo_goal"]
                owner = owner_fresh(now)
                hb_fresh = (now - STATE["last_heartbeat"]) < HEARTBEAT_TIMEOUT
            # Who keeps the loop streaming: the servo owner for servo mode, the
            # heartbeat for the panel modes. Stream mode sends from the request.
            if mode == "servo":
                live = enabled and owner is not None and owner["kind"] == "servo" and goal is not None
            elif mode == "stream":
                live = False
            else:
                live = enabled and hb_fresh
            if live and self.feedback_age() >= FEEDBACK_STALE:
                live = False      # never steer from a pose we can't see
            if not live or mode not in ("cartesian", "servo") or mode != self._guard_mode:
                self.guard.reset()
            self._guard_mode = mode if live else None
            if live:
                if mode in ("cartesian", "servo"):
                    actual = self.end_pose()
                    why = self.guard.update(now, self.joints(), actual)
                    if why:
                        self._trip(why, owner)
                        streaming = False
                        continue
                    if mode == "cartesian":
                        target = self._leash(actual, [actual[i] + jog[i] for i in range(6)],
                                             JOG_LEAD_MM, JOG_LEAD_DEG)
                    else:
                        target = self._leash(actual, goal, SERVO_LEAD_MM, SERVO_LEAD_DEG)
                    with state_lock:
                        STATE["pose"] = target
                    with io_lock:
                        self.piper.EndPoseCtrl(*[int(round(v * 1000)) for v in target])
                else:
                    if any(jjog):
                        with state_lock:
                            for jn in range(6):
                                if jjog[jn]:
                                    STATE["targets"][jn] = clamp(jn + 1, STATE["targets"][jn] + jjog[jn])
                            tgt = list(STATE["targets"])
                    with io_lock:
                        self.piper.JointCtrl(*[int(round(v * MDEG)) for v in tgt])
                streaming = True
            elif streaming:
                streaming = False
                reason = "feedback stale" if enabled else "disabled"
                if enabled and mode == "servo":
                    reason = "servo deadman"
                elif enabled and mode not in ("stream",):
                    reason = "heartbeat lost"
                log_call("(deadman)", {"reason": reason, "action": "stopped streaming"})

    # ---------------------------------------------------------------- smooth mode
    def select_dh(self):
        """Pick the DH variant that matches the arm's own pose feedback (firmware >= V1.5-2
        offsets J2 and J3 by 2 deg). Needs fresh feedback; tried again on enable."""
        if self.feedback_age() >= FEEDBACK_STALE:
            return
        try:
            kin, err = KIN.select_dh_variant(self.joints(), self.end_pose())
        except Exception:
            return
        if kin.dh_variant != self.kin.dh_variant:
            self.kin = self.planner.kin = self.jogger.kin = kin
            log_call("kinematics", {"dh_variant": hex(kin.dh_variant), "fk_error_mm": round(err, 2)})
        self.fk_error_mm = err
        self._dh_selected = True

    def check_fk(self, joints=None):
        """How far (mm) our FK of the measured joints lands from the arm's own end pose."""
        try:
            q = joints if joints is not None else self.joints()
            p, _ = self.kin.fk(q)
            self.fk_error_mm = float(np.linalg.norm(p - np.asarray(self.end_pose()[:3])))
        except Exception:
            pass
        return self.fk_error_mm

    def kinematics_refusal(self):
        """Why smooth Cartesian motion must not run now, or None."""
        err = self.fk_error_mm
        if err is not None and err > FK_TOL_MM:
            return (f"the kinematic model doesn't match the arm (our FK is {err:.0f} mm off its own "
                    "pose); run the driver with --cartesian firmware")
        return None

    def servo_goal(self, pose):
        """Plan to a Cartesian goal and hand the path to the trajectory. The same goal sent
        again (piperd and LLM Arm repeat theirs at 10 Hz) keeps the plan that is running."""
        meas = self.joints()
        with self.plan_lock:
            q0 = self.follower.anchor(meas)
            if self._last_plan is not None:
                plan, pid, t = self._last_plan
                same = plan.goal is not None and max(abs(a - b) for a, b in zip(plan.goal, pose)) < 0.05
                if same and ((plan.moves and pid == self.follower.path_id)
                             or (not plan.moves and time.time() - t < MOTION.REPLAN_FAILED_S)):
                    return plan
            plan = self.planner.plan_pose(q0, pose)
            if plan.moves:
                pid = plan.install(self.follower, anchor=meas)
            else:
                self.follower.stop()
                pid = self.follower.path_id
            self._last_plan = (plan, pid, time.time())
        with state_lock:
            STATE["plan"] = plan.summary()
        if plan.kind not in ("line", "hold"):
            log_call("plan", {"kind": plan.kind, "reason": plan.reason, "ms": round(plan.ms)})
        return plan

    def home(self):
        """HOME = the READY pose, a joint-space move checked like a Cartesian plan."""
        meas = self.joints()
        with self.plan_lock:
            q0 = self.follower.anchor(meas)
            plan = self.planner.plan_joints(q0, READY_POSE)
            with state_lock:
                pid = self.follower.set_path(plan.path, plan.corners, 1.0, anchor=meas)
                STATE["targets"] = list(READY_POSE)
                self._applied_targets = list(READY_POSE)
                STATE["plan"] = plan.summary()
            self._last_plan = (plan, pid, time.time())
        return plan

    def _joint_tick(self, q, jjog, dt):
        """Joint mode: held wrist keys move their targets; changed targets go to the trajectory."""
        with state_lock:
            if STATE["mode"] != "joint":
                return
            for jn in range(6):
                if jjog[jn]:
                    cur = STATE["targets"][jn]
                    v = clamp(jn + 1, cur + (1.0 if jjog[jn] > 0 else -1.0) * JOINT_JOG_DEG_S * dt)
                    if abs(v - q[jn]) <= JJOG_LEAD_DEG or abs(v - q[jn]) < abs(cur - q[jn]):
                        STATE["targets"][jn] = v
            tgt = [clamp(n + 1, float(v)) for n, v in enumerate(STATE["targets"])]
            applied = self._applied_targets
            if applied is None or any(abs(a - b) > 1e-9 for a, b in zip(tgt, applied)):
                try:
                    self.follower.set_target(tgt)
                except RuntimeError:          # stopped meanwhile (E-stop, disable): nothing to do
                    return
                self._applied_targets = list(tgt)

    def _jog_tick(self, q, jog, speed, dt):
        """Panel jog: the held axes move a Cartesian carrot; releasing them brakes."""
        dirs = [0 if not v else (1 if v > 0 else -1) for v in jog]
        if any(dirs):
            try:
                self.follower.set_path(self.jogger.step(q, dirs, speed, dt), brisk=True)
            except RuntimeError:              # stopped meanwhile (E-stop, disable)
                return
            if self.jogger.blocked and time.time() - self._jog_note_t > 1.0:
                self._jog_note_t = time.time()
                log_call("(jog)", {"blocked": self.jogger.blocked})
        elif self.jogger.carrot is not None:
            self.jogger.reset()
            self.follower.stop()

    def _loop_smooth(self):
        """Smooth mode's control loop: one trajectory tick and one JointCtrl per period. A tick
        that raises is logged and the arm holds the measured joints; the loop keeps running, so
        the deadmen and stops keep working."""
        period = 1.0 / MOTION.CONTROL_HZ
        next_t = time.monotonic()
        ctx = {"streaming": False, "last_fk": 0.0, "fault_seen": None, "error_t": 0.0}
        while not self._stop.is_set():
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -period:
                self.overruns += 1             # never replay missed ticks in a burst
                next_t = time.monotonic()
            try:
                self._smooth_tick(ctx, period)
            except Exception as e:             # pragma: no cover - a bug, not a path
                if time.time() - ctx["error_t"] > 1.0:
                    ctx["error_t"] = time.time()
                    print(f"control loop error: {e!r}; holding", flush=True)
                    log_call("(control loop)", {"error": repr(e)[:200], "action": "held"})
                try:
                    with state_lock:
                        enabled = STATE["enabled"] and not STATE["estopped"]
                    if enabled:
                        self.hold(hard=True)
                except Exception:
                    pass
                ctx["streaming"] = False

    def _smooth_tick(self, ctx, period):
        fol = self.follower
        now = time.time()
        with state_lock:
            enabled, estopped, mode = STATE["enabled"], STATE["estopped"], STATE["mode"]
            jog, jjog = list(STATE["jog"]), list(STATE["jjog"])
            goal = STATE["servo_goal"]
            owner = owner_fresh(now)
            hb_fresh = (now - STATE["last_heartbeat"]) < HEARTBEAT_TIMEOUT
        speed = self.speed
        if not enabled or estopped or mode == "stream":
            if fol.active:
                fol.deactivate()
                self.jogger.reset()
            if ctx["streaming"]:
                ctx["streaming"] = False
                if mode != "stream":
                    log_call("(deadman)", {"reason": "disabled", "action": "stopped streaming"})
            self.guard.reset()
            self._guard_mode = None
            return
        # Who keeps the arm moving: the servo owner in servo mode, the heartbeat otherwise.
        if mode == "servo":
            live = owner is not None and owner["kind"] == "servo" and goal is not None
            why_not = "servo deadman"
        else:
            live, why_not = hb_fresh, "heartbeat lost"
        meas = None
        if self.feedback_age() < FEEDBACK_STALE:
            try:
                meas = self.joints()
            except Exception:
                meas = None
        if meas is None:
            live, why_not = False, "feedback stale"     # brake, never steer blind
        elif not self._dh_selected:
            self.select_dh()
        elif now - ctx["last_fk"] > 1.0:
            self.check_fk(meas)
            ctx["last_fk"] = now
        q_cmd = fol.q                     # one snapshot: other threads may deactivate it
        if q_cmd is None:
            if not live:
                if ctx["streaming"]:
                    ctx["streaming"] = False
                    log_call("(deadman)", {"reason": why_not, "action": "stopped streaming"})
                return
            q_cmd = fol.anchor(meas)
        # Guards. Tracking: the arm must be where the trajectory says (else it is blocked,
        # faulted or ignoring commands). IK flip: as in firmware mode, from the feedback.
        # A slow arm is not a blocked arm: the owner's J3 stopped at -110 because the firmware moved it
        # slower than the trajectory and the old guard tripped on the lag alone. Now it trips only when
        # the gap is large AND hasn't closed for TRACK_STALL_S; and the trajectory waits while the arm
        # is more than LAG_PAUSE_DEG behind, so the lag can't build up in the first place.
        tripped = False
        lagging = False
        if meas is not None:
            err = np.abs(np.asarray(meas) - q_cmd)
            worst = int(np.argmax(err))
            lagging = err[worst] > LAG_PAUSE_DEG
            why = None
            if err[worst] > self.track_err_deg:
                # With the pause below, slow following can't open a gap this big: something moved the arm.
                why = f"J{worst + 1} is {err[worst]:.0f} deg from its command"
            elif lagging:
                st = ctx.get("stall")
                if st is None or st[0] != worst or err[worst] < st[2] - TRACK_PROGRESS_DEG:
                    ctx["stall"] = (worst, now, err[worst])          # (re)start the clock on progress
                elif now - st[1] > TRACK_STALL_S:
                    why = f"J{worst + 1} is {err[worst]:.0f} deg behind its command and not catching up"
            else:
                ctx["stall"] = None
            if why:
                ctx["stall"] = None
                self._trip(why + " (blocked, or the arm is not following)", owner, hard=True)
                ctx["streaming"] = False
                return
        if not live or mode not in ("cartesian", "servo") or mode != self._guard_mode:
            self.guard.reset()
        self._guard_mode = mode if live else None
        if live and mode in ("cartesian", "servo"):
            why = self.guard.update(now, meas, self.end_pose())
            if why:
                self._trip(why, owner)     # the trajectory brakes; the stop is streamed below
                tripped = True
        if live and not tripped:
            if mode == "joint":
                self._joint_tick(q_cmd, jjog, period)
            elif mode == "cartesian":
                self._jog_tick(q_cmd, jog, speed, period)
            # servo: /api/servo hands plans to the trajectory
        elif not live and fol.moving() and not fol.stopping:
            fol.stop()
            log_call("(deadman)", {"reason": why_not, "action": "braking to a stop"})
        if not ((live and not tripped) or fol.moving() or fol.stopping):
            if ctx["streaming"]:
                ctx["streaming"] = False
                log_call("(deadman)", {"reason": why_not, "action": "stopped streaming"})
            if not live and fol.idle():
                fol.deactivate()
                self.jogger.reset()
            return
        q = fol.hold_tick() if lagging and fol.moving() else fol.step(speed)
        if q is None:
            return
        if fol.fault and fol.fault != ctx["fault_seen"]:
            ctx["fault_seen"] = fol.fault
            log_call("(trajectory)", {"fault": fol.fault, "action": "held"})
            # Where it is held becomes the target, so nothing keeps pulling toward the refused one.
            with state_lock:
                if STATE["mode"] == "joint":
                    STATE["targets"] = [clamp(n + 1, float(v)) for n, v in enumerate(q)]
                    self._applied_targets = list(STATE["targets"])
        cmd = [int(round(clamp(n + 1, float(v)) * MDEG)) for n, v in enumerate(q)]
        with io_lock:
            # estop() sets the flag before it sends EmergencyStop under this lock, so a command
            # can never follow an E-stop.
            with state_lock:
                go = STATE["enabled"] and not STATE["estopped"]
            if not go:
                return
            self.piper.JointCtrl(*cmd)
        with state_lock:
            STATE["pose"] = self.kin.fk_pose(q)
        ctx["streaming"] = True

    def close(self):
        self._stop.set()


ARM = None
PANEL_HTML = b""

# Address of a direct point-to-point Ethernet link to a laptop, if there is one
# (the RDK used 10.90.0.153 on eth0). Only `--bind auto` uses it. It must never be
# 0.0.0.0: that publishes arm control on every network, including shared Wi-Fi.
DIRECT_LINK_IP = "10.90.0.153"


def pick_bind_address():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((DIRECT_LINK_IP, 0))
        s.close()
        return DIRECT_LINK_IP
    except OSError:
        return "127.0.0.1"


def finite(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def bus_has_traffic(iface, timeout=1.0):
    """True if any frame shows up on the CAN interface within `timeout` (a raw socket of our own)."""
    import socket
    try:
        s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        s.settimeout(timeout)
        s.bind((iface,))
        s.recv(16)
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def stall_watchdog(iface, stale_s=5.0):
    """Exit if the SDK has stopped receiving while the arm still talks on the bus.

    Seen on the Surface after the arm was power-cycled or switched out of teach mode: the SDK's reader went
    quiet (feedback stale, 0 frames/s) while candump showed the arm's status frames, and only a restart
    brought it back. Exiting lets systemd restart the driver (Restart=on-failure); the motors are never
    disabled on the way out, so the arm keeps holding."""
    since = None
    while True:
        time.sleep(1.0)
        if ARM.feedback_age() < FEEDBACK_STALE:
            since = None
            continue
        since = since or time.time()
        if time.time() - since >= stale_s and bus_has_traffic(iface):
            print(f"feedback stale for {time.time() - since:.0f} s while {iface} has traffic: "
                  "restarting the driver", flush=True)
            os._exit(3)


def guard_latched():
    """The IK guard's reason while its latch is on, else None."""
    with state_lock:
        g = STATE["guard"]
    if g and time.time() - g["time"] < IK_GUARD_LATCH:
        return g["reason"]
    return None


def state_json(lite=False):
    now = time.time()
    age = ARM.feedback_age()
    fresh = age < FEEDBACK_STALE
    st = ARM.status()
    grip_mm, grip_eff = ARM.gripper_feedback()
    try:
        cur = ARM.joints()
        pose_cur = ARM.end_pose()
    except Exception:
        cur, pose_cur = [0.0] * 6, [0.0] * 6
    with state_lock:
        o = owner_fresh(now)
        body = {
            "current": cur, "targets": STATE["targets"],
            "mode": STATE["mode"], "pose": STATE["pose"], "pose_current": pose_cur,
            "arm_status": st.get("arm_status", "?"),
            "gripper_mm": STATE["gripper_mm"], "enabled": STATE["enabled"],
            "estopped": STATE["estopped"], "speed": STATE["speed"],
            "limits": JOINT_LIMITS, "gripper_range": list(GRIPPER_MM),
            "speed_range": [SPEED_MIN, SPEED_MAX],
            # --- added for piperd and the ACT engine ---
            "time": now,
            "feedback_fresh": fresh,
            "feedback_age_ms": finite(round(age * 1000.0, 1)) if math.isfinite(age) else None,
            "can_fps": ARM.can_fps(),
            "status": {k: finite(v) for k, v in st.items()},
            "teaching": st.get("ctrl_mode_code") == 0x02,
            "soft_fault": st.get("arm_status_code") in SOFT_FAULTS,
            "gripper_actual_mm": grip_mm, "gripper_effort_nm": grip_eff,
            "motors_enabled": ARM.motors_enabled(),
            "owner": ({"client": o["client"], "kind": o["kind"],
                       "age_ms": round((now - o["last"]) * 1000.0)} if o else None),
            "servo_goal": STATE["servo_goal"],
            "guard": STATE["guard"], "ik_guard_deg": ARM.guard.limit,
            "ready_pose": READY_POSE, "workspace_mm": WORKSPACE_MM,
            "firmware": ARM.firmware,
            "motor_limits": getattr(ARM, "motor_limits", None),
            # --- Cartesian control mode ---
            "cartesian": ARM.cartesian,
            "cartesian_note": ARM.cartesian_note,
            "control_hz": MOTION.CONTROL_HZ if ARM.smooth else STREAM_HZ,
        }
        if ARM.smooth:
            body["plan"] = STATE["plan"]
            body["trajectory"] = ARM.follower.snapshot()
            body["fk_error_mm"] = finite(round(ARM.fk_error_mm, 2)) if ARM.fk_error_mm is not None else None
            body["dh_variant"] = ARM.kin.dh_variant
            body["jog_blocked"] = ARM.jogger.blocked or None
            body["overruns"] = ARM.overruns
        if not lite:
            body["log"] = STATE["log"][:14]
    return body


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _json(self, code, obj):
        return self._send(code, json.dumps(obj))

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, PANEL_HTML, "text/html; charset=utf-8")
        if u.path == "/api/state":
            try:
                lite = parse_qs(u.query).get("lite", ["0"])[0] == "1"
                return self._json(200, state_json(lite))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"error": "body is not JSON"})
        client = str(data.get("client") or "panel")[:32]
        try:
            return self._dispatch(self.path, data, client)
        except Busy as b:
            return self._json(409, {"error": "busy", "owner": b.owner})
        except (ValueError, TypeError) as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:
            return self._json(500, {"error": str(e)})

    def _dispatch(self, path, data, client):
        # ---- always allowed ----
        if path == "/api/heartbeat":
            with state_lock:
                STATE["last_heartbeat"] = time.time()
                o = STATE["owner"]
                if o and o["client"] == client:
                    o["last"] = STATE["last_heartbeat"]
            return self._json(200, {"ok": True})
        if path == "/api/estop":
            ARM.estop()
            return self._json(200, {"ok": True})
        if path == "/api/stop":
            ARM.hold()
            with state_lock:
                STATE["owner"] = None
            return self._json(200, {"ok": True})
        if path == "/api/release":
            with state_lock:
                o = STATE["owner"]
                if o and o["client"] == client:
                    STATE["owner"] = None
            ARM.hold()
            return self._json(200, {"ok": True})
        if path == "/api/enable":
            ARM.enable() if data.get("on") else ARM.disable()
            return self._json(200, {"ok": True})
        if path == "/api/recover":
            ARM.recover()
            return self._json(200, {"ok": True})
        if path == "/api/speed":
            v = int(max(SPEED_MIN, min(SPEED_MAX, int(data.get("speed", 15)))))
            with state_lock:
                STATE["speed"] = v
            ARM.set_speed(v)
            return self._json(200, {"ok": True, "speed": v})

        # ---- motion: one controller at a time ----
        if path == "/api/servo":
            pose = [float(v) for v in data.get("pose", [])]
            if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
                raise ValueError("pose must be 6 finite numbers: x,y,z mm, rx,ry,rz deg")
            why = guard_latched()
            if why:
                return self._json(409, {"error": f"ik guard: {why}", "guard": True})
            claim(client, "servo")
            with state_lock:
                enabled = STATE["enabled"]
            if not enabled:
                return self._json(409, {"error": "motors are not enabled"})
            if ARM.feedback_age() >= FEEDBACK_STALE:
                return self._json(409, {"error": "arm feedback is stale"})
            if ARM.smooth and ARM.kinematics_refusal():
                return self._json(409, {"error": ARM.kinematics_refusal()})
            clamped = False
            for ax, key in enumerate(("x", "y", "z")):
                lo, hi = WORKSPACE_MM[key]
                if not lo <= pose[ax] <= hi:
                    pose[ax] = max(lo, min(hi, pose[ax]))
                    clamped = True
            if "speed" in data:
                v = int(max(SPEED_MIN, min(SPEED_MAX, int(data["speed"]))))
                if v != ARM.speed:
                    with state_lock:
                        STATE["speed"] = v
                    ARM.speed = v
                    ARM.motion_ctrl(ARM.move_mode)
            with state_lock:
                if STATE["mode"] != "servo":
                    entering = True
                else:
                    entering = False
            if entering:
                ARM.enter_cartesian("servo")
            with state_lock:
                STATE["servo_goal"] = pose
            if ARM.smooth:
                plan = ARM.servo_goal(pose)
                return self._json(200, {"ok": True, "goal": pose, "clamped": clamped,
                                        "plan": plan.kind, "reason": plan.reason})
            return self._json(200, {"ok": True, "goal": pose, "clamped": clamped})

        if path == "/api/stream":
            joints = [float(v) for v in data.get("joints", [])]
            if len(joints) != 6 or not all(math.isfinite(v) for v in joints):
                raise ValueError("joints must be 6 finite numbers in degrees")
            claim(client, "stream")
            with state_lock:
                enabled = STATE["enabled"]
                entering = STATE["mode"] != "stream"
            if not enabled:
                return self._json(409, {"error": "motors are not enabled"})
            if ARM.feedback_age() >= FEEDBACK_STALE:
                return self._json(409, {"error": "arm feedback is stale"})
            if entering:
                ARM.enter_stream()
            sent = ARM.stream_joints(joints)
            grip = None
            if data.get("gripper_mm") is not None:
                grip = ARM.set_gripper(float(data["gripper_mm"]), data.get("gripper_effort"))
            return self._json(200, {"ok": True, "joints": sent, "gripper_mm": grip})

        if path == "/api/gripper":
            claim(client, CLIENT_KIND.get(client, "panel"))
            mm = ARM.set_gripper(float(data.get("mm", 0.0)), data.get("effort"), force=True)
            return self._json(200, {"ok": True, "mm": mm})

        if path == "/api/joint":
            j, deg = int(data.get("joint", 0)), float(data.get("deg", 0.0))
            if j not in JOINT_LIMITS:
                raise ValueError("bad joint")
            claim(client, "panel")
            if STATE["mode"] != "joint":
                ARM.enter_joint()
            v = clamp(j, deg)
            fol = getattr(ARM, "follower", None)
            with state_lock:
                # A slider moves its own joint only. Targets the other joints never reached (a hold, a
                # fault, a stop) are reset to where the arm is held, or moving J5 dragged J2 along.
                if fol is not None and fol.q is not None and not fol.moving():
                    for k in range(6):
                        if k != j - 1:
                            STATE["targets"][k] = clamp(k + 1, float(fol.q[k]))
                STATE["targets"][j - 1] = v
            log_call("JointCtrl", {f"joint_{j}": round(v, 2), "note": "smooth" if fol is not None else "streamed @20Hz"})
            return self._json(200, {"ok": True, "deg": v})

        if path == "/api/mode":
            claim(client, "panel")
            m = "cartesian" if data.get("mode") == "cartesian" else "joint"
            (ARM.enter_cartesian if m == "cartesian" else ARM.enter_joint)()
            return self._json(200, {"ok": True, "mode": m})

        if path == "/api/jog":
            axis = int(data.get("axis", -1))
            direction = 1.0 if float(data.get("dir", 0)) >= 0 else -1.0
            on = bool(data.get("on", False))
            if not 0 <= axis <= 5:
                raise ValueError("bad axis")
            if on and guard_latched():
                return self._json(409, {"error": f"ik guard: {guard_latched()}", "guard": True})
            if on and ARM.smooth and ARM.kinematics_refusal():
                return self._json(409, {"error": ARM.kinematics_refusal()})
            if on:
                claim(client, "panel")
            if STATE["mode"] != "cartesian":
                ARM.enter_cartesian()
            lead = JOG_LEAD_MM if axis < 3 else JOG_LEAD_DEG
            with state_lock:
                if on:
                    STATE["jog"][axis] = direction * lead
                elif STATE["jog"][axis] * direction > 0:
                    STATE["jog"][axis] = 0.0
                jog = list(STATE["jog"])
            return self._json(200, {"ok": True, "jog": jog})

        if path == "/api/jointjog":
            jn = int(data.get("joint", -1))
            direction = 1.0 if float(data.get("dir", 0)) >= 0 else -1.0
            on = bool(data.get("on", False))
            if not 0 <= jn <= 5:
                raise ValueError("bad joint")
            if on:
                claim(client, "panel")
            if STATE["mode"] != "joint":
                ARM.enter_joint()
            with state_lock:
                if on:
                    STATE["jjog"][jn] = direction * JOINT_JOG_STEP
                elif STATE["jjog"][jn] * direction > 0:
                    STATE["jjog"][jn] = 0.0
                jjog = list(STATE["jjog"])
            return self._json(200, {"ok": True, "jjog": jjog})

        if path == "/api/jogstop":
            with state_lock:
                STATE["jog"] = [0.0] * 6
                STATE["jjog"] = [0.0] * 6
            return self._json(200, {"ok": True})

        if path == "/api/home":
            # HOME = the READY pose, in joint mode (MOVE J needs no IK, so it never
            # faults). Joint zero is the singular pose where only Z can move.
            claim(client, "panel")
            if STATE["mode"] != "joint":
                ARM.enter_joint()
            with state_lock:
                STATE["jjog"] = [0.0] * 6
                STATE["last_heartbeat"] = time.time()
            if ARM.smooth and STATE["enabled"] and ARM.feedback_age() < FEEDBACK_STALE:
                plan = ARM.home()
                log_call("JointCtrl", {"targets": READY_POSE, "note": f"home = READY pose ({plan.kind})",
                                       "reason": plan.reason})
                return self._json(200, {"ok": True, "targets": READY_POSE, "plan": plan.kind})
            with state_lock:
                STATE["targets"] = list(READY_POSE)
            log_call("JointCtrl", {"targets": READY_POSE, "note": "home = READY pose (MOVE J)"})
            return self._json(200, {"ok": True, "targets": READY_POSE})

        return self._json(404, {"error": "not found"})


def main():
    global ARM, PANEL_HTML
    ap = argparse.ArgumentParser(description="PiPER arm driver (NervusOS hardware layer)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--iface", default="can0")
    ap.add_argument("--speed", type=int, default=15)
    ap.add_argument("--ik-guard-deg", type=float, default=IK_GUARD_DEG,
                    help="joint swing (deg in %.1f s) the TCP motion must explain in the Cartesian "
                         "modes, else the arm holds (0 = off)" % IK_GUARD_WINDOW)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="address to listen on (default loopback). 'auto' = the direct-link "
                         "IP if that Ethernet cable is up, else loopback.")
    ap.add_argument("--cartesian", choices=("smooth", "firmware"), default="smooth",
                    help="smooth (default): our IK and planner, joints streamed at 100 Hz from a "
                         "jerk-limited trajectory (also for joint moves and HOME). firmware: the "
                         "arm's own IK, EndPoseCtrl leashed at 20 Hz (the previous behaviour).")
    ap.add_argument("--track-err-deg", type=float, default=TRACK_ERR_DEG,
                    help="smooth mode: hold if a joint lags its command by more than this (deg)")
    ap.add_argument("--workspace", metavar="X0,X1,Y0,Y1,Z0,Z1",
                    help="the Cartesian box (mm) servo goals are clamped to and smooth-mode paths "
                         "keep the TCP in (default %s)" % ",".join(
                             f"{WORKSPACE_MM[k][0]:.0f},{WORKSPACE_MM[k][1]:.0f}" for k in "xyz"))
    args = ap.parse_args()

    if args.bind == "auto":
        args.bind = pick_bind_address()
    if args.workspace:
        v = [float(s) for s in args.workspace.split(",")]
        if len(v) != 6 or not (v[0] < v[1] and v[2] < v[3] and v[4] < v[5]):
            raise SystemExit("--workspace needs X0,X1,Y0,Y1,Z0,Z1 with each min < max")
        WORKSPACE_MM.update({"x": (v[0], v[1]), "y": (v[2], v[3]), "z": (v[4], v[5])})
    try:
        with open(os.path.join(HERE, "panel.html"), "rb") as f:
            PANEL_HTML = f.read()
    except OSError:
        PANEL_HTML = b"<p>panel.html is missing; the API is still available under /api/.</p>"
    with state_lock:
        STATE["speed"] = args.speed
    ARM = Arm(args.iface, args.speed, cartesian=args.cartesian, track_err_deg=args.track_err_deg)
    ARM.guard.limit = max(0.0, args.ik_guard_deg)
    threading.Thread(target=stall_watchdog, args=(args.iface,), daemon=True, name="stall-watchdog").start()
    try:
        ARM.firmware = ARM.piper.GetPiperFirmwareVersion()
    except Exception:
        ARM.firmware = None
    ARM.motor_limits = ARM.read_motor_limits()
    if ARM.motor_limits and getattr(ARM, "follower", None) is not None:
        # Never plan faster than the arm goes: the firmware moves each joint at most at its own maximum
        # times the MOVE J speed percentage; the trajectory keeps 20 % below that.
        ARM.follower.vcap = np.array([0.8 * m["max_deg_s"] for m in ARM.motor_limits])
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    srv.daemon_threads = True

    def stop(*_):
        # Never disable the motors on the way out: a disabled PiPER drops under
        # gravity. Hold the current pose and let the arm keep it.
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"PiPER driver listening on http://{args.bind}:{args.port} (firmware {ARM.firmware}, "
          f"cartesian {ARM.cartesian})", flush=True)
    try:
        srv.serve_forever()
    finally:
        try:
            ARM.hold(hard=True)      # the control loop stops with us: hold the measured joints
        except Exception:
            pass
        ARM.close()
        print("PiPER driver stopped; motors left as they were", flush=True)


if __name__ == "__main__":
    main()
