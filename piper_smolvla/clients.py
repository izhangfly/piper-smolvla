"""HTTP clients for the two services the robot side talks to.

* DriverClient: the PiPER driver (hardware/piper/piper_webctl.py in NervusOS, 127.0.0.1:8770). It is the only
  program that touches the CAN bus; every arm client talks to it over loopback, one controller at a time by
  client name. Adapted from the ACT engine's arm client (act/engine/nervus_act/hardware.py, same author): state
  is polled on a thread so readers never wait on HTTP, and feedback the driver reports as stale is never
  treated as a pose (the SDK keeps returning cached values, and cached zeros look like a plausible pose).
* CameraClient: one camera role served by the ACT engine (127.0.0.1:8795) as GET /api/camera/<role>.jpg. The
  engine owns the cameras (it opens them by their stable /dev/v4l/by-id names); polling it also keeps them
  awake, as ACT Studio does.

Units are the driver's: joints in degrees, the gripper opening in millimetres.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections import deque

import numpy as np

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def mono() -> float:
    return time.monotonic()


class DriverError(RuntimeError):
    pass


class Busy(DriverError):
    """Another client (Robot Arm, Hand Teleop, the ACT engine) controls the arm."""


# Adapted from izhangfly/NervusOS-Cambridge/act/engine/nervus_act/hardware.py (Arm).
class DriverClient:
    def __init__(self, base: str = "http://127.0.0.1:8770", client: str = "smolvla", poll_hz: float = 50.0):
        self.base = base.rstrip("/")
        self.client = client
        self.poll_period = 1.0 / poll_hz
        self.latest: dict | None = None
        self.latest_at = 0.0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- transport ----
    def get(self, path: str, timeout: float = 1.0) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return json.loads(r.read())

    def post(self, path: str, body: dict | None = None, timeout: float = 3.0) -> dict:
        data = dict(body or {})
        data["client"] = self.client
        req = urllib.request.Request(self.base + path, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read())
            except Exception:
                body = {}
            msg = body.get("error", str(e))
            if e.code == 409 and msg == "busy":
                raise Busy(f"the arm is controlled by '{body.get('owner')}': close that app or press Stop there")
            raise DriverError(msg)
        except urllib.error.URLError as e:
            raise DriverError(f"PiPER driver unreachable at {self.base}: {e.reason}")

    # ---- polling ----
    def start(self) -> "DriverClient":
        if self._thread is None:
            self._thread = threading.Thread(target=self._poll, daemon=True, name="driver-poll")
            self._thread.start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            t0 = mono()
            try:
                s = self.get("/api/state?lite=1", timeout=0.5)
                self.latest, self.latest_at, self.error = s, mono(), None
            except Exception as e:
                self.error = f"PiPER driver at {self.base}: {e}"
            dt = self.poll_period - (mono() - t0)
            if dt > 0:
                self._stop.wait(dt)

    def wait_first(self, timeout: float) -> bool:
        t_end = mono() + timeout
        while mono() < t_end:
            if self.latest is not None:
                return True
            time.sleep(0.02)
        return self.latest is not None

    def reachable(self) -> bool:
        return self.latest is not None and mono() - self.latest_at < 1.0

    # ---- reading ----
    def read(self) -> tuple[dict | None, float]:
        """(state, age_s): JOINTS -> degrees, the gripper in millimetres. The age is how old the arm's own
        feedback is plus how long ago it was polled; stale feedback is infinitely old, never a pose."""
        s = self.latest
        if s is None or not s.get("feedback_fresh"):
            return None, math.inf
        g = s.get("gripper_actual_mm")
        cur = s.get("current")
        if g is None or not cur or len(cur) != 6:
            return None, math.inf
        age = (mono() - self.latest_at) + (s.get("feedback_age_ms") or 0) / 1000.0
        state = {f"joint{i + 1}": float(cur[i]) for i in range(6)}
        state["gripper"] = float(g)
        return state, age

    def info(self) -> dict:
        s = self.latest or {}
        st = s.get("status") or {}
        return {
            "reachable": self.reachable(), "error": self.error, "fresh": bool(s.get("feedback_fresh")),
            "feedback_age_ms": s.get("feedback_age_ms"), "enabled": bool(s.get("enabled")),
            "estopped": bool(s.get("estopped")), "teaching": bool(s.get("teaching")),
            "ctrl_mode": st.get("ctrl_mode"), "arm_status": st.get("arm_status"),
            "soft_fault": bool(s.get("soft_fault")), "owner": s.get("owner"), "joints": s.get("current"),
            "gripper_mm": s.get("gripper_actual_mm"), "gripper_range": s.get("gripper_range"),
            "limits": s.get("limits"), "firmware": s.get("firmware"), "sim": bool(s.get("sim")),
        }

    def teaching(self) -> bool:
        s = self.latest or {}
        return bool(s.get("teaching")) and mono() - self.latest_at < 1.0

    def hard_fault(self) -> str | None:
        """A fault that needs Recover. Soft faults (singularity, no IK solution, target past a limit) clear
        themselves about a second after the arm stops being pushed, so they are not a reason to stop."""
        s = self.latest or {}
        st = s.get("status") or {}
        if not s.get("feedback_fresh"):
            return None  # staleness is handled by the age checks
        if s.get("estopped"):
            return "the arm is emergency-stopped"
        if st.get("arm_status_code") not in (None, 0x00) and not s.get("soft_fault"):
            return f"arm status {st.get('arm_status')}"
        return None

    def owner(self) -> str | None:
        o = (self.latest or {}).get("owner")
        return o.get("client") if o else None

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """{"joint1": (lo, hi), ...} from the driver (the AgileX manual's limits), plus the gripper range."""
        s = self.latest or {}
        lim = s.get("limits") or {}
        out = {}
        for i in range(1, 7):
            v = lim.get(str(i), lim.get(i))
            if v:
                out[f"joint{i}"] = (float(v[0]), float(v[1]))
        g = s.get("gripper_range")
        if g:
            out["gripper"] = (float(g[0]), float(g[1]))
        return out

    # ---- commanding ----
    def speed(self, pct: int):
        self.post("/api/speed", {"speed": int(pct)})

    def stream(self, target: dict, gripper_effort: int):
        """One joint target (degrees) plus the gripper (mm), sent now. The driver clamps joints to the manual's
        limits and re-sends the gripper only when it changes (a stalled gripper used to buzz)."""
        self.post("/api/stream", {"joints": [float(target[j]) for j in JOINTS[:6]],
                                  "gripper_mm": float(target["gripper"]), "gripper_effort": int(gripper_effort)},
                  timeout=1.0)

    def release(self):
        """Give up control. The driver holds the measured joints, motors on."""
        try:
            self.post("/api/release")
        except DriverError:
            pass

    def close(self):
        self._stop.set()


# Adapted from izhangfly/NervusOS-Cambridge/act/engine/nervus_act/hardware.py (Camera: latest frame on its own thread).
class CameraClient:
    """The newest frame of one camera role, fetched from the ACT engine on its own thread.

    The engine re-encodes a JPEG only when its camera delivers a new frame and keeps serving the last one when
    the camera stalls, so a frame's age here is the time since the JPEG last CHANGED: a frozen camera goes
    stale even though every request succeeds. (Sensor noise makes every real frame's JPEG differ, even of a
    still scene.)"""

    def __init__(self, base: str, role: str, width: int, height: int, poll_hz: float = 20.0):
        self.base, self.role = base.rstrip("/"), role
        self.w, self.h = width, height
        self.period = 1.0 / poll_hz
        self.frame: np.ndarray | None = None
        self.stamp = 0.0
        self.error: str | None = None
        self.shape: tuple | None = None
        self.times: deque = deque(maxlen=30)
        self._digest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"{self.base}/api/camera/{self.role}.jpg"

    def start(self) -> "CameraClient":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.role}")
            self._thread.start()
        return self

    def fetch(self, timeout: float = 1.0) -> tuple[np.ndarray, bytes]:
        """One frame now: (RGB uint8 (H, W, 3), a digest of the JPEG)."""
        import cv2
        with urllib.request.urlopen(self.url, timeout=timeout) as r:
            raw = r.read()
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"{self.role} camera: the engine sent a JPEG that does not decode")
        digest = hashlib.blake2b(raw, digest_size=16).digest()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, digest

    def _run(self):
        while not self._stop.is_set():
            t0 = mono()
            try:
                rgb, digest = self.fetch(timeout=max(0.5, 4 * self.period))
                self.shape = tuple(rgb.shape)
                if digest != self._digest:
                    t = mono()
                    with self._lock:
                        self.frame, self.stamp, self._digest = rgb, t, digest
                    self.times.append(t)
                self.error = None
            except urllib.error.HTTPError as e:
                self.error = f"{self.role} camera: the ACT engine answered {e.code} (is the camera plugged in?)"
            except Exception as e:
                self.error = f"{self.role} camera via {self.base}: {e}"
            dt = self.period - (mono() - t0)
            if dt > 0:
                self._stop.wait(dt)

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self.frame, (mono() - self.stamp) if self.stamp else math.inf

    def wait_first(self, timeout: float) -> bool:
        t_end = mono() + timeout
        while mono() < t_end:
            if self.frame is not None:
                return True
            time.sleep(0.02)
        return self.frame is not None

    def fps(self) -> float:
        t = list(self.times)
        if len(t) < 2 or mono() - t[-1] > 1.0:
            return 0.0
        return (len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 0.0

    def close(self):
        self._stop.set()
