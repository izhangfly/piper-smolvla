"""Tests for the PiPER driver that need no arm: a fake piper_sdk stands in for the CAN bus.

    python3 -m unittest discover -s hardware/piper -p "test_*.py"

The fake arm can follow its commands: JointCtrl targets are integrated into its joint state at the
speed cap the firmware's MOVE J would apply (the MotionCtrl_2 speed times its rated 172 deg/s), and
its end pose is the forward kinematics of those joints, so the smooth mode's control loop runs
end to end. Nothing here talks to a real arm.
"""

import collections
import json
import math
import sys
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace as NS

try:
    import numpy as np
    import ruckig  # noqa: F401
    HAVE_SMOOTH = True
except ImportError:
    HAVE_SMOOTH = False

sys.path.insert(0, __import__("os").path.dirname(__file__))
if HAVE_SMOOTH:
    import kinematics as K
    import motion as M

READY = [0.0, 45.0, -60.0, 0.0, 45.0, 0.0]
RATED_DEG_S = 172.0


class FakePiper:
    """Just enough of C_PiperInterface_V2. With follow off (the default), feedback is whatever the
    test sets; with follow on, JointCtrl moves the joints and the pose is their FK."""

    def __init__(self, iface):
        self.joints = list(READY)
        self._pose = [250.0, 0.0, 300.0, 180.0, 60.0, 0.0]
        self.sent = collections.deque(maxlen=20000)      # (time, name, args)
        self.follow = False              # JointCtrl moves the joints
        self.fk_feedback = False         # the end pose is the FK of the joints (else the fixed pose)
        self.stale = False
        self.offset = [0.0] * 6          # added to the reported joints (a fault the arm didn't command)
        self.rate = 50
        self.target = None
        self.lock = threading.Lock()
        self.kin = K.PiperKinematics(0x01) if HAVE_SMOOTH else None
        self._quit = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        dt = 0.005
        while not self._quit:
            time.sleep(dt)
            with self.lock:
                if self.follow and self.target is not None:
                    cap = RATED_DEG_S * self.rate / 100.0 * dt
                    self.joints = [j + max(-cap, min(cap, t - j)) for j, t in zip(self.joints, self.target)]

    def teleport(self, joints):
        with self.lock:
            self.joints = list(joints)
            self.target = list(joints)

    def measured(self):
        with self.lock:
            return [j + o for j, o in zip(self.joints, self.offset)]

    @property
    def pose(self):
        if self.fk_feedback:
            return self.kin.fk_pose(self.measured())
        return self._pose

    @pose.setter
    def pose(self, p):
        self._pose = list(p)

    def joint_cmds(self, since=0.0):
        return [(t, [v / 1000.0 for v in a]) for t, n, a in list(self.sent) if n == "JointCtrl" and t >= since]

    def ConnectPort(self):
        pass

    def isOk(self):
        return True

    def GetArmJointMsgs(self):
        js = NS(**{f"joint_{n + 1}": int(round(v * 1000)) for n, v in enumerate(self.measured())})
        return NS(time_stamp=time.time() - (10.0 if self.stale else 0.0), joint_state=js)

    def GetArmEndPoseMsgs(self):
        x, y, z, rx, ry, rz = (int(round(v * 1000)) for v in self.pose)
        return NS(end_pose=NS(X_axis=x, Y_axis=y, Z_axis=z, RX_axis=rx, RY_axis=ry, RZ_axis=rz))

    def GetArmStatus(self):
        s = NS(ctrl_mode=1, arm_status=0, teach_status=0, motion_status=0, mode_feed=0, err_status=NS())
        return NS(time_stamp=time.time(), arm_status=s)

    def GetArmGripperMsgs(self):
        return NS(gripper_state=NS(grippers_angle=0, grippers_effort=0))

    def GetArmEnableStatus(self):
        return [True] * 6

    def GetCanFps(self):
        return 200.0

    def GetPiperFirmwareVersion(self):
        return "fake"

    def JointCtrl(self, *a):
        self.sent.append((time.time(), "JointCtrl", a))
        with self.lock:
            self.target = [v / 1000.0 for v in a]

    def MotionCtrl_2(self, *a):
        self.sent.append((time.time(), "MotionCtrl_2", a))
        self.rate = a[2] if len(a) > 2 else self.rate

    def __getattr__(self, name):          # every other command: record it
        return lambda *a: self.sent.append((time.time(), name, a))


sys.modules["piper_sdk"] = types.SimpleNamespace(C_PiperInterface_V2=FakePiper)
import piper_webctl as W  # noqa: E402


def reset_state():
    with W.state_lock:
        W.STATE.update(targets=[0.0] * 6, mode="joint", pose=[0.0] * 6, jog=[0.0] * 6, jjog=[0.0] * 6,
                       servo_goal=None, enabled=False, estopped=False, last_heartbeat=0.0, owner=None,
                       guard=None, plan=None, speed=15)


class FlipGuardTest(unittest.TestCase):
    def test_turn_ignores_euler_wrap(self):
        self.assertAlmostEqual(W.turn_deg([0, 0, 0, 179, 0, 0], [0, 0, 0, -179, 0, 0]), 2.0, places=3)
        self.assertAlmostEqual(W.turn_deg([0, 0, 0, 0, 0, 10], [0, 0, 0, 0, 0, 40]), 30.0, places=3)

    def feed(self, g, n, joints_at, pose_at, dt=0.05):
        for i in range(n + 1):
            why = g.update(i * dt, joints_at(i / n), pose_at(i / n))
            if why:
                return why
        return None

    def test_ordinary_moves_pass(self):
        g = W.FlipGuard()
        base_j = [0.0, 45.0, -60.0, 0.0, 45.0, 0.0]
        # 60 mm of travel in 0.5 s with the shoulder joints moving ~20 deg.
        why = self.feed(g, 10, lambda w: [base_j[0] + 15 * w, base_j[1] + 20 * w, base_j[2] - 10 * w, 0, 45, 0],
                        lambda w: [250 + 60 * w, 0, 300, 180, 60, 0])
        self.assertIsNone(why)
        # A 50 deg wrist turn: J6 swings 50 deg, and the TCP turns as much.
        g.reset()
        why = self.feed(g, 10, lambda w: [0, 45, -60, 0, 45, 50 * w], lambda w: [250, 0, 300, 180, 60, 50 * w])
        self.assertIsNone(why)

    def test_flip_trips(self):
        g = W.FlipGuard()
        # J4 and J6 swing half a turn in opposite directions; the TCP hardly moves.
        why = self.feed(g, 8, lambda w: [0, 45, -60, 90 * w, 2, -90 * w], lambda w: [250 + 2 * w, 0, 300, 180, 60, 3 * w])
        self.assertIsNotNone(why)
        self.assertIn("J4", why)

    def test_old_samples_drop_out(self):
        g = W.FlipGuard()
        # The same 90 deg swing spread over 3 s is slow enough to be deliberate, not a flip.
        why = self.feed(g, 60, lambda w: [0, 45, -60, 90 * w, 2, -90 * w], lambda w: [250, 0, 300, 180, 60, 0])
        self.assertIsNone(why)

    def test_off(self):
        g = W.FlipGuard(limit_deg=0)
        self.assertIsNone(self.feed(g, 8, lambda w: [0, 45, -60, 90 * w, 2, -90 * w], lambda w: [250, 0, 300, 180, 60, 0]))


class HttpMixin:
    @classmethod
    def start(cls, cartesian):
        reset_state()
        W.ARM = W.Arm("can0", 15, cartesian=cartesian)
        cls.fake = W.ARM.piper
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), W.Handler)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        W.ARM.close()
        cls.fake._quit = True
        time.sleep(0.05)
        reset_state()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def get_state(self):
        with urllib.request.urlopen(self.base + "/api/state?lite=1", timeout=3) as r:
            return json.loads(r.read())


class DriverTest(HttpMixin, unittest.TestCase):
    """The firmware mode's control loop and HTTP API with the fake arm (its IK is the firmware's,
    so the fake only records EndPoseCtrl)."""

    @classmethod
    def setUpClass(cls):
        cls.start("firmware")

    def test_reports_firmware_mode(self):
        st = self.get_state()
        self.assertEqual(st["cartesian"], "firmware")
        self.assertEqual(st["control_hz"], W.STREAM_HZ)

    def test_servo_flip_holds_and_latches(self):
        self.assertEqual(self.post("/api/enable", {"on": True})[0], 200)
        code, body = self.post("/api/servo", {"client": "piperd", "pose": [260, 0, 300, 180, 60, 0]})
        self.assertEqual(code, 200, body)
        # The arm's IK flips the wrist while the goal is being streamed.
        for i in range(10):
            self.post("/api/servo", {"client": "piperd", "pose": [260, 0, 300, 180, 60, 0]})
            self.fake.joints = [0.0, 45.0, -60.0, 12.0 * i, 2.0, -12.0 * i]
            time.sleep(0.05)
        time.sleep(0.2)
        with W.state_lock:
            guard, mode, owner = W.STATE["guard"], W.STATE["mode"], W.STATE["owner"]
        self.assertIsNotNone(guard, "the guard tripped")
        self.assertEqual(mode, "joint", "the arm holds in joint mode")
        self.assertIsNone(owner)
        self.assertTrue(W.STATE["enabled"], "the motors stay on")
        code, body = self.post("/api/servo", {"client": "piperd", "pose": [260, 0, 300, 180, 60, 0]})
        self.assertEqual(code, 409)
        self.assertTrue(body.get("guard"))
        # After the latch, goals are accepted again.
        with W.state_lock:
            W.STATE["guard"]["time"] -= W.IK_GUARD_LATCH
        self.assertEqual(self.post("/api/servo", {"client": "piperd", "pose": [260, 0, 300, 180, 60, 0]})[0], 200)
        self.post("/api/stop", {})
        self.assertTrue(any(n == "EndPoseCtrl" for _, n, _ in self.fake.sent), "firmware mode streams MOVE P")


@unittest.skipUnless(HAVE_SMOOTH, "smooth mode needs numpy and ruckig")
class SmoothDriverTest(HttpMixin, unittest.TestCase):
    """The smooth mode end to end: planner, trajectory and 100 Hz JointCtrl into a fake arm that
    follows them."""

    @classmethod
    def setUpClass(cls):
        cls.start("smooth")
        cls.fake.follow = True
        cls.fake.fk_feedback = True
        code, body = cls.post(cls, "/api/enable", {"on": True})
        assert code == 200, body

    def setUp(self):
        self.post("/api/stop", {})
        deadline = time.time() + 3
        while W.ARM.follower.moving() and time.time() < deadline:
            time.sleep(0.02)
        self.fake.offset = [0.0] * 6
        self.fake.stale = False
        self.fake.teleport(READY)
        W.ARM.follower.deactivate()
        with W.state_lock:
            W.STATE.update(owner=None, guard=None, mode="joint", servo_goal=None, estopped=False, enabled=True,
                           targets=list(READY), jog=[0.0] * 6, jjog=[0.0] * 6)
            W.ARM._applied_targets = list(READY)
        W.ARM._last_plan = None
        W.ARM.fk_error_mm = None
        W.WORKSPACE_MM.update({"x": (60.0, 480.0), "y": (-320.0, 320.0), "z": (80.0, 600.0)})
        W.ARM.workspace.box = dict(W.WORKSPACE_MM)
        W.ARM.speed = 60
        time.sleep(0.05)

    # -- helpers -----------------------------------------------------------------------
    def servo_until(self, pose, client="piperd", hz=10.0, timeout=8.0, speed=60, tol_mm=1.0, sample=None):
        """Send the same goal at hz until the fake arm's TCP is there (as piperd's MoveToPose)."""
        t_end = time.time() + timeout
        last = None
        while time.time() < t_end:
            last = self.post("/api/servo", {"client": client, "pose": pose, "speed": speed})
            self.assertEqual(last[0], 200, last)
            if sample is not None:
                sample.append(self.fake.pose)
            p = self.fake.pose
            if math.dist(p[:3], pose[:3]) < tol_mm and not W.ARM.follower.moving():
                return last[1]
            time.sleep(1.0 / hz)
        self.fail(f"never reached {pose}: at {self.fake.pose}, last reply {last}")

    def check_smooth(self, cmds, speed):
        """The JointCtrl stream: one command per 10 ms tick, joint speeds and accelerations
        within the trajectory's limits (brakes within the brake limits)."""
        self.assertGreater(len(cmds), 10)
        q = np.array([c for _, c in cmds])
        t = np.array([t for t, _ in cmds])
        rate = (len(t) - 1) / (t[-1] - t[0])
        self.assertGreater(rate, 0.8 * M.CONTROL_HZ, f"stream rate {rate:.0f} Hz")
        dt = 1.0 / M.CONTROL_HZ
        v = np.diff(q, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        vmax, _, _ = M.motion_limits(speed)
        self.assertTrue(np.all(np.abs(v) <= vmax * 1.02 + 0.5), f"peak joint speed {np.abs(v).max(axis=0)}")
        self.assertTrue(np.all(np.abs(a) <= M.BRAKE_AMAX * 1.05 + 5), f"peak joint acc {np.abs(a).max(axis=0)}")

    # -- tests ---------------------------------------------------------------------------
    def test_state_reports_smooth_mode(self):
        self.post("/api/heartbeat", {})
        deadline = time.time() + 2.0
        while W.ARM.fk_error_mm is None and time.time() < deadline:
            self.post("/api/heartbeat", {})
            time.sleep(0.1)
        st = self.get_state()
        self.assertEqual(st["cartesian"], "smooth")
        self.assertEqual(st["control_hz"], M.CONTROL_HZ)
        self.assertIn("trajectory", st)
        self.assertLess(st["fk_error_mm"], 0.5, "our FK matches the (fake) arm's pose")

    def test_servo_line_is_straight_and_smooth(self):
        start = self.fake.pose
        goal = [start[0] + 60.0, start[1] + 40.0, start[2] - 50.0] + start[3:]
        t0 = time.time()
        seen = []
        reply = self.servo_until(goal, sample=seen)
        self.assertEqual(reply["plan"], "line")
        self.assertIn("JointCtrl", [n for _, n, _ in self.fake.sent])
        self.assertFalse(any(n == "EndPoseCtrl" for _, n, _ in self.fake.sent), "no firmware IK in smooth mode")
        self.check_smooth(self.fake.joint_cmds(t0), 60)
        # The commanded TCP stays on the straight line.
        a, b = np.array(start[:3]), np.array(goal[:3])
        u = (b - a) / np.linalg.norm(b - a)
        worst = 0.0
        for _, q in self.fake.joint_cmds(t0):
            p = np.array(W.ARM.kin.fk(q)[0])
            worst = max(worst, float(np.linalg.norm((p - a) - np.dot(p - a, u) * u)))
        self.assertLess(worst, 3.0, f"strayed {worst:.1f} mm off the line")

    def test_teleop_stream_is_continuous(self):
        """Goals at 20 Hz along y at 60 mm/s (Hand Teleop): the trajectory never stops between
        them. (From a comfortable pose: at READY the wrist centre is 56 mm from the J1 axis, and
        60 mm/s sideways there needs more joint speed than speed 60 allows.)"""
        self.fake.teleport([0.0, 80.0, -70.0, 0.0, 60.0, 0.0])
        W.ARM.follower.deactivate()
        start = self.fake.pose
        t0 = time.time()
        for k in range(30):
            goal = [start[0], start[1] + 3.0 * k, start[2]] + start[3:]
            self.assertEqual(self.post("/api/servo", {"client": "hand", "pose": goal, "speed": 60})[0], 200)
            time.sleep(0.05)
        cmds = self.fake.joint_cmds(t0)
        self.check_smooth(cmds, 60)
        tcp = np.array([W.ARM.kin.fk(q)[0] for _, q in cmds])
        speed = np.linalg.norm(np.diff(tcp, axis=0), axis=1) * M.CONTROL_HZ
        mid = speed[len(speed) // 3: 2 * len(speed) // 3]
        self.assertGreater(mid.min(), 30.0, f"the TCP slowed to {mid.min():.0f} mm/s between goals")
        self.assertLess(mid.max(), 90.0)

    def test_servo_deadman_brakes(self):
        start = self.fake.pose
        goal = [start[0] + 150.0, start[1], start[2] - 100.0] + start[3:]
        self.assertEqual(self.post("/api/servo", {"client": "piperd", "pose": goal, "speed": 60})[0], 200)
        time.sleep(0.3)
        self.assertTrue(W.ARM.follower.moving())
        time.sleep(W.SERVO_DEADMAN + 0.8)          # no more goals: the deadman lapses
        self.assertFalse(W.ARM.follower.moving(), "braked to a stop")
        n = len(self.fake.joint_cmds())
        time.sleep(0.2)
        self.assertEqual(len(self.fake.joint_cmds()), n, "and stopped streaming")
        self.assertGreater(math.dist(self.fake.pose[:3], goal[:3]), 20.0, "short of the goal")
        self.assertTrue(any(e["fn"] == "(deadman)" for e in W.STATE["log"]))

    def test_stop_brakes_quickly(self):
        start = self.fake.pose
        goal = [start[0] + 150.0, start[1] - 150.0, start[2] - 100.0] + start[3:]
        for _ in range(8):                       # keep the servo deadman fed, as piperd does
            self.post("/api/servo", {"client": "piperd", "pose": goal, "speed": 100})
            time.sleep(0.1)
        self.assertTrue(W.ARM.follower.moving())
        t_stop = time.time()
        self.assertEqual(self.post("/api/stop", {})[0], 200)
        self.assertEqual(W.STATE["mode"], "joint")
        deadline = time.time() + 1.0
        while W.ARM.follower.moving() and time.time() < deadline:
            time.sleep(0.01)
        self.assertLess(time.time() - t_stop, 0.6, "stopped within 0.6 s")
        cmds = self.fake.joint_cmds(t_stop - 0.2)
        self.check_smooth(cmds, 100)
        time.sleep(0.1)
        self.assertLess(max(abs(a - b) for a, b in zip(W.STATE["targets"], self.fake.measured())), 0.05,
                        "it holds where the brake ended")

    def test_estop_stops_streaming_at_once(self):
        start = self.fake.pose
        goal = [start[0] + 150.0, start[1], start[2] - 100.0] + start[3:]
        self.post("/api/servo", {"client": "piperd", "pose": goal, "speed": 60})
        time.sleep(0.3)
        self.assertEqual(self.post("/api/estop", {})[0], 200)
        t = time.time()
        self.assertTrue(any(n == "EmergencyStop" for _, n, _ in self.fake.sent))
        time.sleep(0.2)
        self.assertEqual(len(self.fake.joint_cmds(t + 0.02)), 0, "nothing streamed after the E-stop")
        with W.state_lock:
            W.STATE["estopped"] = False
            W.STATE["enabled"] = True

    def test_one_controller(self):
        start = self.fake.pose
        self.assertEqual(self.post("/api/servo", {"client": "piperd", "pose": start})[0], 200)
        code, body = self.post("/api/servo", {"client": "llm", "pose": start})
        self.assertEqual(code, 409)
        self.assertEqual(body.get("owner"), "piperd")

    def test_feedback_stale_brakes_and_stops(self):
        start = self.fake.pose
        goal = [start[0] + 150.0, start[1], start[2] - 100.0] + start[3:]
        stop = threading.Event()

        def keep_sending():
            while not stop.is_set():
                self.post("/api/servo", {"client": "piperd", "pose": goal, "speed": 60})
                time.sleep(0.1)
        th = threading.Thread(target=keep_sending, daemon=True)
        th.start()
        time.sleep(0.4)
        self.fake.stale = True
        time.sleep(0.8)
        n = len(self.fake.joint_cmds())
        time.sleep(0.2)
        stop.set()
        th.join()
        self.assertEqual(len(self.fake.joint_cmds()), n, "no commands while the feedback is stale")
        self.assertFalse(W.ARM.follower.moving())
        self.fake.stale = False

    def test_tracking_guard_holds(self):
        """The arm stops following (blocked): the driver holds the measured joints."""
        start = self.fake.pose
        goal = [start[0] + 150.0, start[1], start[2] - 100.0] + start[3:]
        self.post("/api/servo", {"client": "piperd", "pose": goal, "speed": 60})
        time.sleep(0.2)
        with self.fake.lock:
            self.fake.follow = False            # the joints freeze where they are
        try:
            for _ in range(20):
                self.post("/api/servo", {"client": "piperd", "pose": goal, "speed": 60})
                time.sleep(0.1)
                if W.STATE["guard"]:
                    break
            self.assertIsNotNone(W.STATE["guard"])
            self.assertIn("its command", W.STATE["guard"]["reason"])
            self.assertEqual(W.STATE["mode"], "joint")
            code, body = self.post("/api/servo", {"client": "piperd", "pose": goal})
            self.assertEqual(code, 409)
        finally:
            self.fake.follow = True

    def test_wrist_flip_trips_a_guard(self):
        """The (fake) arm's wrist swings by itself: one of the guards holds and latches."""
        start = self.fake.pose
        for i in range(12):
            self.post("/api/servo", {"client": "piperd", "pose": start})
            self.fake.offset = [0.0, 0.0, 0.0, 9.0 * i, 0.0, -9.0 * i]
            time.sleep(0.05)
            if W.STATE["guard"]:
                break
        self.assertIsNotNone(W.STATE["guard"])
        self.assertEqual(self.post("/api/servo", {"client": "piperd", "pose": start})[0], 409)

    def test_jog_moves_one_axis(self):
        start = self.fake.pose
        self.post("/api/speed", {"speed": 50})
        self.assertEqual(self.post("/api/jog", {"axis": 0, "dir": 1, "on": True})[0], 200)
        t0 = time.time()
        while time.time() - t0 < 1.0:
            self.post("/api/heartbeat", {})
            time.sleep(0.2)
        self.post("/api/jog", {"axis": 0, "dir": 1, "on": False})
        deadline = time.time() + 2
        while W.ARM.follower.moving() and time.time() < deadline:
            self.post("/api/heartbeat", {})
            time.sleep(0.05)
        end = self.fake.pose
        expected = M.JOG_MM_S * 0.5                       # held for ~1 s at speed 50
        self.assertGreater(end[0] - start[0], 0.6 * expected, f"moved {end[0] - start[0]:.0f} mm in x")
        self.assertLess(end[0] - start[0], 1.3 * expected + 10.0)
        self.assertLess(abs(end[1] - start[1]), 1.5)
        self.assertLess(abs(end[2] - start[2]), 1.5)
        self.check_smooth(self.fake.joint_cmds(t0), 50)

    def test_home_and_slider_go_through_the_trajectory(self):
        self.fake.teleport([20.0, 70.0, -40.0, 10.0, 30.0, -20.0])
        W.ARM.follower.deactivate()
        t0 = time.time()
        code, body = self.post("/api/home", {})
        self.assertEqual(code, 200, body)
        deadline = time.time() + 6
        while time.time() < deadline:
            self.post("/api/heartbeat", {})
            if max(abs(a - b) for a, b in zip(self.fake.measured(), READY)) < 0.2 and not W.ARM.follower.moving():
                break
            time.sleep(0.1)
        self.assertLess(max(abs(a - b) for a, b in zip(self.fake.measured(), READY)), 0.2)
        self.check_smooth(self.fake.joint_cmds(t0), 60)
        t1 = time.time()
        self.assertEqual(self.post("/api/joint", {"joint": 1, "deg": 25.0})[0], 200)
        deadline = time.time() + 4
        while time.time() < deadline and abs(self.fake.measured()[0] - 25.0) > 0.2:
            self.post("/api/heartbeat", {})
            time.sleep(0.1)
        self.assertLess(abs(self.fake.measured()[0] - 25.0), 0.2)
        self.check_smooth(self.fake.joint_cmds(t1), 60)

    # -- the owner's report, 2026-10-25: sliders stuck short of their ends, J5 dragging J2 -----
    def slide(self, joint, deg, timeout=10.0):
        """Move one slider and wait for the fake arm to settle; returns the measured joints."""
        code, body = self.post("/api/joint", {"joint": joint, "deg": deg})
        self.assertEqual(code, 200, body)
        deadline = time.time() + timeout
        time.sleep(0.15)
        while time.time() < deadline:
            self.post("/api/heartbeat", {})
            if not W.ARM.follower.moving():
                break
            time.sleep(0.1)
        time.sleep(0.2)
        return self.fake.measured()

    def test_sliders_reach_their_ends_and_never_freeze(self):
        limits = {j: tuple(v) for j, v in self.get_state()["limits"].items()}
        self.assertEqual(limits["3"], (-170.0, 0.0), "the panel is given the limits the follower plans in")
        self.assertEqual(limits["5"], (-70.0, 70.0))
        for joint, ask, expect in ((3, -175.0, -170.0 + M.EDGE_DEG), (2, 0.0, 0.0 + M.EDGE_DEG),
                                   (5, 75.0, 70.0 - M.EDGE_DEG), (3, -60.0, -60.0), (2, 58.3, 58.3), (2, 0.0, M.EDGE_DEG)):
            q = self.slide(joint, ask)
            self.assertAlmostEqual(q[joint - 1], expect, delta=0.3, msg=f"J{joint} -> {ask}: at {q}")
            self.assertIsNone(W.ARM.follower.fault, f"J{joint} -> {ask}")

    def test_one_slider_moves_one_joint(self):
        # A target J2 never reached (a hold or a fault left it behind), as the owner saw.
        with W.state_lock:
            W.STATE["targets"][1] = 5.0
            W.ARM._applied_targets = list(W.STATE["targets"])
        W.ARM.follower.reset(self.fake.measured())
        t0 = time.time()
        q = self.slide(5, 20.0)
        self.assertAlmostEqual(q[4], 20.0, delta=0.3)
        j2 = [c[1] for _, c in self.fake.joint_cmds(t0)]
        self.assertTrue(j2, "the arm was streamed to")
        self.assertLess(max(abs(v - READY[1]) for v in j2), 0.2, "moving J5 must not move J2")

    def test_left_to_right_in_limited_space(self):
        """The hard case through the API: a 250 mm deep box in front of the arm. The planner backs
        up, turns and extends, and the arm's TCP never leaves the box."""
        W.WORKSPACE_MM["x"] = (60.0, 250.0)
        W.ARM.workspace.box = dict(W.WORKSPACE_MM)
        left = [150.0, 300.0, 200.0, 180.0, 65.0, -110.0]
        right = [150.0, -300.0, 200.0, 180.0, 65.0, 110.0]
        q_left = W.ARM.planner.solve(np.array(READY), *K.pose_to_rt(left))
        self.fake.teleport(q_left)
        W.ARM.follower.deactivate()
        seen = []
        sampler_stop = threading.Event()

        def sample():
            while not sampler_stop.is_set():
                seen.append(self.fake.pose)
                time.sleep(0.01)
        th = threading.Thread(target=sample, daemon=True)
        th.start()
        try:
            reply = self.servo_until(right, speed=100, timeout=15.0, tol_mm=1.5)
        finally:
            sampler_stop.set()
            th.join()
        self.assertEqual(reply["plan"], "via", reply)
        self.assertIn("retract", reply["reason"])
        xs = [p[0] for p in seen]
        self.assertLess(max(xs), 250.0 + 2.0, f"the TCP reached x = {max(xs):.0f} mm")
        self.assertGreater(min(p[2] for p in seen), 80.0 - 2.0)


if __name__ == "__main__":
    unittest.main()
