"""Tests for the PiPER driver that need no arm: a fake piper_sdk stands in for the CAN bus.

    python3 -m unittest discover -s hardware/piper -p "test_*.py"
"""

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


class FakePiper:
    """Just enough of C_PiperInterface_V2: feedback is whatever the test sets."""

    def __init__(self, iface):
        self.joints = [0.0, 45.0, -60.0, 0.0, 45.0, 0.0]
        self.pose = [250.0, 0.0, 300.0, 180.0, 60.0, 0.0]
        self.sent = []

    def ConnectPort(self):
        pass

    def isOk(self):
        return True

    def GetArmJointMsgs(self):
        js = NS(**{f"joint_{n + 1}": int(round(v * 1000)) for n, v in enumerate(self.joints)})
        return NS(time_stamp=time.time(), joint_state=js)

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

    def __getattr__(self, name):          # every command: record it
        return lambda *a: self.sent.append((name, a))


sys.modules["piper_sdk"] = types.SimpleNamespace(C_PiperInterface_V2=FakePiper)
sys.path.insert(0, __import__("os").path.dirname(__file__))
import piper_webctl as W  # noqa: E402


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
        base_p = [250.0, 0.0, 300.0, 180.0, 60.0, 0.0]
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


class DriverTest(unittest.TestCase):
    """The control loop and HTTP API with the fake arm."""

    @classmethod
    def setUpClass(cls):
        W.ARM = W.Arm("can0", 15)
        cls.fake = W.ARM.piper
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), W.Handler)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        W.ARM.close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

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


if __name__ == "__main__":
    unittest.main()
