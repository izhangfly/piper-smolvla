"""Planner and trajectory tests, no arm: the Ruckig follower's limits and continuity, and the planner's
choice between the straight line, joint space and via points.

    python3 -m unittest discover -s hardware/piper -p "test_*.py"
"""

import math
import os
import sys
import unittest

try:
    import numpy as np
    import ruckig  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

sys.path.insert(0, os.path.dirname(__file__))
if HAVE:
    import kinematics as K
    import motion as M
    import sim_smooth as S

BOX = {"x": (60.0, 480.0), "y": (-320.0, 320.0), "z": (80.0, 600.0)}
READY = [0.0, 45.0, -60.0, 0.0, 45.0, 0.0]


def run(fol, speed, ticks, on_tick=None):
    """Step the follower; returns the commanded joints, one row per tick."""
    out = [fol.q.copy()]
    for k in range(ticks):
        if on_tick:
            on_tick(k)
        out.append(fol.step(speed))
    return np.array(out)


def derivatives(q, dt=1.0 / 100):
    v = np.diff(q, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    return v, a, j


@unittest.skipUnless(HAVE, "needs numpy and ruckig")
class FollowerTest(unittest.TestCase):
    def test_respects_limits_and_arrives(self):
        for speed in (15, 50, 100):
            fol = M.Follower()
            fol.reset(READY)
            goal = np.array([60.0, 90.0, -20.0, 50.0, -30.0, 80.0])
            fol.set_target(goal)
            q = run(fol, speed, 1500)
            v, a, j = derivatives(q)
            vmax, amax, jmax = M.motion_limits(speed)
            self.assertTrue(np.all(np.abs(v) <= vmax * 1.001 + 1e-6), (speed, np.abs(v).max(axis=0)))
            self.assertTrue(np.all(np.abs(a) <= amax * 1.01 + 1e-3), (speed, np.abs(a).max(axis=0)))
            self.assertTrue(np.all(np.abs(j) <= jmax * 1.05 + 1.0), (speed, np.abs(j).max(axis=0)))
            np.testing.assert_allclose(q[-1], goal, atol=1e-6)
            self.assertTrue(fol.idle())

    def test_targets_at_20_hz_stay_continuous(self):
        """A target that moves every 50 ms (Hand Teleop, LLM Arm): velocity and acceleration stay
        continuous across every change, and the motion never stops in between."""
        fol = M.Follower()
        fol.reset(READY)
        speed = 50
        base = np.array(READY)

        def move(k):
            if k % 5 == 0:                     # every 50 ms
                t = k / 100.0
                fol.set_target(base + np.array([20.0, 10.0, -10.0, 15.0, 5.0, 20.0]) * math.sin(0.8 * t))
        q = run(fol, speed, 400, move)
        v, a, j = derivatives(q)
        vmax, amax, jmax = M.motion_limits(speed)
        self.assertTrue(np.all(np.abs(a) <= amax * 1.01 + 1e-3))
        self.assertTrue(np.all(np.abs(j) <= jmax * 1.05 + 1.0), np.abs(j).max(axis=0))
        speed_j1 = np.abs(v[50:150, 0])       # the J1 target moves at ~16 deg/s here
        self.assertGreater(speed_j1.min(), 4.0, "no stop-start between targets")

    def test_stop_brakes_where_it_said(self):
        fol = M.Follower()
        fol.reset(READY)
        fol.set_target(np.array(READY) + 60.0 * np.array([1, 0.5, -0.3, 1, 0.2, 1]))
        run(fol, 100, 60)
        self.assertTrue(fol.moving())
        where = fol.stop()
        q = run(fol, 100, 200)
        np.testing.assert_allclose(q[-1], where, atol=1e-6)
        stopped = int(np.argmax(np.all(np.abs(np.diff(q, axis=0)) < 1e-9, axis=1)))
        # From full speed at 100 % (the wrist at 120 deg/s) the brake limits stop it in about 0.6 s,
        # about as fast as the firmware's own 5 rad/s^2 joint deceleration would.
        self.assertLess(stopped, 70, f"took {stopped / 100:.2f} s to stop")
        v, a, _ = derivatives(q)
        self.assertTrue(np.all(np.abs(a) <= M.BRAKE_AMAX * 1.01 + 1e-3))

    def test_never_leaves_the_joint_limits(self):
        fol = M.Follower()
        fol.reset(READY)
        top = K.LIMITS_DEG[:, 1] - 0.5
        fol.set_target(top)
        q = run(fol, 100, 800)
        self.assertTrue(np.all(q <= K.LIMITS_DEG[:, 1] + 1e-9))
        np.testing.assert_allclose(q[-1], top, atol=1e-6)


@unittest.skipUnless(HAVE, "needs numpy and ruckig")
class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.kin = K.PiperKinematics(0x01)

    def planner(self, box=BOX):
        return M.Planner(self.kin, M.Workspace(dict(box)))

    def test_straight_line_when_feasible(self):
        pl = self.planner()
        start = self.kin.fk_pose(READY)
        goal = [start[0] + 60.0, start[1] + 40.0, start[2] - 50.0] + start[3:]
        plan, t, q, tcp = S.simulate(pl, np.array(READY), goal, speed=50)
        self.assertEqual(plan.kind, "line", plan.reason)
        a, b = np.array(start[:3]), np.array(goal[:3])
        u = (b - a) / np.linalg.norm(b - a)
        off = max(float(np.linalg.norm((p - a) - np.dot(p - a, u) * u)) for p in tcp)
        self.assertLess(off, 2.0, f"strayed {off:.2f} mm off the line")
        self.assertLess(math.dist(tcp[-1], goal[:3]), 0.05)
        # Flange speed cap for lines: 200 mm/s at speed 100 %, so 100 mm/s at 50 %.
        self.assertLess(np.max(np.linalg.norm(np.diff(tcp, axis=0), axis=1)) * M.CONTROL_HZ, 0.5 * M.TCP_MM_S * 1.1)

    def test_teleop_increments_plan_fast(self):
        pl = self.planner()
        start = self.kin.fk_pose(READY)
        for d in (2.0, 5.0, 10.0):
            plan = pl.plan_pose(np.array(READY), [start[0] + d, start[1] - d, start[2]] + start[3:])
            self.assertEqual(plan.kind, "line")
            self.assertLess(plan.ms, 25.0)

    def test_left_to_right_in_limited_space_backs_up_turns_and_extends(self):
        """The hard case. The straight line has no continuous IK; with the default box the plain
        joint-space swing works; with 250 mm of room in front, it would reach x = 337 mm, so the
        planner pulls the TCP in, turns, and extends again, and the traced TCP stays in the box."""
        case = S.CASES["limited"]
        wide = self.planner()
        q0 = S.start_joints(wide, case["start"])
        plan = wide.plan_pose(q0, case["goal"])
        self.assertEqual(plan.kind, "joint")
        self.assertIn("straight line", plan.reason)

        box = dict(BOX, x=(60.0, case["box_x_max"]))
        pl = self.planner(box)
        plan, t, q, tcp = S.simulate(pl, q0, case["goal"], speed=50)
        self.assertEqual(plan.kind, "via", plan.reason)
        self.assertIn("retract", plan.reason)
        self.assertLessEqual(tcp[:, 0].max(), box["x"][1] + 1.0)
        self.assertGreaterEqual(tcp[:, 0].min(), box["x"][0] - 1.0)
        self.assertGreaterEqual(tcp[:, 2].min(), box["z"][0] - 1.0)
        self.assertLess(math.dist(tcp[-1], case["goal"][:3]), 0.05)
        # It really backs up first: the reach shrinks before J1 turns far.
        reach = np.hypot(tcp[:, 0], tcp[:, 1])
        i_turn = int(np.argmax(np.abs(q[:, 0] - q[0, 0]) > 10.0))
        self.assertLess(reach[i_turn], reach[0] - 50.0)
        vmax, _, _ = M.motion_limits(50)
        self.assertTrue(np.all(np.abs(np.diff(q, axis=0)) * M.CONTROL_HZ <= vmax * 1.001 + 1e-6))

    def test_low_swing_lifts_first(self):
        """Low left to low right: joint space would dip below the box floor half way; the plan
        lifts first and the traced TCP stays above it."""
        case = S.CASES["low"]
        pl = self.planner()
        q0 = S.start_joints(pl, case["start"])
        jp, _ = M.densify([q0, pl.solve(q0, *K.pose_to_rt(case["goal"]))])
        self.assertLess(min(self.kin.points(x)["flange"][2] for x in jp), BOX["z"][0], "the plain swing dips")
        plan, t, q, tcp = S.simulate(pl, q0, case["goal"], speed=50)
        self.assertEqual(plan.kind, "via", plan.reason)
        self.assertIn("raise", plan.reason)
        self.assertGreaterEqual(tcp[:, 2].min(), BOX["z"][0] - 1.0)
        self.assertLess(math.dist(tcp[-1], case["goal"][:3]), 0.05)

    def test_goal_joints_are_the_nearest_solution(self):
        """A goal with a wrist-flipped twin inside the limits: the plan ends on the branch the arm
        is already on, never the flip."""
        a = np.array([10.0, 60.0, -70.0, 85.0, 30.0, 90.0])
        b = np.array([10.0, 60.0, -70.0, -95.0, -30.0, -90.0])
        pl = self.planner(dict(BOX, x=(-480.0, 480.0)))
        goal = self.kin.fk_pose(a)
        for start, want in ((a + [5, 5, -5, -5, 5, -5], a), (b + [5, 5, -5, 5, -5, 5], b)):
            plan = pl.plan_pose(start, goal)
            self.assertTrue(plan.moves, plan.reason)
            np.testing.assert_allclose(plan.path[-1], want, atol=0.05)

    def test_unreachable_goes_as_far_as_it_can(self):
        pl = self.planner(dict(BOX, x=(60.0, 900.0)))
        start = self.kin.fk_pose(READY)
        plan = pl.plan_pose(np.array(READY), [700.0, 0.0, start[2]] + start[3:])
        self.assertIn(plan.kind, ("partial", "unreachable"))
        if plan.kind == "partial":
            end = self.kin.fk(plan.path[-1])[0]
            self.assertGreater(end[0], start[0] + 50.0, "it got closer")
        self.assertIn("out of reach", plan.reason)

    def test_home_plan_is_joint_space(self):
        pl = self.planner()
        plan = pl.plan_joints([20.0, 70.0, -40.0, 10.0, 30.0, -20.0], READY)
        self.assertIn(plan.kind, ("joint", "via"))
        np.testing.assert_allclose(plan.path[-1], READY)


@unittest.skipUnless(HAVE, "needs numpy and ruckig")
class JogTest(unittest.TestCase):
    def test_jog_x_moves_only_x(self):
        kin = K.PiperKinematics(0x01)
        jog = M.CartesianJog(kin, M.Workspace(dict(BOX)))
        fol = M.Follower()
        fol.reset(READY)
        start = kin.fk_pose(READY)
        for _ in range(100):                            # 1 s held
            fol.set_path(jog.step(fol.q, [1, 0, 0, 0, 0, 0], 50, fol.dt), brisk=True)
            fol.step(50)
        fol.stop()
        for _ in range(100):
            fol.step(50)
        end = kin.fk_pose(fol.q)
        # The carrot ran 1 s at 100 mm/s; the arm takes about half a second to get up to speed.
        self.assertGreater(end[0] - start[0], 0.65 * M.JOG_MM_S * 0.5)
        self.assertLess(end[0] - start[0], 1.0 * M.JOG_MM_S * 0.5 + 10.0)
        self.assertLess(abs(end[1] - start[1]), 1.0)
        self.assertLess(abs(end[2] - start[2]), 1.0)
        self.assertLess(K.angle_deg(K.rot_sxyz(*end[3:]), K.rot_sxyz(*start[3:])), 0.5)

    def test_jog_stops_at_the_box(self):
        kin = K.PiperKinematics(0x01)
        box = dict(BOX, x=(60.0, 160.0))
        jog = M.CartesianJog(kin, M.Workspace(box))
        fol = M.Follower()
        fol.reset(READY)
        for _ in range(300):
            fol.set_path(jog.step(fol.q, [1, 0, 0, 0, 0, 0], 100, fol.dt), brisk=True)
            fol.step(100)
        self.assertLessEqual(kin.fk_pose(fol.q)[0], 160.0 + 0.6)


if __name__ == "__main__":
    unittest.main()
