"""Kinematics tests: FK against piper_sdk's and the arm's own numbers, IK round trips, nearest solutions.

    python3 -m unittest discover -s hardware/piper -p "test_*.py"
"""

import math
import os
import sys
import unittest

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

sys.path.insert(0, os.path.dirname(__file__))
if HAVE_NUMPY:
    import kinematics as K

try:      # the real SDK's FK, when it is installed (the driver's venv); the reference values below always run
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics
except Exception:
    C_PiperForwardKinematics = None

# (joints deg, end pose [x, y, z mm, rx, ry, rz deg]) the arm itself reported.
MEASURED = [
    # READY, legacy/piper-act/docs/HARDWARE.md section 4
    ([0.0, 45.0, -60.0, 0.0, 45.0, 0.0], [138.0, 0.0, 418.0, 180.0, 65.0, 180.0], 1.0),
    # joint zero, ibid. (piper_sdk's init_pos for this DH variant: 56.128, 0, 213.266)
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [56.0, 0.0, 213.0, 0.0, 85.0, 0.0], 1.0),
    # the Surface's arm resting folded, firmware S-V1.9-0, read from GET /api/state (2026-10-25)
    ([1.005, -2.27, 0.132, -5.757, 30.912, 3.386], [45.685, -3.889, 166.883, -173.511, 66.224, -175.134], 0.01),
]


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class FKTest(unittest.TestCase):
    def setUp(self):
        self.kin = K.PiperKinematics(0x01)

    def test_matches_the_arm(self):
        for q, pose, tol in MEASURED:
            got = self.kin.fk_pose(q)
            self.assertLess(math.dist(got[:3], pose[:3]), tol + 0.5, (q, got, pose))
            self.assertLess(K.angle_deg(K.rot_sxyz(*got[3:]), K.rot_sxyz(*pose[3:])), 0.1, (q, got, pose))

    def test_euler_branch_is_the_firmwares(self):
        rx, ry, rz = self.kin.fk_pose([0.0, 45.0, -60.0, 0.0, 45.0, 0.0])[3:]
        self.assertAlmostEqual(abs(rx), 180.0, places=3)
        self.assertAlmostEqual(ry, 65.0, places=3)
        self.assertAlmostEqual(abs(rz), 180.0, places=3)

    @unittest.skipUnless(C_PiperForwardKinematics, "piper_sdk not installed")
    def test_matches_piper_sdk(self):
        rng = np.random.default_rng(0)
        lo, hi = K.joint_bounds()
        for variant in (0x00, 0x01):
            kin, sdk = K.PiperKinematics(variant), C_PiperForwardKinematics(variant)
            for _ in range(200):
                q = rng.uniform(lo, hi)
                a = kin.fk_pose(q)
                b = sdk.CalFK(np.radians(q).tolist())[-1]
                self.assertLess(math.dist(a[:3], b[:3]), 1e-6)
                self.assertLess(K.angle_deg(K.rot_sxyz(*a[3:]), K.rot_sxyz(*b[3:])), 1e-4)

    def test_dh_variant_selection(self):
        q, pose, _ = MEASURED[2]
        kin, err = K.select_dh_variant(q, pose)
        self.assertEqual(kin.dh_variant, 0x01)
        self.assertLess(err, 0.05)

    def test_jacobian_matches_finite_differences(self):
        q = np.radians([10.0, 60.0, -70.0, 20.0, 40.0, -30.0])
        J = self.kin.jacobian_rad(q)
        p0, R0 = self.kin.fk_rad(q)
        for i in range(6):
            dq = q.copy()
            dq[i] += 1e-6
            p1, R1 = self.kin.fk_rad(dq)
            np.testing.assert_allclose(J[:3, i], (p1 - p0) / 1e-6, atol=1e-5)
            np.testing.assert_allclose(J[3:, i], K.rotvec(R1 @ R0.T) / 1e-6, atol=1e-5)

    def test_manipulability_sees_the_singularities(self):
        good = self.kin.manipulability([0.0, 100.0, -60.0, 0.0, 45.0, 0.0])
        self.assertGreater(good, 0.2)
        self.assertLess(self.kin.manipulability([0.0, 80.0, -60.0, 0.0, 0.5, 0.0]), 0.01)   # wrist, J5 = 0
        self.assertLess(self.kin.manipulability([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.01)      # joint zero


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class RotationTest(unittest.TestCase):
    def test_euler_round_trip(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            e = [rng.uniform(-180, 180), rng.uniform(-89, 89), rng.uniform(-180, 180)]
            R = K.rot_sxyz(*e)
            self.assertLess(K.angle_deg(R, K.rot_sxyz(*K.euler_sxyz(R))), 1e-4)

    def test_rotvec_round_trip_including_half_turns(self):
        rng = np.random.default_rng(2)
        for th in list(rng.uniform(0, math.pi, 100)) + [math.pi - 1e-5, math.pi, 1e-9]:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            R = K.exp_rotvec(axis * th)
            self.assertLess(K.angle_deg(R, K.exp_rotvec(K.rotvec(R))), 1e-4)


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class IKTest(unittest.TestCase):
    def setUp(self):
        self.kin = K.PiperKinematics(0x01)
        self.rng = np.random.default_rng(3)
        self.lo, self.hi = K.joint_bounds()

    def test_dls_round_trip_from_a_nearby_seed(self):
        ok = 0
        for _ in range(200):
            q = self.rng.uniform(self.lo, self.hi)
            pose = self.kin.fk_pose(q)
            sol = self.kin.ik(pose, q + self.rng.uniform(-5, 5, 6))
            if sol is None:
                continue
            ok += 1
            got = self.kin.fk_pose(sol)
            self.assertLess(math.dist(got[:3], pose[:3]), 0.6)
            self.assertLess(K.angle_deg(K.rot_sxyz(*got[3:]), K.rot_sxyz(*pose[3:])), 0.6)
            self.assertTrue(np.all(sol >= self.lo - 1e-9) and np.all(sol <= self.hi + 1e-9))
        self.assertGreater(ok, 190)

    def test_closed_form_finds_every_branch(self):
        for _ in range(300):
            q = self.rng.uniform(self.lo, self.hi)
            p, R = self.kin.fk(q)
            sols = self.kin.ik_all_rt(p, R, q)
            self.assertTrue(any(np.max(np.abs(s - q)) < 1e-3 for s in sols), q)

    def test_nearest_solution_never_flips_the_wrist(self):
        # Two solutions inside the limits: the wrist flip (J4 +- 180, -J5, J6 +- 180).
        a = np.array([10.0, 60.0, -70.0, 85.0, 30.0, 90.0])
        b = np.array([10.0, 60.0, -70.0, -95.0, -30.0, -90.0])
        pa, pb = self.kin.fk_pose(a), self.kin.fk_pose(b)
        self.assertLess(math.dist(pa[:3], pb[:3]), 1e-6, "they are the same pose")
        for seed, want in ((a, a), (b, b)):
            near = seed + self.rng.uniform(-3, 3, 6)
            dls = self.kin.ik(pa, near)
            self.assertIsNotNone(dls)
            self.assertLess(np.max(np.abs(dls - want)), 1.0, "DLS stays on the seed's branch")
            self.assertLess(np.max(np.abs(self.kin.nearest_ik(pa, near) - want)), 1e-3)

    def test_seed_distance_guard(self):
        # The only solution inside the limits is 60 deg from the seed on J4 and J6: a solve that
        # has to wander that far is refused (streaming it would swing the wrist).
        q = np.array([30.0, 80.0, -60.0, 20.0, 40.0, -10.0])
        pose = self.kin.fk_pose(q)
        p, R = K.pose_to_rt(pose)
        self.assertEqual(len(self.kin.ik_all_rt(p, R, q)), 1)
        far = q + np.array([0.0, 0.0, 0.0, 60.0, 0.0, -60.0])
        self.assertIsNone(self.kin.ik(pose, far, max_seed_dist_deg=10.0))
        free = self.kin.ik(pose, far, max_seed_dist_deg=0.0, max_iters=60)
        self.assertIsNotNone(free)
        self.assertLess(np.max(np.abs(free - q)), 0.5)

    def test_unreachable_is_none(self):
        self.assertIsNone(self.kin.ik([900.0, 0.0, 300.0, 180.0, 0.0, 180.0], [0, 45, -60, 0, 45, 0]))
        p, R = K.pose_to_rt([900.0, 0.0, 300.0, 180.0, 0.0, 180.0])
        self.assertEqual(self.kin.ik_all_rt(p, R), [])


if __name__ == "__main__":
    unittest.main()
