"""PiPER kinematics: forward kinematics and damped-least-squares inverse kinematics.

Adapted from showlab/Show-Harness core/piper/kinematics.py, whose FK is the vendor's own
Denavit-Hartenberg model (piper_sdk/kinematics/piper_fk.py, C_PiperForwardKinematics) and whose IK
is damped least squares on the 6-D pose error, seeded at the previous solution, clamped to the
joint limits every iterate, with a near-seed continuity check and a position-first fallback
(ik_bounded). What changed here:

  * degrees and millimetres at the public functions (the driver's units); metres and radians inside,
    so the damping and tolerances tuned in Show-Harness still apply;
  * no scipy: the rotation helpers (log/exp map, sxyz Euler) are numpy;
  * an analytic geometric Jacobian from the DH frames instead of a numeric one (one FK pass);
  * the end pose is the PiPER's own: x, y, z of the flange, and static-XYZ Euler angles,
    R = Rz(rz) Ry(ry) Rx(rx), on the branch piper_sdk and the firmware report (ry in [-90, 90]);
  * the joint limits are the stricter of the AgileX manual's and piper_sdk's;
  * the link points (elbow, wrist centre, flange) and a manipulability measure for the planner.

Checked against piper_sdk's FK and the arm: DH variant 0x01 (firmware >= V1.5-2, joints 2 and 3
offset by 2 deg) gives the READY pose (0, 45, -60, 0, 45, 0) deg -> [138.2, 0, 417.9, 180, 65, 180],
measured as [138, 0, 418, 180, 65, 180], and matches the S-V1.9-0 arm's reported end pose to
0.01 mm. See test_kinematics.py.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

DOF = 6

# Joint limits (deg): the stricter of the AgileX manual (the driver's JOINT_LIMITS) and
# piper_sdk/piper_param (j1 +-150, j2 0..180, j3 -170..0, j4 +-100, j5 +-70, j6 +-120).
LIMITS_DEG = np.array([
    [-150.0, 150.0],
    [0.0, 180.0],
    [-170.0, 0.0],
    [-100.0, 100.0],
    [-70.0, 70.0],
    [-100.0, 100.0],
])
# IK solutions stay this far inside the limits, so a streamed target never trips the firmware's
# joint-limit error and the planner keeps room to brake.
LIMIT_MARGIN_DEG = 2.0

# Vendored DH tables (a, d in mm; alpha, theta offset in rad), piper_sdk C_PiperForwardKinematics.
# dh_is_offset 0x00 is older firmware; 0x01 (firmware >= V1.5-2) offsets joints 2 and 3 by 2 deg.
_DH_VARIANTS = {
    0x00: {
        "a": [0.0, 0.0, 285.03, -21.98, 0.0, 0.0],
        "alpha": [0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, math.pi / 2],
        "theta": [0.0, -math.pi * 174.22 / 180, -100.78 / 180 * math.pi, 0.0, 0.0, 0.0],
        "d": [123.0, 0.0, 0.0, 250.75, 0.0, 91.0],
    },
    0x01: {
        "a": [0.0, 0.0, 285.03, -21.98, 0.0, 0.0],
        "alpha": [0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, math.pi / 2],
        "theta": [0.0, -math.pi * 172.22 / 180, -102.78 / 180 * math.pi, 0.0, 0.0, 0.0],
        "d": [123.0, 0.0, 0.0, 250.75, 0.0, 91.0],
    },
}
FLANGE_MM = 91.0     # wrist centre to flange, along the last joint axis


# ---------------------------------------------------------------- rotations
def rot_sxyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation matrix of a PiPER end pose's RX, RY, RZ (deg): R = Rz Ry Rx."""
    a, b, c = math.radians(rx), math.radians(ry), math.radians(rz)
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    return np.array([[cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
                     [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
                     [-sb, cb * sa, cb * ca]])


def euler_sxyz(R: np.ndarray) -> Tuple[float, float, float]:
    """RX, RY, RZ (deg) of a rotation, on the branch piper_sdk's FK and the firmware report:
    ry in [-90, 90], and at gimbal lock rz = 0 with the yaw folded into rx."""
    r20 = float(R[2, 0])
    if r20 < -1 + 1e-4:
        return math.degrees(math.atan2(R[0, 1], R[1, 1])), 90.0, 0.0
    if r20 > 1 - 1e-4:
        return -math.degrees(math.atan2(R[0, 1], R[1, 1])), -90.0, 0.0
    ry = math.atan2(-r20, math.hypot(R[0, 0], R[1, 0]))
    return (math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(ry),
            math.degrees(math.atan2(R[1, 0], R[0, 0])))


def rotvec(R: np.ndarray) -> np.ndarray:
    """Log map: the rotation vector (axis * angle, rad) of a rotation matrix."""
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-7:
        return 0.5 * w
    if th > math.pi - 1e-3:
        # Near a half turn the antisymmetric part vanishes: the axis comes from the symmetric
        # part, (R + R^T) / 2 = cos(th) I + (1 - cos(th)) a a^T, and its sign from w.
        S = (np.asarray(R) + np.asarray(R).T) / 2.0
        B = (S - c * np.eye(3)) / (1.0 - c)
        k = int(np.argmax(np.diag(B)))
        axis = B[:, k] / math.sqrt(max(B[k, k], 1e-12))
        if float(np.dot(axis, w)) < 0:
            axis = -axis
        return axis / np.linalg.norm(axis) * th
    return w * (th / (2.0 * math.sin(th)))


def exp_rotvec(w: np.ndarray) -> np.ndarray:
    """Exp map (Rodrigues): the rotation matrix of a rotation vector (rad)."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w, dtype=float) / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle (deg) of the rotation between two orientations."""
    c = max(-1.0, min(1.0, (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0))
    return math.degrees(math.acos(c))


def pose_to_rt(pose: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """[x, y, z mm, rx, ry, rz deg] -> (position in mm, rotation matrix)."""
    return np.asarray(pose[:3], dtype=float), rot_sxyz(*pose[3:6])


def rt_to_pose(p_mm: np.ndarray, R: np.ndarray) -> List[float]:
    return [float(p_mm[0]), float(p_mm[1]), float(p_mm[2]), *euler_sxyz(R)]


def interpolate_pose(pa: np.ndarray, Ra: np.ndarray, pb: np.ndarray, Rb: np.ndarray, w: float):
    """A point of the straight line between two poses: position lerp, orientation slerp."""
    return pa + (pb - pa) * w, Ra @ exp_rotvec(rotvec(Ra.T @ Rb) * w)


# ---------------------------------------------------------------- the arm
def joint_bounds(margin_deg: float = LIMIT_MARGIN_DEG) -> Tuple[np.ndarray, np.ndarray]:
    return LIMITS_DEG[:, 0] + margin_deg, LIMITS_DEG[:, 1] - margin_deg


class PiperKinematics:
    """FK, Jacobian and DLS IK for one DH variant.

    The *_rad methods work in metres and radians (Show-Harness's units, which its damping and
    tolerances were tuned in); the others take and return degrees and millimetres."""

    def __init__(self, dh_variant: int = 0x01, margin_deg: float = LIMIT_MARGIN_DEG) -> None:
        if dh_variant not in _DH_VARIANTS:
            raise ValueError(f"dh_variant must be 0x00 or 0x01, got {dh_variant!r}")
        self.dh_variant = dh_variant
        dh = _DH_VARIANTS[dh_variant]
        self._a = np.asarray(dh["a"], dtype=float) / 1000.0
        self._theta0 = np.asarray(dh["theta"], dtype=float)
        self._d = np.asarray(dh["d"], dtype=float) / 1000.0
        alpha = np.asarray(dh["alpha"], dtype=float)
        self._ca, self._sa = np.cos(alpha), np.sin(alpha)
        lo, hi = joint_bounds(margin_deg)
        self.lo_rad, self.hi_rad = np.radians(lo), np.radians(hi)

    # -- forward -------------------------------------------------------------
    def frames(self, q_rad: np.ndarray) -> List[np.ndarray]:
        """T_0i for i = 1..6 (modified DH, exactly as the vendor FK chains them)."""
        out, T = [], np.eye(4)
        for i in range(DOF):
            ct, st = math.cos(q_rad[i] + self._theta0[i]), math.sin(q_rad[i] + self._theta0[i])
            ca, sa, a, d = self._ca[i], self._sa[i], self._a[i], self._d[i]
            T = T @ np.array([[ct, -st, 0.0, a],
                              [st * ca, ct * ca, -sa, -sa * d],
                              [st * sa, ct * sa, ca, ca * d],
                              [0.0, 0.0, 0.0, 1.0]])
            out.append(T)
        return out

    def fk_rad(self, q_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Joints (rad) -> (flange position m, rotation 3x3) in the base frame."""
        T = self.frames(np.asarray(q_rad, dtype=float))[-1]
        return T[:3, 3].copy(), T[:3, :3].copy()

    def fk(self, q_deg: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Joints (deg) -> (flange position mm, rotation 3x3)."""
        p, R = self.fk_rad(np.radians(np.asarray(q_deg, dtype=float)))
        return p * 1000.0, R

    def fk_pose(self, q_deg: Sequence[float]) -> List[float]:
        """Joints (deg) -> the PiPER end pose [x, y, z mm, rx, ry, rz deg]."""
        p, R = self.fk(q_deg)
        return rt_to_pose(p, R)

    def points(self, q_deg: Sequence[float]) -> dict:
        """Link points (mm) the planner keeps clear: elbow (J3 axis), wrist centre, flange."""
        F = self.frames(np.radians(np.asarray(q_deg, dtype=float)))
        return {"elbow": F[2][:3, 3] * 1000.0, "wrist": F[3][:3, 3] * 1000.0,
                "flange": F[5][:3, 3] * 1000.0, "R": F[5][:3, :3]}

    def jacobian_rad(self, q_rad: np.ndarray, F: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Geometric Jacobian d[flange velocity m/s, angular velocity rad/s] / dq, base frame.
        In modified DH the axis of joint i is z of frame i, through frame i's origin."""
        F = F if F is not None else self.frames(q_rad)
        pe = F[-1][:3, 3]
        J = np.empty((6, DOF))
        for i in range(DOF):
            z, o = F[i][:3, 2], F[i][:3, 3]
            J[:3, i] = np.cross(z, pe - o)
            J[3:, i] = z
        return J

    def manipulability(self, q_deg: Sequence[float], length_m: float = 0.3) -> float:
        """Smallest singular value of the Jacobian with translation scaled by a 0.3 m lever, so a
        unit joint speed moving the flange 0.3 m/s counts like one turning it 1 rad/s. Near 0 at a
        singularity (the wrist at J5 = 0, the elbow straight, the wrist centre on the J1 axis)."""
        J = self.jacobian_rad(np.radians(np.asarray(q_deg, dtype=float)))
        J[:3] /= length_m
        return float(np.linalg.svd(J, compute_uv=False)[-1])

    # -- inverse -------------------------------------------------------------
    def _clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lo_rad, self.hi_rad)

    def ik_rad(self, target_pos: np.ndarray, target_rot: np.ndarray, q_seed: np.ndarray,
               pos_tol_m: float = 5e-4, ori_tol_rad: float = math.radians(0.5),
               max_iters: int = 16, damping: float = 0.05,
               max_seed_dist_rad: float = 0.35) -> Optional[np.ndarray]:
        """Damped-least-squares IK (Show-Harness's solver). Returns joints (rad) or None.

        Seeded from q_seed; joint limits are clamped every iterate. A converged solution farther
        than max_seed_dist_rad from the seed on any joint is rejected (it means DLS wandered to
        another branch, e.g. a wrist flip); 0 turns that check off. A solve that stalls under the
        default damping continues once with light damping from where it stalled (the continuity
        check is still against the original seed)."""
        seed = np.asarray(q_seed, dtype=float).reshape(-1)[:DOF]
        q = self._clip(seed.copy())
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        target_rot = np.asarray(target_rot, dtype=float).reshape(3, 3)
        eye6 = np.eye(6)
        for lam in (damping, 0.01):
            for _ in range(max_iters + 1):
                F = self.frames(q)
                pos, rot = F[-1][:3, 3], F[-1][:3, :3]
                e_pos = target_pos - pos
                e_ori = rotvec(target_rot @ rot.T)
                if float(np.linalg.norm(e_pos)) < pos_tol_m and float(np.linalg.norm(e_ori)) < ori_tol_rad:
                    return q if self._near_seed(q, seed, max_seed_dist_rad) else None
                J = self.jacobian_rad(q, F)
                # dq = J^T (J J^T + lambda^2 I)^-1 err: damped, singularity-safe.
                dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * eye6, np.concatenate([e_pos, e_ori]))
                step = float(np.max(np.abs(dq)))
                if step > 0.2:                      # trust region: never fling a joint
                    dq *= 0.2 / step
                q = self._clip(q + dq)
        return None

    def ik_position_rad(self, target_pos: np.ndarray, q_seed: np.ndarray, pos_tol_m: float = 5e-4,
                        max_iters: int = 24, damping: float = 0.03,
                        max_seed_dist_rad: float = 0.35) -> Optional[np.ndarray]:
        """Position-only DLS IK: reach target_pos with whatever orientation results."""
        q = self._clip(np.asarray(q_seed, dtype=float).reshape(-1)[:DOF].copy())
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        for _ in range(max_iters):
            F = self.frames(q)
            e_pos = target_pos - F[-1][:3, 3]
            if float(np.linalg.norm(e_pos)) < pos_tol_m:
                return q if self._near_seed(q, q_seed, max_seed_dist_rad) else None
            J = self.jacobian_rad(q, F)[:3]
            dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(3), e_pos)
            step = float(np.max(np.abs(dq)))
            if step > 0.2:
                dq *= 0.2 / step
            q = self._clip(q + dq)
        pos, _ = self.fk_rad(q)
        if float(np.linalg.norm(target_pos - pos)) < pos_tol_m:
            return q if self._near_seed(q, q_seed, max_seed_dist_rad) else None
        return None

    def ik_bounded_rad(self, target_pos: np.ndarray, target_rot: np.ndarray, q_seed: np.ndarray,
                       max_ori_dev_rad: float, max_seed_dist_rad: float = 0.35
                       ) -> Tuple[Optional[np.ndarray], float]:
        """Position-exact IK with the orientation held within max_ori_dev_rad of target_rot.
        Returns (joints, deviation rad) or (None, 0). Strict solve first; then the orientation the
        mechanism naturally adopts there (position-only IK), accepted if within the budget; else
        the target orientation bent by exactly the budget toward it and solved again."""
        budget = max(0.0, float(max_ori_dev_rad))
        sol = self.ik_rad(target_pos, target_rot, q_seed, max_seed_dist_rad=max_seed_dist_rad)
        if sol is not None or budget <= 0.0:
            return sol, 0.0
        q_free = self.ik_position_rad(target_pos, q_seed, max_seed_dist_rad=max_seed_dist_rad)
        if q_free is None:
            return None, 0.0
        _, rot_free = self.fk_rad(q_free)
        dev_vec = rotvec(rot_free @ np.asarray(target_rot, float).T)
        dev = float(np.linalg.norm(dev_vec))
        if dev <= budget:
            return q_free, dev
        bent = exp_rotvec(dev_vec / dev * budget) @ target_rot
        sol = self.ik_rad(target_pos, bent, q_seed, ori_tol_rad=math.radians(2.0),
                          max_seed_dist_rad=max_seed_dist_rad)
        return (sol, budget) if sol is not None else (None, 0.0)

    # -- closed form --------------------------------------------------------------
    def ik_all_rt(self, p_mm: np.ndarray, R: np.ndarray, q_hint_deg: Optional[Sequence[float]] = None,
                  margin_deg: float = LIMIT_MARGIN_DEG) -> List[np.ndarray]:
        """Every IK solution (deg) inside the joint limits, closed form.

        The PiPER has a spherical wrist (J4-J6 axes meet at the wrist centre), so: wrist centre ->
        J1 (front and back), J2/J3 from the planar two-link problem (two elbow branches), J4-J6
        from R36 as ZYZ Euler angles (two wrist branches): up to 8 solutions. Structure adapted
        from agilexrobotics/Agilex-College piper/piper_kinematics/include/piper_analytical_ik.hpp;
        the planar and wrist steps are re-derived for the modified-DH frames above, and every
        candidate is checked by FK, so a convention slip can only lose solutions, never return a
        wrong one. At the wrist singularity (J5 = 0) J4 keeps q_hint's value."""
        p = np.asarray(p_mm, dtype=float) / 1000.0
        R = np.asarray(R, dtype=float)
        wc = p - self._d[5] * R[:, 2]
        a2, a3, d4, d1 = self._a[2], self._a[3], self._d[3], self._d[0]
        vx, vy = a3, -d4                                   # wrist centre in frame 3
        L3sq = vx * vx + vy * vy
        hint = None if q_hint_deg is None else np.radians(np.asarray(q_hint_deg, dtype=float))
        lo, hi = joint_bounds(margin_deg)
        out: List[np.ndarray] = []
        h = wc[2] - d1
        base_az = math.atan2(wc[1], wc[0]) if math.hypot(wc[0], wc[1]) > 1e-9 else \
            (0.0 if hint is None else hint[0] + self._theta0[0])
        for t1 in (base_az, base_az + math.pi):
            rho = math.cos(t1) * wc[0] + math.sin(t1) * wc[1]
            ux, uy = rho, -h                               # wrist centre in frame 2 before J2 turns
            C = (ux * ux + uy * uy - a2 * a2 - L3sq) / (2.0 * a2)
            A, B = vx, -vy
            r = math.hypot(A, B)
            if abs(C) > r + 1e-9:
                continue
            base = math.atan2(B, A)
            dlt = math.acos(max(-1.0, min(1.0, C / r)))
            for t3 in (base + dlt, base - dlt):
                wx = a2 + math.cos(t3) * vx - math.sin(t3) * vy
                wy = math.sin(t3) * vx + math.cos(t3) * vy
                t2 = math.atan2(uy, ux) - math.atan2(wy, wx)
                q123 = np.array([t1, t2, t3]) - self._theta0[:3]
                F = self.frames(np.r_[q123, 0.0, 0.0, 0.0])
                M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]) @ (F[2][:3, :3].T @ R)
                s5 = math.hypot(M[0, 2], M[1, 2])
                wrists = []
                if s5 < 1e-6:
                    t4 = 0.0 if hint is None else hint[3]
                    wrists.append((t4, 0.0, math.atan2(M[1, 0], M[0, 0]) - t4))
                else:
                    t5 = math.atan2(s5, M[2, 2])
                    t4 = math.atan2(M[1, 2], M[0, 2])
                    t6 = math.atan2(M[2, 1], -M[2, 0])
                    wrists += [(t4, t5, t6), (t4 + math.pi, -t5, t6 + math.pi)]
                for t4, t5, t6 in wrists:
                    q = np.degrees(np.r_[q123, t4, t5, t6] - np.r_[0.0, 0.0, 0.0, self._theta0[3:]])
                    q = (q + 180.0) % 360.0 - 180.0
                    if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
                        continue
                    pf, Rf = self.fk(q)
                    if float(np.linalg.norm(pf - p * 1000.0)) < 0.5 and angle_deg(Rf, R) < 0.5:
                        out.append(q)
        return out

    def nearest_ik(self, pose: Sequence[float], q_ref_deg: Sequence[float]) -> Optional[np.ndarray]:
        """The closed-form solution nearest q_ref (largest single-joint change), or None."""
        p, R = pose_to_rt(pose)
        sols = self.ik_all_rt(p, R, q_ref_deg)
        if not sols:
            return None
        ref = np.asarray(q_ref_deg, dtype=float)
        return min(sols, key=lambda q: float(np.max(np.abs(q - ref))))

    @staticmethod
    def _near_seed(q: np.ndarray, q_seed: np.ndarray, max_dist_rad: float) -> bool:
        if max_dist_rad <= 0.0:
            return True
        seed = np.asarray(q_seed, dtype=float).reshape(-1)[:DOF]
        return float(np.max(np.abs(q - seed))) <= max_dist_rad

    # -- degree / millimetre wrappers ------------------------------------------
    def ik(self, pose: Sequence[float], seed_deg: Sequence[float], max_seed_dist_deg: float = 20.0,
           ori_slack_deg: float = 0.0, max_iters: int = 16) -> Optional[np.ndarray]:
        """IK of an end pose [x, y, z mm, rx, ry, rz deg], seeded at seed_deg. Returns joints (deg)
        or None. ori_slack_deg > 0 lets the orientation give that much where the exact one is out
        of reach (ik_bounded); max_seed_dist_deg 0 allows any distance from the seed."""
        p, R = pose_to_rt(pose)
        return self.ik_rt(p, R, seed_deg, max_seed_dist_deg, ori_slack_deg, max_iters)

    def ik_rt(self, p_mm: np.ndarray, R: np.ndarray, seed_deg: Sequence[float],
              max_seed_dist_deg: float = 20.0, ori_slack_deg: float = 0.0,
              max_iters: int = 16) -> Optional[np.ndarray]:
        seed = np.radians(np.asarray(seed_deg, dtype=float))
        dist = math.radians(max_seed_dist_deg)
        p = np.asarray(p_mm, dtype=float) / 1000.0
        if ori_slack_deg > 0:
            q, _ = self.ik_bounded_rad(p, R, seed, math.radians(ori_slack_deg), max_seed_dist_rad=dist)
        else:
            q = self.ik_rad(p, R, seed, max_iters=max_iters, max_seed_dist_rad=dist)
        return None if q is None else np.degrees(q)


def select_dh_variant(q_deg: Sequence[float], pose: Sequence[float]) -> Tuple[PiperKinematics, float]:
    """The DH variant whose FK best matches the arm's own pose feedback, and its position error
    (mm). The caller should not stream IK targets when the error is large (wrong model)."""
    best: Tuple[Optional[PiperKinematics], float] = (None, math.inf)
    for variant in _DH_VARIANTS:
        kin = PiperKinematics(variant)
        p, _ = kin.fk(q_deg)
        err = float(np.linalg.norm(p - np.asarray(pose[:3], dtype=float)))
        if err < best[1]:
            best = (kin, err)
    assert best[0] is not None
    return best[0], best[1]
