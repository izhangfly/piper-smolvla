#!/usr/bin/env python3
"""Simulate the smooth Cartesian mode's planner and trajectory, with no arm: see the planned motion.

    python3 sim_smooth.py                       # the hard case: left to right, little room in front
    python3 sim_smooth.py --case low            # a low pick-and-place whose joint swing would dip
    python3 sim_smooth.py --case line           # an easy move: the straight line
    python3 sim_smooth.py --box-x-max 480       # the hard case with the default box
    python3 sim_smooth.py --plot trace.png      # joint and TCP traces (needs matplotlib)
    python3 sim_smooth.py --csv trace.csv       # every tick: t, J1..J6, x, y, z

It runs exactly what the driver runs (motion.Planner.plan_pose, then motion.Follower.step at
100 Hz) and prints the plan, a trace every 0.25 s, and a summary: duration, peak joint speeds
against their limits, and how close the TCP came to the box and the table.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import kinematics as K  # noqa: E402
import motion as M  # noqa: E402

BOX = {"x": (60.0, 480.0), "y": (-320.0, 320.0), "z": (80.0, 600.0)}
CASES = {
    # TCP on the left, tool pointing out to the left, to the mirror pose on the right, with only
    # 250 mm of room in front of the base. The straight line has no continuous IK (the wrist
    # would have to flip half way) and the plain joint-space swing reaches x = 337 mm, so the
    # planner pulls the TCP in, turns, and extends again.
    "limited": {"start": [150.0, 300.0, 200.0, 180.0, 65.0, -110.0],
                "goal": [150.0, -300.0, 200.0, 180.0, 65.0, 110.0], "box_x_max": 250.0},
    # Low on the left to low on the right: the joint-space swing would dip below the box floor
    # (z = 80 mm) half way, so the planner lifts first.
    "low": {"start": [321.0, 82.0, 108.0, -143.0, 71.0, -137.0],
            "goal": [112.0, -150.0, 88.0, 158.0, -8.0, 60.0], "box_x_max": 480.0},
    # An ordinary move from READY: the straight line.
    "line": {"start": [138.2, 0.0, 417.9, 180.0, 65.0, 180.0],
             "goal": [198.0, 40.0, 368.0, 180.0, 65.0, 180.0], "box_x_max": 480.0},
}


def start_joints(planner, pose):
    q = planner.solve(np.array(M.READY_POSE), *K.pose_to_rt(pose))
    if q is None:
        raise SystemExit(f"start pose {pose} is out of reach")
    return q


def simulate(planner, q0, goal, speed=50.0, hz=M.CONTROL_HZ, max_s=30.0):
    """Plan from q0 to goal and run the trajectory to the end. Returns (plan, t, q, tcp)."""
    fol = M.Follower(hz)
    fol.reset(q0)
    plan = planner.plan_pose(q0, goal)
    if plan.moves:
        plan.install(fol)
    ts, qs = [0.0], [np.asarray(q0, dtype=float)]
    t = 0.0
    while t < max_s:
        q = fol.step(speed)
        t += 1.0 / hz
        ts.append(t)
        qs.append(q)
        if fol.idle():
            break
    qs = np.array(qs)
    tcp = np.array([planner.kin.fk(q)[0] for q in qs])
    return plan, np.array(ts), qs, tcp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", choices=sorted(CASES), default="limited")
    ap.add_argument("--speed", type=float, default=50.0, help="the driver's speed setting, percent")
    ap.add_argument("--box-x-max", type=float, help="far edge of the workspace box in x (mm)")
    ap.add_argument("--plot", metavar="PNG")
    ap.add_argument("--csv", metavar="FILE")
    args = ap.parse_args()

    case = CASES[args.case]
    box = dict(BOX)
    box["x"] = (BOX["x"][0], args.box_x_max if args.box_x_max else case["box_x_max"])
    kin = K.PiperKinematics(0x01)
    ws = M.Workspace(box)
    planner = M.Planner(kin, ws)
    q0 = start_joints(planner, case["start"])
    plan, t, q, tcp = simulate(planner, q0, case["goal"], args.speed)

    print(f"case {args.case}: {case['start'][:3]} -> {case['goal'][:3]} mm, box x {box['x'][0]:.0f}..{box['x'][1]:.0f} "
          f"y {box['y'][0]:.0f}..{box['y'][1]:.0f} z {box['z'][0]:.0f}..{box['z'][1]:.0f}, speed {args.speed:.0f} %")
    print(f"plan: {plan.kind} ({plan.ms:.0f} ms to plan)")
    if plan.reason:
        print(f"  why: {plan.reason}")
    for v in plan.vias:
        print(f"  via: x {v[0]:.0f}  y {v[1]:.0f}  z {v[2]:.0f} mm")
    print()
    print("    t s     J1     J2     J3     J4     J5     J6  |      x      y      z mm")
    step = max(1, int(round(0.25 * M.CONTROL_HZ)))
    for i in list(range(0, len(t), step)) + [len(t) - 1]:
        j = " ".join(f"{v:6.1f}" for v in q[i])
        print(f"  {t[i]:5.2f} {j}  | {tcp[i][0]:6.1f} {tcp[i][1]:6.1f} {tcp[i][2]:6.1f}")
    vmax, _, _ = M.motion_limits(args.speed)
    v = np.abs(np.diff(q, axis=0)) * M.CONTROL_HZ
    pts = [kin.points(x) for x in q]
    out = max(ws.outside(p["flange"]) for p in pts)
    low = min(min(p["elbow"][2], p["wrist"][2]) for p in pts)
    print()
    print(f"duration {t[-1]:.2f} s, TCP path {np.sum(np.linalg.norm(np.diff(tcp, axis=0), axis=1)):.0f} mm, "
          f"peak TCP speed {np.max(np.linalg.norm(np.diff(tcp, axis=0), axis=1)) * M.CONTROL_HZ:.0f} mm/s")
    print("peak joint speed deg/s " + " ".join(f"J{n + 1} {v[:, n].max():.0f}/{vmax[n]:.0f}" for n in range(6)))
    print(f"TCP x {tcp[:, 0].min():.0f}..{tcp[:, 0].max():.0f}, y {tcp[:, 1].min():.0f}..{tcp[:, 1].max():.0f}, "
          f"z {tcp[:, 2].min():.0f}..{tcp[:, 2].max():.0f} mm; furthest outside the box {out:.1f} mm; "
          f"lowest elbow/wrist {low:.0f} mm")
    end = kin.fk_pose(q[-1])
    print(f"final TCP error {math.dist(end[:3], case['goal'][:3]):.2f} mm")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("t,j1,j2,j3,j4,j5,j6,x,y,z\n")
            for i in range(len(t)):
                f.write(",".join(f"{v:.4f}" for v in [t[i], *q[i], *tcp[i]]) + "\n")
        print(f"wrote {args.csv}")
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise SystemExit("--plot needs matplotlib (pip install matplotlib)")
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        for n in range(6):
            ax[0].plot(t, q[:, n], label=f"J{n + 1}")
        ax[0].set_xlabel("t (s)")
        ax[0].set_ylabel("joint (deg)")
        ax[0].legend(fontsize=8)
        ax[0].set_title(f"joints ({plan.kind})")
        x0, x1 = box["x"]
        y0, y1 = box["y"]
        ax[1].plot([y0, y1, y1, y0, y0], [x0, x0, x1, x1, x0], "k--", lw=1, label="box")
        ax[1].plot(tcp[:, 1], tcp[:, 0], lw=2, label="TCP")
        ax[1].plot(0, 0, "ko", label="base")
        ax[1].set_xlabel("y (mm)")
        ax[1].set_ylabel("x (mm)")
        ax[1].invert_xaxis()
        ax[1].set_aspect("equal")
        ax[1].legend(fontsize=8)
        ax[1].set_title("TCP from above")
        ax[2].plot(t, tcp[:, 2], label="TCP z")
        ax[2].axhline(box["z"][0], color="k", ls="--", lw=1, label="box floor")
        ax[2].set_xlabel("t (s)")
        ax[2].set_ylabel("z (mm)")
        ax[2].legend(fontsize=8)
        ax[2].set_title("TCP height")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
