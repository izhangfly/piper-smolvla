# PiPER arm driver (hardware layer)

The AgileX PiPER (6-DOF arm plus parallel gripper) talks CAN 2.0B at 1 Mbit/s.
On this Ubuntu host it is plugged in through the candleLight USB-CAN adapter that
ships with the arm (USB `1d50:606f`, kernel driver `gs_usb`), which shows up as
`can0`.

NervusOS sits above the hardware, so the arm is driven by a small driver at the
bottom of the stack. Apps never touch CAN themselves:

```
Robot Arm app (NervusOS, sandboxed)
        │  HTTP on 127.0.0.1:8770
        ▼
piper_webctl.py   (nervus-piper.service, user nervus-piper)
        │  piper_sdk / SocketCAN
        ▼
can0 → PiPER
```

## Pieces

| File | Installed as | Job |
|---|---|---|
| `piper_webctl.py` | `/opt/nervus/piper/piper_webctl.py` | Driver and HTTP API; also serves a browser control panel |
| `kinematics.py` | `/opt/nervus/piper/kinematics.py` | Forward kinematics, DLS and closed-form IK (smooth mode) |
| `motion.py` | `/opt/nervus/piper/motion.py` | Planner, Ruckig trajectory follower, Cartesian jog (smooth mode) |
| `sim_smooth.py` | (not installed) | Prints or plots the planned motion for test moves, no arm needed |
| `requirements.txt` | `/opt/nervus/piper/venv` | `piper_sdk`, `python-can`, `numpy`, `ruckig` |
| `../hosted/libexec/can-up` | `/usr/libexec/nervus/can-up` | Sets `can0` to 1 Mbit/s and brings it up |
| `../hosted/systemd/nervus-can.service` | enabled at boot | Runs `can-up can0` |
| `../hosted/udev/90-nervus-can.rules` | `/etc/udev/rules.d/` | Runs `can-up` again when the adapter is replugged |
| `../hosted/systemd/nervus-piper.service` | started with NervusOS | Runs the driver; `PartOf=nervud.service`, so quitting NervusOS stops it |

Only one program can own `can0` at a time. That's the reason the driver belongs
to the platform rather than to an app: install or remove apps freely and the bus
still has a single owner.

## Safety

- The motors start **disabled**. Nothing moves until Enable is pressed.
- The client sends a heartbeat. If it stops for more than 1.5 s, the driver stops
  streaming commands and the arm halts.
- E-Stop sends `EmergencyStop` (CAN 0x150). Recover sends `ResetPiper`, which
  clears a latched collision fault without a power cycle (the arm goes limp for a
  moment, so support it).
- Joint targets are clamped to the limits in the AgileX manual, on the server.
- Stops are controlled: in smooth mode a lapsed deadman, stale feedback, Stop or a
  guard trip brakes the trajectory to a standstill at the brake limits (a fraction
  of a second, see below), then streaming stops. E-Stop stops streaming at once.
- Tracking guard (smooth mode): if a joint is more than 20 deg (`--track-err-deg`)
  from its command (the arm is blocked, faulted, or not taking commands), the driver
  holds the measured joints and latches like the IK guard.
- Kinematics check (smooth mode): our FK of the measured joints must land within
  15 mm of the arm's own reported pose (`fk_error_mm` in `/api/state`), else servo
  goals and jogs are refused with a hint to use `--cartesian firmware`. The DH
  variant (firmware older or newer than V1.5-2) is picked from the feedback.
- IK flip guard: in the Cartesian modes (panel jog and `/api/servo`) the driver
  also watches the joints. If one swings more than 45 deg in 0.5 s while the TCP
  motion doesn't explain it (the arm's own IK flipping the wrist near J5 = 0), it
  holds the arm with the motors on, drops the controller and refuses servo goals
  and jogs for 2 s. `/api/state` reports the last trip under `guard`; the panel
  shows it. `--ik-guard-deg 0` turns it off.
- The API has no authentication, so it listens on loopback only. Don't change
  `--bind` to `0.0.0.0`.

## Cartesian control: smooth (default) and firmware

`--cartesian smooth` (the default) does the IK in the driver and streams joints;
`--cartesian firmware` is the previous behaviour, kept as a fallback: `EndPoseCtrl`
(firmware IK, MOVE P) at 20 Hz to a target leashed 6 mm / 5 deg ahead of the
measured pose. That was choppy (every MOVE P setpoint restarts the firmware's
ramp), and the firmware can only head straight for the goal, so it stalled at
joint limits, near the singular region close to the base, or at the workspace edge
instead of going round. If numpy or ruckig is missing the driver starts in firmware
mode and says so in `/api/state` (`cartesian_note`). `/api/state` reports
`cartesian`, `control_hz`, the last `plan`, the `trajectory` state, `fk_error_mm`,
`dh_variant` and `jog_blocked`; `/api/servo` replies also carry `plan` and `reason`.
The API is otherwise unchanged, so the panel, Hand Teleop, LLM Arm and piperd all
get the new behaviour.

**Execution.** Joints go out as `JointCtrl` in MOVE J at 100 Hz (three CAN frames a
tick; the vendor demos stream at 200 Hz, piper_ros at 200, bimanual-vla at 100,
PiperXR at 50), with the firmware speed setting as a backstop above the
trajectory's own limits. The trajectory comes from Ruckig (jerk-limited, time-
optimal online trajectory generation), which keeps position, velocity and
acceleration across every change of target:

- *Streamed goals* (Hand Teleop at ~20 Hz, LLM Arm steps at 10 Hz, sliders): a plan
  shorter than 5 deg of joint travel is a target Ruckig heads for directly; each new
  goal only moves the target, so the arm doesn't stop between goals.
- *Longer plans* (a far goal from piperd's MoveToPose, HOME, the jog): a one-DoF
  jerk-limited profile along the path gives a reference point each tick, and Ruckig
  (velocity interface, reference velocity plus 8/s times the position error) tracks
  it. Straight lines are followed to within about 1 mm at every speed in
  simulation, and the command stays within the joint limits below whatever the
  reference does.

**Planning** (`motion.Planner.plan_pose`, for each new `/api/servo` goal; the same
goal repeated, as piperd and LLM Arm do at 10 Hz, keeps the running plan):

1. *Straight line* if it is feasible: IK every 5 mm / 3 deg along it, each sample
   seeded at the previous one; every solution inside the joint limits (2 deg
   margin), no joint moving more than 8 deg between samples (an IK flip), the
   manipulability above 0.02 (about J5 = 3 deg, the elbow straight, or the wrist
   centre on the J1 axis), and the elbow and wrist centre 30 mm above the table.
2. Otherwise *joint space* to the goal's IK solution nearest the current joints
   (closed-form IK gives every branch; the cost is the time to get there plus
   penalties near limits and singularities, so it never picks the wrist flip). A
   joint-space move turns, backs up and extends by itself. Its TCP is checked by
   sampling FK every 1 deg: the flange must stay inside the workspace box and the
   elbow and wrist above the table.
3. If that check fails, *via points*, quickest first: raise at the start (lift,
   then across), raise at both ends, retract to a short reach, turn and extend, and
   the READY tuck at the start and goal azimuths. Each candidate path is checked the
   same way; the arm passes via points at a crawl, so corners are not cut.
4. If the goal itself is out of reach, the arm goes as far along the straight line
   as is feasible (`partial`), or holds (`unreachable`, `blocked`).

HOME uses the same joint-space check and via points (falling back to the plain
joint move); sliders and wrist keys go through the trajectory directly. The panel's
Cartesian jog advances a carrot along the held axes, solves IK each tick and
follows the trail of solutions, so the TCP stays on the jog line; releasing the key
brakes. ACT streaming (`/api/stream`) is unchanged: the learned policies keep the
dynamics they were trained with.

**Limits** (`motion.py`), at speed 100 %: joint velocity 90 deg/s (J1-J3) and 120
deg/s (J4-J6), acceleration 200 / 300 deg/s^2, jerk 1000 / 1500 deg/s^3. Velocity
scales with the speed setting; acceleration and jerk with max(speed, 30 %). The
flange is capped at 200 mm/s and 120 deg/s on lines and jogs (1.5 times that on
joint-space moves), scaled the same way: at the default speed 15 that is 30 mm/s,
close to the 27 mm/s the firmware jog gave. Stops brake at the 100 % acceleration
and twice the jerk whatever the speed: from 15 % that is a few hundredths of a
second, from full speed about 0.6 s. The firmware's own MOVE J cap (speed % of its
rated 172 deg/s) stays above these.

**Flags**: `--cartesian smooth|firmware`, `--track-err-deg 20`, `--workspace
X0,X1,Y0,Y1,Z0,Z1` (mm; the box servo goals are clamped to and smooth-mode paths
keep the TCP in, default `60,480,-320,320,80,600`; a smaller box models limited
room), and the existing `--speed`, `--ik-guard-deg`, `--bind`, `--port`, `--iface`.

**Simulation**: `python3 sim_smooth.py` runs the planner and trajectory on the hard
case (left to right with only 250 mm of room in front: it backs up, turns and
extends) and prints the joint and TCP trace; `--case low` (a low swing that would
dip below the box floor: it lifts first), `--case line`, `--speed`, `--box-x-max`,
`--plot trace.png` (matplotlib), `--csv trace.csv`.

**Reused code**: the kinematics are adapted from showlab/Show-Harness
`core/piper/kinematics.py` (the vendor's DH model from piper_sdk's
`C_PiperForwardKinematics`, DLS IK with a near-seed check and `ik_bounded`); the
closed-form IK's structure from agilexrobotics/Agilex-College
`piper/piper_kinematics/include/piper_analytical_ik.hpp`; the Ruckig handling from
SUNNYsyy2005/bimanual-vla `bimanual_vla/deployment/servo.py`.

**First run on the arm** (not done yet: nothing here has moved the real arm). Install
the venv (`pip install -r requirements.txt`) and the two new modules, restart
`nervus-piper`, then check: `/api/state` shows `cartesian: smooth` and an
`fk_error_mm` of a millimetre or less; at speed 15, HOME and a slider move look as
before; a held jog key moves along one axis and stops when released; a servo goal
50 mm away goes in a straight line; the left-to-right move from the simulation goes
round instead of stalling; Stop, E-Stop and pulling the heartbeat stop the arm; and
`journalctl -u nervus-piper` shows no `(tracking guard)` trips in normal moves (if
it does, the firmware is not keeping up with the trajectory: lower the limits).

## API

`GET /api/state`, and `POST` to `/api/heartbeat`, `/api/enable {on}`,
`/api/joint {joint, deg}`, `/api/gripper {mm}`, `/api/speed {speed}`,
`/api/mode {mode: joint|cartesian}`, `/api/jog {axis, dir, on}` (leashed Cartesian
jog), `/api/jointjog {joint, dir, on}`, `/api/jogstop`, `/api/home`,
`/api/recover`, `/api/estop`, `/api/stop`, `/api/release`, `/api/servo {pose, speed?}`
(piperd, Hand Teleop, LLM Arm; smooth mode replies with the `plan` kind and its
`reason`), `/api/stream {joints, gripper_mm?}` (ACT engine). Moving
calls take a `client` name; one client controls the arm at a time (HTTP 409 for
the others), and stop and E-stop are accepted from anyone.

Tests (no arm needed; a fake `piper_sdk` stands in, whose joints follow the
`JointCtrl` stream): `python3 -m unittest discover -s hardware/piper -p "test_*.py"`.
`test_kinematics.py` checks FK against piper_sdk and the arm's own numbers and the
IK; `test_motion.py` the trajectory's limits and continuity and the planner's
choices; `test_piper_webctl.py` the driver end to end in both modes. The smooth-mode
tests need numpy and ruckig and are skipped without them.

## Useful commands

```sh
ip -details link show can0          # state, bitrate
candump can0                        # raw traffic (arm powered: ~hundreds of frames/s)
journalctl -u nervus-piper -f       # driver log
curl -s localhost:8770/api/state    # what the driver sees
ssh -L 8770:localhost:8770 <host>   # browser panel from another machine
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `can0` missing | Adapter unplugged, or `gs_usb` not loaded (`sudo modprobe gs_usb`) |
| `can0` is `DOWN` after replugging | The udev rule didn't run: `sudo /usr/libexec/nervus/can-up can0` |
| Driver log: `CAN thread unhealthy on can0` | Arm not powered, or wrong bitrate. `candump can0` should show frames |
| State reads but nothing moves | Press Enable first; check E-Stop isn't latched; Recover clears a collision fault |
| `TARGET_POS_EXCEEDS_LIMIT` | Pose limit. The Cartesian jog is leashed to the actual pose, so this means a real workspace edge |
| `cartesian: firmware` with a `cartesian_note` | numpy or ruckig missing in the driver venv: `pip install -r requirements.txt` |
| Servo goals refused: "kinematic model doesn't match" | `fk_error_mm` over 15 mm: run with `--cartesian firmware` and report the joints and pose |
| `(tracking guard)` in the log | A joint lagged its command by 20 deg: something blocked the arm, or the firmware isn't following |
| Jog stops with `jog_blocked` | The way ahead needs a joint past its limit, a singularity, or leaves the box; jog another axis |
