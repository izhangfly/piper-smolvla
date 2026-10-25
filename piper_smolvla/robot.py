"""A LeRobot Robot for the AgileX PiPER, through the NervusOS PiPER driver and the ACT engine's cameras.

It never touches CAN or a camera device: the driver owns the bus and the ACT engine owns the cameras, so this
class is one more HTTP client of each, claiming the arm as client "smolvla". That keeps it from fighting the
Robot Arm app, Hand Teleop or the ACT engine (the driver lets one client control the arm at a time).

Features match the datasets the ACT engine records (LeRobot v3, CONTRACT.json): observation.state and action
are [joint1..joint6 in degrees, gripper in mm], and each camera role is an RGB image. In LeRobot's flat robot
dictionaries that is {"joint1": float, ..., "gripper": float, "wrist": (H, W, 3), "scene": (H, W, 3)}.

Plugin structure after lerobot_robot_piper (AgRoboticsResearch, Apache-2.0); the safety rules and the hold on
stop are the ACT engine's (runner.py), see safety.py.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from functools import cached_property
from typing import Any

from lerobot.robots.robot import Robot

from .clients import JOINTS, Busy, CameraClient, DriverClient, DriverError, mono
from .config import PiperHttpConfig
from .safety import SafetyFilter, SafetyParams, UnsafeAction, complete_facts, load_facts

log = logging.getLogger(__name__)


# Adapted from AgRoboticsResearch/lerobot_robot_piper/lerobot_robot_piper/piper.py (Robot subclass layout).
class PiperHttp(Robot):
    config_class = PiperHttpConfig
    name = "piper_http"

    def __init__(self, config: PiperHttpConfig):
        super().__init__(config)
        self.config = config
        self.driver: DriverClient | None = None
        self.cameras: dict[str, CameraClient] = {}
        self.safety: SafetyFilter | None = None
        self.facts: dict | None = None
        self._connected = False
        self._sent = 0                  # commands streamed to the driver in this connection
        self._handed_over = False       # the operator took the arm in drag-teach: send nothing more
        self._stale_since: float | None = None
        self._still_since: float | None = None
        self._last_state: dict | None = None
        self.stop_reason: str | None = None
        self.stats = {"actions": 0, "sent": 0, "stale_holds": 0, "max_plan_deg": 0.0}
        self.recent: deque = deque(maxlen=200)   # (time, raw action, command, gripper mode) for reports

    # ---- features ----
    @cached_property
    def observation_features(self) -> dict:
        ft: dict[str, Any] = {j: float for j in JOINTS}
        for role in self.config.camera_roles:
            ft[role] = (self.config.image_height, self.config.image_width, 3)
        return ft

    @cached_property
    def action_features(self) -> dict:
        return {j: float for j in JOINTS}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True  # the arm's encoders are absolute; there is nothing to calibrate

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ---- lifecycle ----
    def connect(self, calibrate: bool = True) -> None:
        cfg = self.config
        if self._connected:
            return
        self.driver = DriverClient(cfg.driver_url, client=cfg.client).start()
        try:
            if not self.driver.wait_first(cfg.connect_timeout_s):
                raise ConnectionError(self.driver.error or f"no answer from the PiPER driver at {cfg.driver_url}")
            state, age = self._wait_fresh(cfg.connect_timeout_s)
            if state is None:
                raise ConnectionError("the arm's feedback is stale. If it is idle (in standby it stops reporting), "
                                      "press its drag-teach button once so the light is on, then Enable")
            info = self.driver.info()
            if not cfg.dry_run:
                problem = self._live_problem(info)
                if problem:
                    raise ConnectionError(problem)
            for role in cfg.camera_roles:
                cam = CameraClient(cfg.camera_url, role, cfg.image_width, cfg.image_height, cfg.camera_poll_hz).start()
                self.cameras[role] = cam
            for role, cam in self.cameras.items():
                if not cam.wait_first(cfg.connect_timeout_s):
                    raise ConnectionError(cam.error or f"no frame from the {role} camera at {cam.url}")
                want = (cfg.image_height, cfg.image_width, 3)
                if cam.shape != want:
                    raise ConnectionError(f"the {role} camera delivers {cam.shape}, the policy expects {want}: "
                                          "set --image-width/--image-height or fix the camera")
            self.facts = complete_facts(load_facts(cfg.facts_path), self.driver.joint_limits())
            if not self.facts["from_training_data"]:
                log.warning("no nervus_policy.json facts: using the driver's joint limits as the envelope and "
                            "4 deg/step as every joint's demonstrated step size")
            self.safety = SafetyFilter(self.facts, SafetyParams(
                step_limit_x=cfg.step_limit_x, lead_limit_x=cfg.lead_limit_x, envelope_margin=cfg.envelope_margin,
                tracking_stop_deg=cfg.tracking_stop_deg, gripper_squeeze_mm=cfg.gripper_squeeze_mm,
                gripper_release_steps=cfg.gripper_release_steps))
            if not cfg.dry_run and cfg.speed_pct:
                self.driver.speed(cfg.speed_pct)
        except Exception:
            self._close_clients()
            raise
        self._connected = True
        self._sent, self._handed_over, self._stale_since, self.stop_reason = 0, False, None, None
        self._still_since, self._last_state = None, None
        log.info("PiPER connected through %s (%s), cameras %s from %s", cfg.driver_url,
                 "DRY RUN: nothing will be sent to the arm" if cfg.dry_run else "live",
                 list(self.cameras), cfg.camera_url)

    def _wait_fresh(self, timeout: float):
        t_end = mono() + timeout
        while True:
            state, age = self.driver.read()
            if state is not None and age * 1000 <= self.config.stale_arm_ms:
                return state, age
            if mono() >= t_end:
                return None, math.inf
            time.sleep(0.05)

    # Adapted from izhangfly/NervusOS-Cambridge/act/engine/nervus_act/runner.py (Runner.start refusals).
    def _live_problem(self, info: dict) -> str | None:
        """Why a live run can't start, in the words of the ACT runner; None if it can."""
        if info.get("estopped"):
            return "the arm is emergency-stopped: Recover it (Robot Arm app or ACT Studio) before running"
        if info.get("teaching"):
            return "the arm is in drag-teach: press Enable (Robot Arm app or ACT Studio) to take it out"
        if not info.get("enabled"):
            return "enable the motors first (Robot Arm app or ACT Studio): this client never enables them itself"
        fault = self.driver.hard_fault()
        if fault:
            return f"{fault}: clear it (Recover) before running"
        owner = self.driver.owner()
        if owner and owner != self.config.client:
            return f"another app is controlling the arm ('{owner}'): close it or press Stop there"
        return None

    # Adapted from izhangfly/NervusOS-Cambridge/act/engine/nervus_act/runner.py (end of Runner._loop: hold, then release).
    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            if not self.config.dry_run and self._sent and not self._handed_over and self.driver is not None:
                state, age = self.driver.read()
                if state is not None and age < 0.5:
                    # Freeze where the arm IS, with a light grip from the measured width; torque stays on.
                    try:
                        self.driver.stream(self.safety.hold_target(state), self.config.gripper_effort)
                    except DriverError as e:
                        log.warning("could not send the hold command: %s", e)
                self.driver.release()
            # After a hand-over the operator holds the arm in drag-teach: send nothing, or a mode command would
            # pull it out of drag-teach and stiffen it in their hands. The driver's deadman lets go of it.
        finally:
            self._close_clients()
            self._connected = False

    def _close_clients(self):
        for cam in self.cameras.values():
            cam.close()
        self.cameras = {}
        if self.driver is not None:
            self.driver.close()

    # ---- observation ----
    def _camera_stale(self) -> str | None:
        for role, cam in self.cameras.items():
            frame, age = cam.latest()
            if frame is None or age * 1000 > self.config.stale_camera_ms:
                return f"{role} camera frames stale" + (f" ({cam.error})" if cam.error else "")
        return None

    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected")
        state, age = self.driver.read()
        if state is None or age * 1000 > self.config.stale_arm_ms:
            raise RuntimeError("arm feedback is stale: not sending an observation from a pose we can't see")
        obs: dict[str, Any] = dict(state)
        for role, cam in self.cameras.items():
            frame, cam_age = cam.latest()
            if frame is None or cam_age * 1000 > self.config.stale_camera_ms:
                raise RuntimeError(f"the {role} camera's frames are stale")
            obs[role] = frame
        return obs

    # ---- action ----
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected")
        cfg = self.config
        try:
            return self._send(action, cfg)
        except Exception as e:
            self.stop_reason = self.stop_reason or str(e)
            raise

    def _send(self, action: dict[str, Any], cfg: PiperHttpConfig) -> dict[str, Any]:
        now = mono()
        self.stats["actions"] += 1
        state, age = self.driver.read()
        stale = None
        if state is None or age * 1000 > cfg.stale_arm_ms:
            stale = "arm feedback stale"
        else:
            stale = self._camera_stale()
        if stale:
            # Hold the last command (never steer from a stale read); a stale stretch past the grace ends the run.
            self._stale_since = self._stale_since or now
            self.stats["stale_holds"] += 1
            if now - self._stale_since > cfg.stale_grace_s:
                raise UnsafeAction(f"{stale} for {now - self._stale_since:.1f} s")
            return dict(self.safety.last or {})
        self._stale_since = None
        if self.driver.teaching():
            self._handed_over = True
            raise UnsafeAction("the arm entered drag-teach: the policy lets go of it")
        fault = self.driver.hard_fault()
        if fault:
            raise UnsafeAction(fault)
        if not cfg.dry_run:
            self._check_stagnation(state, now)
        raw = {j: float(action[j]) for j in JOINTS}
        cmd, gmode = self.safety(raw, state)
        planned = max(abs(raw[j] - state[j]) for j in JOINTS[:6])
        self.stats["max_plan_deg"] = max(self.stats["max_plan_deg"], planned)
        self.recent.append((time.time(), raw, cmd, gmode))
        if cfg.dry_run:
            self.safety.last = dict(state, gripper=cmd["gripper"])  # nothing moved: limit from where the arm is
            return cmd
        try:
            self.driver.stream(cmd, cfg.gripper_effort)
        except Busy as e:
            raise UnsafeAction(f"lost control of the arm: {e}")
        self._sent += 1
        self.stats["sent"] += 1
        return cmd

    # Adapted from zhitaoqiu/Piper-SmolVla/scripts/deploy_smolvla.py (stagnation detection): a run whose arm stays
    # still is a policy that has stopped acting (the ACT runner's "would likely sit still"), so end it rather
    # than wait for the time limit. Timed in seconds here, as the robot doesn't know the control rate.
    def _check_stagnation(self, state: dict, now: float):
        cfg = self.config
        last, self._last_state = self._last_state, state
        if cfg.stagnation_s <= 0 or last is None:
            return
        diff = max(abs(state[j] - last[j]) for j in JOINTS)
        if diff < cfg.stagnation_deg:
            self._still_since = self._still_since or now
            if now - self._still_since >= cfg.stagnation_s:
                raise UnsafeAction(f"the arm has not moved for {now - self._still_since:.0f} s: the policy is stuck "
                                   "(a scene unlike its demonstrations, or the task is done)")
        else:
            self._still_since = None

    # ---- reporting ----
    def report(self) -> dict:
        out = dict(self.stats)
        out["max_plan_deg"] = round(out["max_plan_deg"], 1)
        out["clamps"] = dict(self.safety.counts) if self.safety else {}
        out["stop_reason"] = self.stop_reason
        out["facts_from_training_data"] = bool(self.facts and self.facts.get("from_training_data"))
        out["dry_run"] = self.config.dry_run
        return out
