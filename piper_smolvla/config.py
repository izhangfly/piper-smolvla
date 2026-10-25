"""LeRobot configuration for a PiPER driven through its HTTP driver (robot type "piper_http").

Plugin layout after lerobot_robot_piper (AgRoboticsResearch, Apache-2.0): a RobotConfig subclass registered
under a robot type, next to a Robot subclass of the same name minus "Config". LeRobot's command-line tools
then accept --robot.type=piper_http once this package is imported (piper-smolvla's wrappers do that).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig


# Adapted from AgRoboticsResearch/lerobot_robot_piper/lerobot_robot_piper/config_piper.py (plugin config).
@RobotConfig.register_subclass("piper_http")
@dataclass
class PiperHttpConfig(RobotConfig):
    # The PiPER driver (piper_webctl.py) and the ACT engine that serves the cameras. Both listen on loopback
    # only on the robot host: neither has authentication, and the driver moves the arm.
    driver_url: str = "http://127.0.0.1:8770"
    camera_url: str = "http://127.0.0.1:8795"
    # Camera roles as the ACT engine names them, and the frame size the policy was trained on.
    camera_roles: list[str] = field(default_factory=lambda: ["wrist", "scene"])
    image_width: int = 640
    image_height: int = 480
    # The name this client claims the arm under (the driver lets one client control it at a time).
    client: str = "smolvla"
    # Plan and log everything, send nothing that moves the arm.
    dry_run: bool = False
    # nervus_policy.json (or the policy directory holding it): the demonstrated envelope and step sizes.
    facts_path: str | None = None
    # Firmware MOVE J speed, percent of rated, set when a live run connects (0 = leave it as it is).
    speed_pct: int = 40
    # Safety (see safety.py; the ACT runner's defaults).
    step_limit_x: float = 1.5
    lead_limit_x: float = 4.0
    envelope_margin: float = 5.0
    tracking_stop_deg: float = 30.0
    gripper_effort: int = 500  # 0.001 N*m; 1000 crushed a 3D-printed cube
    gripper_squeeze_mm: float = 1.5
    gripper_release_steps: int = 8
    # Staleness: a stale reading holds the last command; one stale for longer than the grace ends the run.
    stale_arm_ms: int = 300
    stale_camera_ms: int = 500
    stale_grace_s: float = 1.0
    # Stagnation stop (live runs): the arm hasn't moved more than stagnation_deg per step for stagnation_s
    # seconds, so the policy is stuck (0 = off).
    stagnation_s: float = 8.0
    stagnation_deg: float = 0.05
    camera_poll_hz: float = 20.0
    connect_timeout_s: float = 8.0  # covers the ACT engine reopening an idle camera
