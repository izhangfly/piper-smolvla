"""Running a SmolVLA policy on the arm with LeRobot's asynchronous inference.

LeRobot's RobotClient runs on the robot host (the Surface) and the PolicyServer wherever the model fits (the
Mac, on MPS). The client streams observations (joints, gripper, camera frames, the task sentence) over gRPC and
executes the action chunks it gets back at the dataset's rate; while one chunk plays, the next is computed.
This module only wraps that client: it builds its config for robot type piper_http, checks everything before
the arm moves, bounds the run's duration, adds a stop key, and reports what happened.

Joints from overlapping chunks are blended with LeRobot's weighted average; the gripper takes the newest chunk
only (registered here as the aggregate "piper_gripper_latest"). That is the ACT runner's rule: the mean of open
and closed is half-open, which is exactly what dropped the cube.

The structure of run_client() follows LeRobot's async_client() (lerobot/async_inference/robot_client.py,
Copyright 2025 The HuggingFace Inc. team, Apache-2.0).
"""
from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import threading
import time
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

from .clients import JOINTS, CameraClient, DriverClient, DriverError
from .safety import load_facts, preview_verdict

AGGREGATE_NAME = "piper_gripper_latest"
GRIPPER_INDEX = JOINTS.index("gripper")
SHARED = Path(os.environ.get("NERVUS_ACT_SHARED", "~/Shared/NervusACT")).expanduser()
LOG_DIR = Path(os.environ.get("PIPER_SMOLVLA_LOGS", "~/.cache/piper-smolvla/logs")).expanduser()
CAMERA_WAKE_S = 8.0   # an idle ACT engine camera takes a second or two to open again


def piper_aggregate(old, new):
    out = 0.3 * old + 0.7 * new
    out[..., GRIPPER_INDEX] = new[..., GRIPPER_INDEX]
    return out


def register_aggregate():
    from lerobot.async_inference.configs import AGGREGATE_FUNCTIONS
    AGGREGATE_FUNCTIONS.setdefault(AGGREGATE_NAME, piper_aggregate)


@contextlib.contextmanager
def in_dir(path: Path):
    """LeRobot's async classes open logs/<name>_<time>.log relative to the working directory when they are
    imported; import them from a log directory instead of littering wherever the command was run."""
    path.mkdir(parents=True, exist_ok=True)
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def import_robot_client():
    with in_dir(LOG_DIR):
        from lerobot.async_inference.robot_client import RobotClient
    importlib.import_module("piper_smolvla.config")  # registers robot type piper_http
    register_aggregate()
    return RobotClient


# ------------------------------------------------------------------------------------------------ the policy
def resolve_policy(ref: str, shared: Path = SHARED) -> tuple[str, Path | None]:
    """(what to send to the policy server, a local copy of the policy if this machine has one).

    A bare name is looked up in <shared>/smolvla/policies here (for its safety facts) and sent as the name: the
    server resolves it in its own copy of the shared folder, so the two machines' home paths may differ."""
    p = Path(ref).expanduser()
    if (p / "config.json").is_file():
        return str(p.resolve()), p.resolve()
    local = Path(shared).expanduser() / "smolvla" / "policies" / ref
    if "/" not in ref and (local / "config.json").is_file():
        return ref, local
    return ref, None


def policy_inputs(policy_dir: Path | None) -> dict | None:
    """Cameras (role -> (H, W)), state and action sizes from a policy's config.json."""
    if policy_dir is None:
        return None
    try:
        cfg = json.loads((policy_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    cams = {k.removeprefix("observation.images."): tuple(v["shape"][1:]) for k, v in cfg.get("input_features", {}).items()
            if k.startswith("observation.images.")}
    state = (cfg.get("input_features", {}).get("observation.state") or {}).get("shape")
    action = (cfg.get("output_features", {}).get("action") or {}).get("shape")
    return {"type": cfg.get("type"), "cameras": cams, "state": state, "action": action,
            "chunk_size": cfg.get("chunk_size"), "n_action_steps": cfg.get("n_action_steps")}


# ------------------------------------------------------------------------------------------------ options
@dataclass
class RunOptions:
    server: str = "127.0.0.1:8080"
    policy: str = ""
    task: str = ""
    dry_run: bool = False
    driver_url: str = "http://127.0.0.1:8770"
    camera_url: str = "http://127.0.0.1:8795"
    cameras: list[str] | None = None        # default: the policy's, else wrist + scene
    image_width: int | None = None
    image_height: int | None = None
    fps: int | None = None                  # default: the policy's training data (15)
    policy_device: str = "mps"
    actions_per_chunk: int | None = None
    chunk_size_threshold: float = 0.5
    aggregate: str = AGGREGATE_NAME
    max_duration_s: float = 90.0
    dry_run_s: float = 10.0
    goto_start: bool = False
    facts: str | None = None
    speed_pct: int = 40
    gripper_effort: int = 500
    step_limit_x: float = 1.5
    lead_limit_x: float = 4.0
    envelope_margin: float = 5.0
    stagnation_s: float = 8.0
    shared: str = str(SHARED)
    extra: dict = field(default_factory=dict)


def plan_run(o: RunOptions) -> dict:
    """Everything the run needs, resolved, with no network access: the policy reference, cameras, rate."""
    if not o.task.strip():
        raise ValueError("--task is required: SmolVLA follows the sentence the demonstrations were recorded with")
    if not o.policy:
        raise ValueError("--policy is required (a name in <shared>/smolvla/policies, a path, or a Hub id)")
    if ":" not in o.server:
        raise ValueError("--server is host:port")
    send_ref, local = resolve_policy(o.policy, Path(o.shared))
    facts_src = o.facts or (str(local) if local else None)
    facts = load_facts(facts_src) if facts_src else None
    pin = policy_inputs(local)
    cams = o.cameras or (sorted(pin["cameras"]) if pin and pin["cameras"] else None) or \
        (sorted(facts["cameras"]) if facts and facts.get("cameras") else ["wrist", "scene"])
    h, w = 480, 640
    if pin and pin["cameras"]:
        h, w = next(iter(pin["cameras"].values()))
    h, w = o.image_height or h, o.image_width or w
    fps = o.fps or int((facts or {}).get("fps") or 15)
    chunk = o.actions_per_chunk or int((pin or {}).get("chunk_size") or 50)
    return {"send_ref": send_ref, "local_policy": local, "facts_path": facts_src if facts else None, "facts": facts,
            "inputs": pin, "cameras": cams, "image_size": (h, w), "fps": fps, "actions_per_chunk": chunk}


def client_config(o: RunOptions, plan: dict):
    """LeRobot's RobotClientConfig for this run (robot type piper_http)."""
    import_robot_client()
    from lerobot.async_inference.configs import RobotClientConfig

    from .config import PiperHttpConfig
    h, w = plan["image_size"]
    robot = PiperHttpConfig(driver_url=o.driver_url, camera_url=o.camera_url, camera_roles=list(plan["cameras"]),
                            image_width=w, image_height=h, dry_run=o.dry_run, facts_path=plan["facts_path"],
                            speed_pct=o.speed_pct, gripper_effort=o.gripper_effort, step_limit_x=o.step_limit_x,
                            lead_limit_x=o.lead_limit_x, envelope_margin=o.envelope_margin,
                            stagnation_s=o.stagnation_s, **o.extra)
    return RobotClientConfig(policy_type="smolvla", pretrained_name_or_path=plan["send_ref"], robot=robot,
                             actions_per_chunk=plan["actions_per_chunk"], task=o.task.strip(), server_address=o.server,
                             policy_device=o.policy_device, chunk_size_threshold=o.chunk_size_threshold,
                             fps=plan["fps"], aggregate_fn_name=o.aggregate)


# ------------------------------------------------------------------------------------------------ preflight
def preflight(o: RunOptions, plan: dict, check_server: bool = True) -> list[tuple[str, str]]:
    """[(level, message)]: level is ok, warn or error. A dry run tolerates what only matters for moving the
    arm (motors off, drag-teach, someone else holding it); a live run does not."""
    out: list[tuple[str, str]] = []
    arm_level = "warn" if o.dry_run else "error"
    drv = DriverClient(o.driver_url, client="smolvla")
    try:
        s = drv.get("/api/state?lite=1", timeout=2.0)
        drv.latest, drv.latest_at = s, time.monotonic()
        info = drv.info()
        state, _age = drv.read()
        out.append(("ok" if state else "error", f"driver {o.driver_url}: feedback {'fresh' if state else 'STALE'}"
                    + (f", joints {[round(v, 1) for v in info['joints']]}, gripper {info['gripper_mm']} mm" if state else "")
                    + (" (simulated)" if info.get("sim") else "")))
        if info.get("estopped"):
            out.append((arm_level, "the arm is emergency-stopped: Recover it first"))
        elif info.get("teaching"):
            out.append((arm_level, "the arm is in drag-teach: press Enable to take it out"))
        elif not info.get("enabled"):
            out.append((arm_level, "the motors are not enabled (this client never enables them)"))
        owner = (info.get("owner") or {}).get("client")
        if owner and owner != "smolvla":
            out.append((arm_level, f"'{owner}' is controlling the arm"))
        fault = drv.hard_fault()
        if fault:
            out.append((arm_level, fault))
        facts = plan["facts"]
        if facts and facts.get("start_pose") and state:
            far = max(abs(state[j] - facts["start_pose"][j]) for j in JOINTS[:6] if j in facts["start_pose"])
            out.append(("ok" if far < 15 else "warn", f"{far:.0f} deg from the demonstrations' median start pose"
                        + ("" if far < 15 else ": --goto-start moves there slowly first")))
    except Exception as e:
        out.append(("error", f"driver {o.driver_url}: {e}"))
    h, w = plan["image_size"]
    for role in plan["cameras"]:
        cam = CameraClient(o.camera_url, role, w, h)
        # The ACT engine releases idle cameras and opens them again on the first request, which answers 404
        # until the device delivers (a second or two): keep asking for a while.
        t_end, err = time.monotonic() + CAMERA_WAKE_S, None
        while True:
            try:
                img, _ = cam.fetch(timeout=3.0)
                ok = tuple(img.shape) == (h, w, 3)
                out.append(("ok" if ok else "error", f"camera {role}: {img.shape[1]}x{img.shape[0]}"
                            + ("" if ok else f", the policy expects {w}x{h}")))
                break
            except urllib.error.HTTPError as e:   # the engine answered: the camera may still be waking
                err = e
            except Exception as e:                # no engine at all: say so now
                out.append(("error", f"camera {role} at {cam.url}: {e}"))
                break
            if time.monotonic() >= t_end:
                out.append(("error", f"camera {role} at {cam.url}: {err} (for {CAMERA_WAKE_S:g} s)"))
                break
            time.sleep(0.5)
    pin = plan["inputs"]
    if pin:
        if pin["type"] != "smolvla":
            out.append(("error", f"the policy is {pin['type']!r}, not smolvla"))
        if pin["cameras"] and sorted(pin["cameras"]) != sorted(plan["cameras"]):
            out.append(("error", f"the policy was trained on cameras {sorted(pin['cameras'])}, this run sends "
                        f"{sorted(plan['cameras'])}"))
        for k in ("state", "action"):
            if pin[k] and pin[k] != [len(JOINTS)]:
                out.append(("error", f"the policy's {k} has shape {pin[k]}, the PiPER sends {len(JOINTS)} values"))
        if not any(m.startswith("the policy") for lvl, m in out if lvl == "error"):
            out.append(("ok", f"policy {plan['local_policy']}: cameras {sorted(pin['cameras'])}, chunk {pin['chunk_size']}"))
    else:
        out.append(("warn", f"no local copy of the policy {o.policy!r}: its cameras and sizes can't be checked here"))
    out.append(("ok" if plan["facts"] else "warn",
                "safety limits from the training data (nervus_policy.json)" if plan["facts"] else
                "no nervus_policy.json: safety limits fall back to the driver's joint limits and 4 deg/step"))
    if check_server:
        out.append(server_check(o.server))
    return out


def server_check(address: str, timeout: float = 5.0) -> tuple[str, str]:
    try:
        import grpc
        ch = grpc.insecure_channel(address)
        grpc.channel_ready_future(ch).result(timeout=timeout)
        ch.close()
        return ("ok", f"policy server {address} is reachable")
    except Exception:
        return ("error", f"policy server {address} is not reachable: start `piper-smolvla serve` on the Mac "
                "(with --tunnel to this host, or --host on a trusted LAN)")


# ------------------------------------------------------------------------------------------------ moving
# Adapted from izhangfly/NervusOS-Cambridge/act/engine/nervus_act/runner.py (Runner._goto).
def goto_start(o: RunOptions, start_pose: dict, fps: int = 15, steps: int = 60, speed: int = 20):
    """The ACT runner's start-pose move: 4 s of cosine-eased joint streaming at 20 % speed, then 1 s to settle."""
    drv = DriverClient(o.driver_url, client="smolvla").start()
    try:
        if not drv.wait_first(3.0):
            raise DriverError(drv.error or "driver unreachable")
        time.sleep(0.1)
        cur, age = drv.read()
        if cur is None or age > 0.3:
            raise DriverError("no live arm feedback")
        drv.speed(speed)
        pose = {j: float(start_pose.get(j, cur[j])) for j in JOINTS}
        for i in range(1, steps + 1):
            wgt = 0.5 - 0.5 * math.cos(math.pi * i / steps)
            drv.stream({j: cur[j] + (pose[j] - cur[j]) * wgt for j in JOINTS}, o.gripper_effort)
            time.sleep(1 / fps)
        time.sleep(1.0)
    finally:
        drv.close()


# Adapted from huggingface/lerobot/src/lerobot/async_inference/robot_client.py (async_client).
def run_client(cfg, max_duration_s: float, on_started=None) -> dict:
    """LeRobot's async_client(), bounded in time and with a stop key. Every way out (time limit, s, Ctrl+C,
    a safety stop raised by the robot, a lost server) ends in RobotClient.stop(), which disconnects the robot:
    it holds the measured pose with the motors on and releases control."""
    from .keyboard import KeyboardStopWatcher
    RobotClient = import_robot_client()
    reason = {"why": None}
    client = RobotClient(cfg)  # connects the robot: raises if it can't run
    robot = client.robot
    if not client.start():
        client.stop()
        raise RuntimeError(f"could not reach the policy server at {cfg.server_address}")

    def end(why):
        reason["why"] = reason["why"] or why
        client.shutdown_event.set()

    receiver = threading.Thread(target=client.receive_actions, daemon=True, name="receive-actions")
    receiver.start()
    timer = threading.Timer(max_duration_s, end, args=(f"reached the {max_duration_s:g} s limit",))
    timer.daemon = True
    timer.start()
    keys = KeyboardStopWatcher(lambda: end("stopped with the s key"))
    keys.start()
    if on_started:
        on_started()
    t0 = time.monotonic()
    try:
        client.control_loop(task=cfg.task)
    except KeyboardInterrupt:
        end("stopped with Ctrl+C")
    except Exception as e:
        end(f"{type(e).__name__}: {e}")
    finally:
        timer.cancel()
        keys.stop()
        client.stop()
        receiver.join(timeout=5)
    rep = robot.report()
    rep["stop_reason"] = robot.stop_reason or reason["why"] or "stopped"
    rep["elapsed_s"] = round(time.monotonic() - t0, 1)
    rep["chunk_size_seen"] = client.action_chunk_size
    rep["verdict"] = preview_verdict(rep["max_plan_deg"]) if rep["actions"] else \
        "no actions arrived from the policy server (see its log; are the cameras the policy's?)"
    return rep
