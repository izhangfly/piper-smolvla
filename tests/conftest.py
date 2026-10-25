"""Shared fixtures. Nothing here may reach real hardware: the simulated driver and cameras listen on random
loopback ports, and any HTTP request to the real driver's or ACT engine's port fails the test."""
from __future__ import annotations

import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

# Before LeRobot is imported: its log files and calibration folders go to a scratch directory, not ~/.cache.
_SCRATCH = tempfile.mkdtemp(prefix="piper-smolvla-tests-")
os.environ.setdefault("PIPER_SMOLVLA_LOGS", os.path.join(_SCRATCH, "logs"))
os.environ.setdefault("HF_LEROBOT_CALIBRATION", os.path.join(_SCRATCH, "calibration"))

REAL_PORTS = {8770, 8795}


@pytest.fixture(autouse=True)
def no_real_hardware(monkeypatch):
    real = urllib.request.urlopen

    def guarded(url, *a, **kw):
        u = url.full_url if isinstance(url, urllib.request.Request) else str(url)
        port = urllib.parse.urlparse(u).port
        if port in REAL_PORTS:
            raise AssertionError(f"a test tried to reach the real PiPER driver or ACT engine: {u}")
        return real(url, *a, **kw)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)
    yield


@pytest.fixture
def sim():
    from piper_smolvla.sim import SimCameraServer, SimDriver
    drv = SimDriver()
    cams = SimCameraServer(drv.arm, roles=("wrist", "scene"))
    yield drv, cams
    cams.close()
    drv.close()


def post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=2) as r:
        return json.loads(r.read())


@pytest.fixture
def make_robot(sim, tmp_path):
    from piper_smolvla.config import PiperHttpConfig
    from piper_smolvla.robot import PiperHttp
    drv, cams = sim
    made = []

    def make(**kw):
        cfg = PiperHttpConfig(driver_url=drv.url, camera_url=cams.url, calibration_dir=Path(tmp_path) / "cal", **kw)
        r = PiperHttp(cfg)
        made.append(r)
        return r

    yield make
    for r in made:
        try:
            r.disconnect()
        except Exception:
            pass


def fake_dataset(root: Path, cameras=("scene", "wrist"), episodes=2, frames=40, task="pick up the red cube",
                 fps=15, shape=(480, 640, 3)) -> Path:
    """The metadata of a recorder dataset (no video): enough for planning, checks and the CLI."""
    import pandas as pd
    root = Path(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    feats = {"observation.state": {"dtype": "float32", "shape": [7], "names": names},
             "action": {"dtype": "float32", "shape": [7], "names": names}}
    for c in cameras:
        feats[f"observation.images.{c}"] = {"dtype": "video", "shape": list(shape), "names": ["height", "width", "channels"]}
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "robot_type": "agilex_piper", "fps": fps, "total_episodes": episodes,
        "total_frames": episodes * frames, "features": feats}))
    if task is not None:
        pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    (root / "nervus_dataset.json").write_text(json.dumps({"name": root.name, "task": task}))
    return root
