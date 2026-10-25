"""Building a training set from recorder-style LeRobot v3 datasets (tiny ones, with real video)."""
import json
from pathlib import Path

import numpy as np
import pytest

from piper_smolvla import data as D
from piper_smolvla.clients import JOINTS


def make_lerobot_dataset(root: Path, task: str, episodes: int = 2, frames: int = 40, size=(48, 64),
                         cameras=("scene", "wrist"), idle: int = 10) -> Path:
    """Like the ACT engine's recorder: 15 fps, state/action in deg and mm, action[t] = state[t+1], a still
    stretch at both ends of every episode (what the build trims)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    h, w = size
    feats = {"observation.state": {"dtype": "float32", "shape": (7,), "names": JOINTS},
             "action": {"dtype": "float32", "shape": (7,), "names": JOINTS}}
    for c in cameras:
        feats[f"observation.images.{c}"] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(f"local/{root.name}", fps=15, features=feats, root=root, robot_type="agilex_piper",
                               use_videos=True, image_writer_processes=0, image_writer_threads=2, vcodec="h264")
    rng = np.random.default_rng(0)
    for e in range(episodes):
        n = frames + 1
        t = np.clip(np.arange(n) - idle, 0, frames - 2 * idle) / (frames - 2 * idle)
        S = np.zeros((n, 7), np.float32)
        S[:, 1], S[:, 2], S[:, 4] = 45, -60, 45
        S[:, 0] = 30 * t + 5 * e
        S[:, 6] = 10 + 40 * t
        for i in range(frames):
            frame = {"observation.state": S[i], "action": S[i + 1], "task": task}
            for c in cameras:
                frame[f"observation.images.{c}"] = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            ds.add_frame(frame)
        ds.save_episode()
    ds.finalize()
    (root / "nervus_dataset.json").write_text(json.dumps({"name": root.name, "task": task}))
    return root


@pytest.mark.slow
def test_info_build_and_facts(tmp_path):
    a = make_lerobot_dataset(tmp_path / "datasets" / "cube_a", "pick up the red cube")
    b = make_lerobot_dataset(tmp_path / "datasets" / "cube_b", "put the cube in the bowl", episodes=1)
    (b / "nervus_edits.json").write_text(json.dumps({"episodes": {"0": {"start": 2, "end": 30}}}))
    ia, ib = D.info(D.resolve("cube_a", tmp_path)), D.info(b)
    assert ia["tasks"] == ["pick up the red cube"] and ia["episodes"] == 2
    assert ia["cameras"] == {"scene": [48, 64, 3], "wrist": [48, 64, 3]}
    assert ia["state_names"] == JOINTS
    assert D.check([ia, ib]) == ([], [])

    plan = D.plan_episodes(a)
    assert all(p["start"] > 0 and p["end"] < p["frames"] for p in plan)   # idle trimmed at both ends
    rep = D.build([a, b], tmp_path / "train", "local/train", log=lambda m: None)
    assert rep["episodes"] == 3 and set(rep["tasks"]) == {"pick up the red cube", "put the cube in the bowl"}
    assert [r["kept"] for r in rep["report"] if r["source"] == "cube_b"] == [28]   # the manual trim
    built = D.info(tmp_path / "train")
    assert set(built["tasks"]) == set(rep["tasks"]) and built["frames"] == rep["frames"]

    f = D.facts(tmp_path / "train")
    assert set(f) == {"envelope", "step_p99", "start_pose"}
    assert f["envelope"]["gripper"][1] <= 50.0 + 1e-3 and f["step_p99"]["joint1"] > 0

    over = D.build([b], tmp_path / "over", "local/over", log=lambda m: None, task_override="grab it")
    assert over["tasks"] == ["grab it"]
