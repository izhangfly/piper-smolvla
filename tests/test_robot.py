"""The LeRobot robot class against the simulated driver and cameras: features, the observation/action round
trip, the safety clamps on the way out, dry runs, refusals and how a run ends."""
import time

import numpy as np
import pytest

from piper_smolvla.clients import JOINTS
from piper_smolvla.safety import UnsafeAction

from .conftest import post


def test_features_match_the_act_engine_datasets(make_robot):
    r = make_robot(camera_roles=["wrist", "scene"])
    assert list(r.action_features) == JOINTS
    assert list(r.observation_features)[:7] == JOINTS
    assert r.observation_features["wrist"] == (480, 640, 3)
    from lerobot.datasets.utils import hw_to_dataset_features
    feats = hw_to_dataset_features(r.observation_features, "observation", use_video=False)
    assert feats["observation.state"]["names"] == JOINTS
    assert set(feats) == {"observation.state", "observation.images.wrist", "observation.images.scene"}


def test_lerobot_finds_the_robot_class_by_type(tmp_path):
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.config import RobotConfig

    from piper_smolvla.config import PiperHttpConfig
    assert RobotConfig.get_choice_class("piper_http") is PiperHttpConfig
    robot = make_robot_from_config(PiperHttpConfig(driver_url="http://127.0.0.1:1", camera_url="http://127.0.0.1:1",
                                                   calibration_dir=tmp_path))
    assert type(robot).__name__ == "PiperHttp" and not robot.is_connected


def test_observation_round_trip(sim, make_robot):
    drv, _ = sim
    r = make_robot()
    r.connect()
    obs = r.get_observation()
    for i, j in enumerate(JOINTS[:6]):
        assert obs[j] == pytest.approx(drv.arm.joints[i], abs=0.01)
    assert obs["gripper"] == pytest.approx(drv.arm.gripper, abs=0.01)
    for role in ("wrist", "scene"):
        assert obs[role].shape == (480, 640, 3) and obs[role].dtype == np.uint8
    assert r.facts["envelope"]["joint1"] == [-154.0, 154.0]   # the driver's limits, without training facts


def test_actions_are_streamed_and_clamped(sim, make_robot):
    drv, _ = sim
    r = make_robot()
    r.connect()
    obs = r.get_observation()
    want = {j: obs[j] for j in JOINTS}
    want["joint1"] = obs["joint1"] + 40.0
    cmds = []
    for _ in range(6):
        cmds.append(r.send_action(want)["joint1"])
        time.sleep(1 / 15)
    steps = np.diff([obs["joint1"]] + cmds)
    assert (steps <= 6.0 + 1e-6).all()                        # 1.5 x the fallback p99 of 4 deg
    assert drv.posts["/api/stream"] == 6
    assert drv.arm.owner["client"] == "smolvla"
    assert drv.arm.target[0] == pytest.approx(cmds[-1])
    assert r.report()["clamps"]["step"] > 0


def test_disconnect_holds_and_releases(sim, make_robot):
    drv, _ = sim
    r = make_robot()
    r.connect()
    obs = r.get_observation()
    r.send_action({j: obs[j] for j in JOINTS})
    n = drv.posts["/api/stream"]
    r.disconnect()
    assert drv.posts["/api/stream"] == n + 1                  # the hold command
    assert drv.posts["/api/release"] == 1
    assert drv.arm.owner is None


def test_dry_run_never_moves_the_arm(sim, make_robot):
    drv, _ = sim
    post(drv.url, "/api/enable", {"on": False})               # a dry run doesn't need the motors on
    r = make_robot(dry_run=True)
    r.connect()
    obs = r.get_observation()
    for _ in range(5):
        cmd = r.send_action(dict({j: obs[j] for j in JOINTS}, joint2=obs["joint2"] + 20))
    assert cmd["joint2"] > obs["joint2"]
    r.disconnect()
    assert sum(drv.posts.values()) == 1                       # only the test's own /api/enable
    assert r.report()["max_plan_deg"] == pytest.approx(20, abs=0.1)


@pytest.mark.parametrize("setup,match", [
    (lambda url: post(url, "/api/enable", {"on": False}), "enable the motors"),
    (lambda url: post(url, "/sim/teach", {"on": True}), "drag-teach"),
    (lambda url: post(url, "/api/gripper", {"client": "panel", "mm": 10}), "another app"),
])
def test_live_connect_refuses(sim, make_robot, setup, match):
    drv, _ = sim
    setup(drv.url)
    r = make_robot()
    with pytest.raises(ConnectionError, match=match):
        r.connect()
    assert drv.posts["/api/speed"] == 0 and drv.posts["/api/stream"] == 0


def test_wrong_camera_size_refuses(make_robot):
    r = make_robot(image_width=320, image_height=240)
    with pytest.raises(ConnectionError, match="expects"):
        r.connect()


def test_stale_arm_feedback_holds_then_stops(sim, make_robot):
    drv, _ = sim
    r = make_robot(stale_grace_s=0.3)
    r.connect()
    obs = r.get_observation()
    r.send_action({j: obs[j] for j in JOINTS})
    n = drv.posts["/api/stream"]
    post(drv.url, "/sim/stale", {"on": True})
    time.sleep(0.5)
    with pytest.raises(RuntimeError):
        r.get_observation()
    r.send_action({j: obs[j] for j in JOINTS})                # held: nothing new is sent
    assert drv.posts["/api/stream"] == n
    time.sleep(0.4)
    with pytest.raises(UnsafeAction, match="stale"):
        r.send_action({j: obs[j] for j in JOINTS})
    assert drv.posts["/api/stream"] == n


def test_frozen_camera_stops_the_run(sim, make_robot):
    drv, cams = sim
    r = make_robot(stale_grace_s=0.2, stale_camera_ms=300)
    r.connect()
    obs = r.get_observation()
    cams.freeze("wrist")
    time.sleep(0.5)
    r.send_action({j: obs[j] for j in JOINTS})                # first stale step: hold
    time.sleep(0.3)
    with pytest.raises(UnsafeAction, match="wrist"):
        r.send_action({j: obs[j] for j in JOINTS})


def test_drag_teach_mid_run_hands_the_arm_over(sim, make_robot):
    drv, _ = sim
    r = make_robot()
    r.connect()
    obs = r.get_observation()
    r.send_action({j: obs[j] for j in JOINTS})
    post(drv.url, "/sim/teach", {"on": True})
    time.sleep(0.1)
    with pytest.raises(UnsafeAction, match="drag-teach"):
        r.send_action({j: obs[j] for j in JOINTS})
    n = sum(drv.posts.values())
    r.disconnect()
    assert sum(drv.posts.values()) == n                       # nothing that would stiffen the arm in a hand


def test_losing_control_stops_the_run(sim, make_robot):
    drv, _ = sim
    r = make_robot()
    r.connect()
    obs = r.get_observation()
    r.send_action({j: obs[j] for j in JOINTS})
    time.sleep(0.6)                                           # our stream deadman lapses...
    post(drv.url, "/api/gripper", {"client": "panel", "mm": 10})  # ...and the panel takes the arm
    with pytest.raises(UnsafeAction, match="lost control"):
        r.send_action({j: obs[j] for j in JOINTS})


def test_training_facts_set_the_limits(sim, make_robot, tmp_path):
    import json
    drv, _ = sim
    f = {"envelope": {j: [-10.0, 10.0] for j in JOINTS[:6]} | {"joint2": [40, 50], "joint3": [-65, -55],
                                                                "joint5": [40, 50], "gripper": [0, 40]},
         "step_p99": {j: 0.5 for j in JOINTS}, "start_pose": {j: 0.0 for j in JOINTS}, "fps": 15}
    (tmp_path / "nervus_policy.json").write_text(json.dumps(f))
    r = make_robot(facts_path=str(tmp_path))
    r.connect()
    obs = r.get_observation()
    cmd = r.send_action(dict({j: obs[j] for j in JOINTS}, joint1=90.0))
    assert cmd["joint1"] == pytest.approx(0.75)               # 1.5 x p99 0.5 from where the arm is
    assert r.report()["facts_from_training_data"]


def test_a_still_arm_ends_a_live_run(sim, make_robot):
    r = make_robot(stagnation_s=0.5)
    r.connect()
    obs = r.get_observation()
    hold = {j: obs[j] for j in JOINTS}
    with pytest.raises(UnsafeAction, match="has not moved"):
        for _ in range(30):
            r.send_action(hold)
            time.sleep(1 / 15)
    dry = make_robot(stagnation_s=0.5, dry_run=True)        # a dry run never moves the arm: no stagnation stop
    dry.connect()
    for _ in range(12):
        dry.send_action(hold)
        time.sleep(1 / 15)
