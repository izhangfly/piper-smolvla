import math

import pytest

from piper_smolvla.clients import JOINTS
from piper_smolvla.safety import (FALLBACK_STEP_P99, SafetyFilter, SafetyParams, UnsafeAction, complete_facts,
                                  load_facts, preview_verdict)

READY = {"joint1": 0.0, "joint2": 45.0, "joint3": -60.0, "joint4": 0.0, "joint5": 45.0, "joint6": 0.0, "gripper": 10.0}


def facts(**over):
    f = {"envelope": {j: [-50.0, 50.0] for j in JOINTS[:6]}, "step_p99": {j: 1.0 for j in JOINTS}}
    f["envelope"].update({"joint2": [0.0, 90.0], "joint3": [-120.0, 0.0], "gripper": [0.0, 60.0]})
    f.update(over)
    return complete_facts(f, None)


def test_fallbacks_match_the_act_runner():
    f = complete_facts(None, {"joint1": (-154.0, 154.0)})
    assert f["envelope"]["joint1"] == [-154.0, 154.0]
    assert f["envelope"]["gripper"] == [0.0, 70.0]
    assert all(f["step_p99"][j] == FALLBACK_STEP_P99 for j in JOINTS)
    assert not f["from_training_data"]


def test_envelope_margin_and_joint_limits():
    sf = SafetyFilter(facts(), SafetyParams(step_limit_x=0, lead_limit_x=0, envelope_margin=5.0, tracking_stop_deg=0))
    cmd, _ = sf(dict(READY, joint1=80.0, joint3=10.0), READY)
    assert cmd["joint1"] == 55.0            # envelope 50 + margin 5
    assert cmd["joint3"] == 0.0             # envelope says 5, the joint limit says 0
    assert sf.counts["envelope"] >= 1 and sf.counts["joint_limit"] >= 1


def test_step_limit_applies_to_the_command_from_the_first_step():
    sf = SafetyFilter(facts(), SafetyParams(step_limit_x=1.5, lead_limit_x=0, tracking_stop_deg=0))
    want = dict(READY, joint1=30.0)
    seen = []
    for _ in range(4):
        cmd, _ = sf(want, READY)            # the arm doesn't move: the limit is on the command, not the arm
        seen.append(cmd["joint1"])
    assert seen == pytest.approx([1.5, 3.0, 4.5, 6.0])


def test_lead_limit_caps_the_command_ahead_of_the_arm():
    sf = SafetyFilter(facts(), SafetyParams(step_limit_x=0, lead_limit_x=4.0, tracking_stop_deg=0))
    cmd, _ = sf(dict(READY, joint1=30.0), READY)
    assert cmd["joint1"] == 4.0             # p99 1.0 x 4
    cmd, _ = sf(dict(READY, joint1=30.0), dict(READY, joint1=3.0))
    assert cmd["joint1"] == 7.0


def test_gripper_step_is_three_times_the_joint_step():
    sf = SafetyFilter(facts(), SafetyParams(step_limit_x=1.0, lead_limit_x=0, tracking_stop_deg=0,
                                            gripper_squeeze_mm=0.0))
    cmd, _ = sf(dict(READY, gripper=50.0), READY)
    assert cmd["gripper"] == pytest.approx(13.0)   # 10 + 3 x 1.0


def test_grasp_latch_holds_then_releases_after_repeated_open_requests():
    sf = SafetyFilter(facts(), SafetyParams(step_limit_x=0, lead_limit_x=0, tracking_stop_deg=0,
                                            gripper_squeeze_mm=1.5, gripper_release_steps=3))
    held = dict(READY, gripper=30.0)        # fingers stopped on a 30 mm object
    cmd, mode = sf(dict(READY, gripper=27.0), held)  # the policy closes: command 25.5 (squeeze)
    for _ in range(6):                      # the width settles, the policy keeps closing
        cmd, mode = sf(dict(READY, gripper=27.0), held)
    assert mode == "hold" and cmd["gripper"] == pytest.approx(28.5)
    cmd, mode = sf(dict(READY, gripper=55.0), held)  # one flicker to open is ignored
    assert mode == "hold"
    cmd, mode = sf(dict(READY, gripper=27.0), held)
    modes = [sf(dict(READY, gripper=55.0), held)[1] for _ in range(3)]
    assert modes[-1] == "release"


def test_non_finite_action_stops_the_run():
    sf = SafetyFilter(facts())
    with pytest.raises(UnsafeAction):
        sf(dict(READY, joint4=math.nan), READY)


def test_tracking_stop():
    sf = SafetyFilter(facts(), SafetyParams(tracking_stop_deg=10.0))
    sf(READY, READY)
    with pytest.raises(UnsafeAction, match="joint2"):
        sf(READY, dict(READY, joint2=60.0))  # the arm is 15 deg from where it was told to be


def test_hold_target_grips_lightly_unless_wide_open():
    sf = SafetyFilter(facts(), SafetyParams(gripper_squeeze_mm=1.5))
    assert sf.hold_target(dict(READY, gripper=20.0))["gripper"] == 18.5
    assert sf.hold_target(dict(READY, gripper=59.0))["gripper"] == 59.0


def test_load_facts(tmp_path):
    (tmp_path / "nervus_policy.json").write_text('{"envelope": {"joint1": [0, 1]}, "step_p99": {"joint1": 2}, "fps": 15}')
    f = load_facts(tmp_path)
    assert f["fps"] == 15 and f["envelope"]["joint1"] == [0, 1]
    assert load_facts(tmp_path / "missing") is None


def test_preview_verdict():
    assert preview_verdict(25) == "plans to move"
    assert preview_verdict(10).startswith("weak")
    assert "sit still" in preview_verdict(2)
