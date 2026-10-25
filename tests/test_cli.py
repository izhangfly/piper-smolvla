"""Command-line handling and the plans it produces, with no training, no network and no hardware."""
import json
from pathlib import Path

import pytest
import torch

from piper_smolvla import cli
from piper_smolvla import train as T
from piper_smolvla.clients import JOINTS

from .conftest import fake_dataset


@pytest.fixture
def shared(tmp_path):
    s = tmp_path / "Shared" / "NervusACT"
    fake_dataset(s / "datasets" / "cube_v1")
    fake_dataset(s / "datasets" / "cube_v2", task="put the cube in the bowl")
    fake_dataset(s / "datasets" / "wrist_only", cameras=("wrist",))
    fake_dataset(s / "datasets" / "placeholder", task="demonstration")
    return s


def plan(shared, tmp_path, *ds, **kw):
    return T.make_plan(list(ds), shared=str(shared), work=str(tmp_path / "work"), **kw)


def args_dict(args):
    return dict(a.removeprefix("--").split("=", 1) for a in args)


def test_mac_profile_defaults(shared, tmp_path):
    p = plan(shared, tmp_path, "cube_v1")
    a = args_dict(T.lerobot_args(p, "/data", "/base"))
    assert a["policy.device"] == "mps" and a["batch_size"] == "4" and a["num_workers"] == "2"
    assert a["policy.path"] == "/base" and a["dataset.root"] == "/data" and a["policy.push_to_hub"] == "false"
    assert a["dataset.image_transforms.enable"] == "true" and a["wandb.enable"] == "false"
    assert p.steps == 20_000 and p.base == "lerobot/smolvla_base" and p.role == "trained"
    assert p.publish_dir == str(shared / "smolvla" / "policies")


def test_remote_defaults_to_the_cuda_profile(shared, tmp_path):
    p = plan(shared, tmp_path, "cube_v1", remote="ian@gpu")
    a = args_dict(T.lerobot_args(p, "/d", "/b"))
    assert a["policy.device"] == "cuda" and a["batch_size"] == "64"


def test_posttrain_starts_from_the_parent(shared, tmp_path):
    parent = shared / "smolvla" / "policies" / "cube_v1-smolvla_020000"
    parent.mkdir(parents=True)
    (parent / "config.json").write_text(json.dumps({"type": "smolvla"}))
    p = plan(shared, tmp_path, "cube_v1", "cube_v2", parent="cube_v1-smolvla_020000")
    assert p.steps == T.POSTTRAIN_STEPS and p.base is None and p.role == "candidate"
    assert p.parent == str(parent.resolve())
    (parent / "config.json").write_text(json.dumps({"type": "act"}))
    with pytest.raises(ValueError, match="SmolVLA"):
        plan(shared, tmp_path, "cube_v1", parent="cube_v1-smolvla_020000")


def test_dataset_checks(shared, tmp_path, capsys):
    with pytest.raises(ValueError, match="camera"):
        plan(shared, tmp_path, "cube_v1", "wrist_only")
    with pytest.raises(ValueError, match="no-build"):
        plan(shared, tmp_path, "cube_v1", "cube_v2", build=False)
    with pytest.raises(FileNotFoundError):
        plan(shared, tmp_path, "nope")
    plan(shared, tmp_path, "placeholder")
    assert "placeholder" in capsys.readouterr().err
    p = plan(shared, tmp_path, "placeholder", task="pick up the red cube")
    assert p.task == "pick up the red cube"
    nt = fake_dataset(tmp_path / "no_task", task=None)
    (nt / "nervus_dataset.json").unlink()
    with pytest.raises(ValueError, match="task"):
        plan(shared, tmp_path, str(nt))


def test_supervisor_config_publishes_the_final_checkpoint(shared, tmp_path):
    p = plan(shared, tmp_path, "cube_v1", steps=100, save_freq=50)
    cfg = T.supervisor_config(p, "/out", ["--x=1"], "/bin/lerobot-train", True, {"a": 1})
    assert cfg["required_extra"] == ["training_state/scheduler_state.json"]
    assert cfg["publish_min_step"] == 100 and cfg["publish_prefix"] == "cube_v1-smolvla"
    assert cfg["publish_dir"] == p.publish_dir
    p2 = plan(shared, tmp_path, "cube_v1", steps=100, save_freq=50, publish_every=50)
    assert T.supervisor_config(p2, "/o", [], "x", False, {})["publish_min_step"] == 50


def test_publish_meta_carries_the_robot_facts(shared, tmp_path):
    p = plan(shared, tmp_path, "cube_v1", "cube_v2")
    m = T.publish_meta(p, {"envelope": {}, "step_p99": {}, "start_pose": {}}, None)
    assert m["policy"] == "smolvla" and m["cameras"] == ["scene", "wrist"] and m["fps"] == 15
    assert set(m["tasks"]) == {"pick up the red cube", "put the cube in the bowl"}
    assert m["units"] == {"joints": "deg", "gripper": "mm"} and "envelope" in m


def test_cli_train_print_only(shared, tmp_path, capsys):
    rc = cli.main(["train", "cube_v1", "--shared", str(shared), "--work-dir", str(tmp_path / "w"), "--steps", "2",
                   "--batch-size", "1", "--print-only", "--policy.optimizer_lr=5e-5"])
    out = capsys.readouterr().out
    assert rc == 0 and "lerobot-train" in out and "--steps=2" in out and "--policy.optimizer_lr=5e-5" in out
    assert not (tmp_path / "w").exists()                      # --print-only writes nothing


def test_cli_remote_print_only_runs_no_ssh(shared, tmp_path, capsys, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("ran a command"))
    rc = cli.main(["train", "cube_v1", "--shared", str(shared), "--work-dir", str(tmp_path / "w"),
                   "--remote", "ian@gpu", "--print-only"])
    out = capsys.readouterr().out
    assert rc == 0 and "rsync" in out and "remote_launch.sh" in out and "--policy.device=cuda" in out


def test_cli_rejects_bad_arguments(capsys):
    with pytest.raises(SystemExit):
        cli.main(["posttrain", "cube_v1"])                    # --parent is required
    with pytest.raises(SystemExit):
        cli.main(["train", "cube_v1", "-x"])                  # not --key=value: not for lerobot-train either
    with pytest.raises(SystemExit):
        cli.main(["serve", "--policy.x=1"])
    with pytest.raises(SystemExit):
        cli.main(["run", "--policy", "p"])                    # --task is required


def test_cli_datasets_lists_tasks(shared, capsys):
    assert cli.main(["datasets", "--shared", str(shared)]) == 0
    out = capsys.readouterr().out
    assert "cube_v2" in out and "put the cube in the bowl" in out


# ---- run ----
@pytest.fixture
def policy(shared):
    d = shared / "smolvla" / "policies" / "cube_v1-smolvla_020000"
    d.mkdir(parents=True)
    feats = {"observation.state": {"type": "STATE", "shape": [7]},
             "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
             "observation.images.scene": {"type": "VISUAL", "shape": [3, 480, 640]}}
    (d / "config.json").write_text(json.dumps({"type": "smolvla", "input_features": feats, "chunk_size": 50,
                                               "output_features": {"action": {"type": "ACTION", "shape": [7]}}}))
    (d / "nervus_policy.json").write_text(json.dumps({
        "envelope": {j: [-10, 10] for j in JOINTS}, "step_p99": {j: 1.0 for j in JOINTS}, "fps": 15,
        "cameras": ["scene", "wrist"], "start_pose": {j: 0.0 for j in JOINTS}}))
    return d


def test_run_plan_resolves_a_policy_name(shared, policy):
    from piper_smolvla.run import RunOptions, plan_run
    pl = plan_run(RunOptions(policy=policy.name, task="pick up the red cube", shared=str(shared)))
    assert pl["send_ref"] == policy.name                       # the server resolves the name in its own folder
    assert pl["local_policy"] == policy and pl["facts_path"] == str(policy)
    assert pl["cameras"] == ["scene", "wrist"] and pl["image_size"] == (480, 640)
    assert pl["fps"] == 15 and pl["actions_per_chunk"] == 50
    with pytest.raises(ValueError, match="task"):
        plan_run(RunOptions(policy=policy.name, task=" ", shared=str(shared)))
    hub = plan_run(RunOptions(policy="someone/smolvla_piper", task="t", shared=str(shared), cameras=["wrist"]))
    assert hub["local_policy"] is None and hub["cameras"] == ["wrist"] and hub["facts"] is None


def test_client_config_is_lerobots(shared, policy):
    from piper_smolvla.run import AGGREGATE_NAME, RunOptions, client_config, plan_run
    o = RunOptions(server="127.0.0.1:9999", policy=policy.name, task="pick up the red cube", shared=str(shared),
                   dry_run=True)
    cfg = client_config(o, plan_run(o))
    assert cfg.robot.type == "piper_http" and cfg.robot.dry_run and cfg.robot.camera_roles == ["scene", "wrist"]
    assert cfg.policy_type == "smolvla" and cfg.pretrained_name_or_path == policy.name
    assert cfg.fps == 15 and cfg.actions_per_chunk == 50 and cfg.aggregate_fn_name == AGGREGATE_NAME
    assert cfg.robot.facts_path == str(policy)


def test_gripper_is_never_averaged():
    from piper_smolvla.run import piper_aggregate
    old, new = torch.zeros(7), torch.ones(7) * 10
    out = piper_aggregate(old, new)
    assert out[0] == pytest.approx(7.0) and out[6] == 10.0


def test_cli_run_refuses_on_preflight_errors(shared, policy, capsys):
    # Nothing listens on these ports: the preflight reports it and the run never starts.
    rc = cli.main(["run", "--policy", policy.name, "--task", "pick", "--shared", str(shared), "--dry-run",
                   "--driver", "http://127.0.0.1:9", "--engine", "http://127.0.0.1:9", "--server", "127.0.0.1:9"])
    out = capsys.readouterr().out
    assert rc == 2 and "not running" in out and "[error]" in out


def test_serve_resolves_names(tmp_path):
    from piper_smolvla.serve import resolve_name
    (tmp_path / "p1").mkdir()
    (tmp_path / "p1" / "config.json").write_text("{}")
    assert resolve_name("p1", tmp_path) == str(tmp_path / "p1")
    assert resolve_name("lerobot/smolvla_base", tmp_path) == "lerobot/smolvla_base"
    assert resolve_name("missing", tmp_path) == "missing"


def test_base_preparation_leaves_inputs_to_the_dataset(tmp_path):
    from piper_smolvla.base import prepare
    src = tmp_path / "hub"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"type": "smolvla", "input_features": {
        "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}}, "output_features": {}}))
    (src / "model.safetensors").write_bytes(b"x")
    (src / "policy_preprocessor.json").write_text("{}")
    dest = prepare(str(src), tmp_path / "base")
    assert json.loads((dest / "config.json").read_text())["input_features"] == {}
    assert (dest / "model.safetensors").is_symlink() and (dest / "policy_preprocessor.json").exists()
    assert prepare(str(src), tmp_path / "base") == dest       # idempotent


def test_publish_run_copies_complete_checkpoints(tmp_path):
    from safetensors.torch import save_file
    work = tmp_path / "work"
    run = work / "runs" / "cube"
    pub = tmp_path / "pub"
    T.write_json(run / "plan.json", {"steps": 4, "publish_dir": str(pub), "publish_every": None})
    T.write_json(run / "publish_meta.json", {"policy": "smolvla", "envelope": {}})
    for step, whole in ((2, True), (4, True)):
        pm = run / "out" / "checkpoints" / f"{step:06d}" / "pretrained_model"
        pm.mkdir(parents=True)
        (pm / "config.json").write_text('{"type": "smolvla"}')
        save_file({"w": torch.zeros(2)}, str(pm / "model.safetensors"))
        for pj in ("policy_preprocessor.json", "policy_postprocessor.json"):
            (pm / pj).write_text('{"steps": []}')
    out = T.publish_run("cube", work)
    assert out == ["cube-smolvla_000004"]                      # the final one only
    meta = json.loads((pub / "cube-smolvla_000004" / "nervus_policy.json").read_text())
    assert meta["step"] == 4 and meta["policy"] == "smolvla" and "created" in meta
    # a truncated weights file is not published
    pm = run / "out" / "checkpoints" / "000004" / "pretrained_model"
    raw = (pm / "model.safetensors").read_bytes()
    (pm / "model.safetensors").write_bytes(raw[:-3])
    import shutil
    shutil.rmtree(pub)
    assert T.publish_run("cube", work) == []


def test_local_guard_refuses_next_to_another_trainer(monkeypatch):
    monkeypatch.setattr(T, "busy_trainers", lambda: ["123 lerobot-train --policy.type=act"])
    with pytest.raises(RuntimeError, match="another training run"):
        T.guard_local(force=False)
    T.guard_local(force=True)


def test_status_line():
    assert T.status_line({}) == "starting (no status yet)"
    line = T.status_line({"state": "training", "step": 50, "target": 100, "metrics": {"loss": "0.12"},
                          "eta_hours": 0.5, "checkpoints": ["000050"]})
    assert "step 50/100" in line and "loss 0.12" in line and "000050" in line


def test_package_ships_the_remote_scripts():
    env = Path(T.package_dir()) / "env"
    for f in ("remote_launch.sh", "remote_run.sh", "linux-cuda.txt"):
        assert (env / f).is_file()
    assert "lerobot==0.4.4" in (env / "linux-cuda.txt").read_text()
