"""The GPU-box side without a GPU box: remote_launch.sh and remote_run.sh run here with stand-ins for nvidia-smi,
the training venv and lerobot-train, and `pull` runs against a local directory instead of ssh."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from piper_smolvla import train as T

ENV = Path(T.package_dir()) / "env"

FAKE_TRAIN = r'''#!{py}
"""Stands in for lerobot-train: writes one complete checkpoint at the target step, like lerobot 0.4.4."""
import json, os, sys
from pathlib import Path
import torch
from safetensors.torch import save_file
args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
open(os.path.join(os.path.dirname(__file__), "argv.json"), "w").write(json.dumps(sys.argv[1:]))
step = int(args["steps"])
out = Path(args.get("output_dir") or Path(args["config_path"]).parents[3])
ck = out / "checkpoints" / f"{step:06d}"
pm, ts = ck / "pretrained_model", ck / "training_state"
pm.mkdir(parents=True, exist_ok=True); ts.mkdir(parents=True, exist_ok=True)
(pm / "config.json").write_text('{"type": "smolvla"}')
(pm / "train_config.json").write_text("{}")
for f in ("policy_preprocessor.json", "policy_postprocessor.json"):
    (pm / f).write_text('{"steps": []}')
for f in (pm / "model.safetensors", ts / "optimizer_state.safetensors", ts / "rng_state.safetensors"):
    save_file({"w": torch.zeros(2)}, str(f))
(ts / "optimizer_param_groups.json").write_text("[]")
(ts / "training_step.json").write_text(json.dumps({"step": step}))
(ts / "scheduler_state.json").write_text("{}")
print(f"Training: 100%|##| {step}/{step} [00:01<00:00, 1.00step/s]")
print(f"INFO step:{step} smpl:{step} ep:0 epch:0.01 loss:0.123 grdn:1.0 lr:1e-4 updt_s:0.5 data_s:0.01", flush=True)
'''

FAKE_PY = '''#!/bin/bash
# The GPU box's venv python: claims CUDA, otherwise it is this machine's Python.
if [ "$1" = "-c" ]; then echo "torch stub cuda 12.8 available True"; exit 0; fi
exec {py} "$@"
'''


def stub(path: Path, text: str):
    path.write_text(text)
    path.chmod(0o755)


@pytest.mark.skipif(sys.platform == "win32", reason="bash scripts")
def test_remote_scripts_train_and_resume(tmp_path):
    root, name = tmp_path / "box", "cube"
    run = root / "runs" / name
    (root / "src").mkdir(parents=True)
    shutil.copytree(T.package_dir(), root / "src" / "piper_smolvla", ignore=shutil.ignore_patterns("__pycache__"))
    venv = root / "venv" / "bin"
    venv.mkdir(parents=True)
    stub(venv / "python", FAKE_PY.replace("{py}", sys.executable))
    stub(venv / "lerobot-train", FAKE_TRAIN.replace("{py}", sys.executable))
    lock = root / "src" / "piper_smolvla" / "env" / "linux-cuda.txt"
    (root / "venv" / ".lock.sha256").write_text(hashlib.sha256(lock.read_bytes()).hexdigest() + "\n")
    bins = tmp_path / "bin"
    bins.mkdir()
    stub(bins / "nvidia-smi", '#!/bin/bash\ncase "$*" in *memory.free*) echo 23000;; *compute-apps*) ;; '
                              '*) echo "NVIDIA GeForce RTX 4090, 24564 MiB, 570.86";; esac\n')
    stub(bins / "sha256sum", '#!/bin/bash\nshasum -a 256 "$@"\n')
    stub(bins / "setsid", '#!/bin/bash\nexec "$@"\n')
    run.mkdir(parents=True)
    (run / "base").mkdir()
    (run / "base" / "config.json").write_text('{"type": "smolvla"}')
    cfg = {"output_dir": str(run / "out"), "steps": 2, "args": ["--policy.path=x"], "lerobot_train": str(venv / "lerobot-train"),
           "required_extra": T.REQUIRED_EXTRA, "poll_s": 1, "hang_timeout_s": 60, "backoff_base_s": 1, "max_restarts": 2,
           "min_free_gb": 0, "publish_dir": None, "publish_prefix": "cube-smolvla", "publish_min_step": 2,
           "publish_every": 2, "publish_meta": {}}
    (run / "supervisor.json").write_text(json.dumps(cfg))
    # No tmux, no ffmpeg, no apt-get on this PATH: the nohup path and the PyAV fallback.
    env = {"PATH": f"{bins}:/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(tmp_path), "HF_HOME": str(tmp_path / "hf")}
    out = subprocess.run(["bash", str(ENV / "remote_launch.sh"), str(root), name], env=env, capture_output=True,
                         text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert "nohup" in out.stdout
    t_end = time.time() + 60
    while time.time() < t_end:
        st = json.loads((run / "out.status.json").read_text()) if (run / "out.status.json").exists() else {}
        if st.get("state") == "finished":
            break
        time.sleep(0.5)
    log = (run / "remote.log").read_text()
    assert st.get("state") == "finished", log
    assert "RTX 4090" in log and "decoding video with PyAV" in log
    assert "--dataset.video_backend=pyav" in json.loads((run / "supervisor.json").read_text())["args"]
    assert (run / "supervisor.pid").exists()
    # A second launch resumes from the complete checkpoint rather than starting over.
    cfg2 = json.loads((run / "supervisor.json").read_text())
    cfg2["steps"] = 4
    (run / "supervisor.json").write_text(json.dumps(cfg2))
    subprocess.run(["bash", str(ENV / "remote_launch.sh"), str(root), name], env=env, capture_output=True, timeout=30)
    t_end = time.time() + 60
    while time.time() < t_end:
        st = json.loads((run / "out.status.json").read_text())
        if st.get("state") == "finished" and st.get("step") == 4:
            break
        time.sleep(0.5)
    argv = json.loads((venv / "argv.json").read_text())
    assert "--resume=true" in argv and any(a.startswith("--config_path=") and "000002" in a for a in argv)
    assert st.get("step") == 4


def test_pull_copies_back_and_publishes(tmp_path, monkeypatch):
    """pull() with rsync pointed at a local 'box' directory."""
    from safetensors.torch import save_file
    import torch
    work, box, pub = tmp_path / "work", tmp_path / "box", tmp_path / "pub"
    run, rrun = work / "runs" / "cube", box / "runs" / "cube"
    T.write_json(run / "plan.json", {"steps": 2, "publish_dir": str(pub), "publish_every": None})
    T.write_json(run / "publish_meta.json", {"policy": "smolvla"})
    T.write_json(run / "remote.json", {"host": "gpu", "run": str(rrun)})
    pm = rrun / "out" / "checkpoints" / "000002" / "pretrained_model"
    pm.mkdir(parents=True)
    (pm / "config.json").write_text('{"type": "smolvla"}')
    save_file({"w": torch.zeros(2)}, str(pm / "model.safetensors"))
    for f in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (pm / f).write_text('{"steps": []}')
    (rrun / "out" / "checkpoints" / "000002" / "training_state").mkdir()
    (rrun / "out" / "checkpoints" / "000002" / "training_state" / "big.bin").write_bytes(b"0" * 100)
    T.write_json(rrun / "out.status.json", {"state": "finished", "step": 2, "metrics": {"loss": "0.1"}})
    calls = []

    def local_rsync(self, src, dst, excludes=(), delete=False, copy_links=False, timeout=None):
        calls.append((src, dst, excludes))
        s = Path(src[1:] if src.startswith(":") else src)
        d = Path(dst)
        if not s.exists():
            raise RuntimeError("No such file or directory")
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*excludes))
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)

    monkeypatch.setattr(T.Remote, "rsync", local_rsync)
    out = T.pull("cube", work)
    assert out == ["cube-smolvla_000002"]
    assert (pub / "cube-smolvla_000002" / "nervus_policy.json").exists()
    assert not (run / "out" / "checkpoints" / "000002" / "training_state").exists()   # optimizer state stays there
    meta = json.loads((pub / "cube-smolvla_000002" / "nervus_policy.json").read_text())
    assert meta["loss"] == "0.1" and meta["step"] == 2
    assert os.path.exists(run / "remote" / "out.status.json")
