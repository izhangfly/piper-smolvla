"""The starting point of a fine-tune: lerobot/smolvla_base, with its input features left for the dataset to set.

LeRobot 0.4.4 refreshes a pretrained policy's output features from the dataset but keeps its input features
(huggingface/lerobot#4517, the root cause of #2418). smolvla_base declares a 6-value state and three 256x256
cameras called camera1..camera3 (its SO-100 pretraining data), so a plain `--policy.path=lerobot/smolvla_base`
fine-tune on our data either fails the camera check or, with a rename map, trains fine but saves a config that
says the policy wants 256x256 camera1..3. LeRobot's async policy server resizes incoming images to the shapes in
that config, so at run time our 640x480 frames would be squashed to 256x256 before SmolVLA pads them, which is
not what it was trained on.

The fix: train from a local copy of the base whose config.json has empty input_features. make_policy then fills
them from the dataset (observation.state of 7 values, the camera roles at their recorded size), the camera check
passes, and the fine-tuned policy's config describes exactly what the robot sends. The weights are symlinked
(or copied with --copy), not changed: SmolVLA pads state and actions to 32 and projects them with layers whose
shapes don't depend on the dataset.

    python -m piper_smolvla.base lerobot/smolvla_base <dest>      # also run on the GPU box before training
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

DEFAULT_BASE = "lerobot/smolvla_base"


def fetch(ref: str) -> Path:
    """A local directory holding the base: the path itself, or a Hugging Face Hub snapshot."""
    p = Path(ref).expanduser()
    if (p / "config.json").is_file():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=ref, allow_patterns=["*.json", "*.safetensors", "*.md"]))


def prepare(ref: str, dest: Path, copy: bool = False) -> Path:
    """Write dest/ = the base with input_features emptied. Idempotent; returns dest."""
    src = fetch(ref)
    dest = Path(dest)
    marker = dest / ".piper_smolvla_base.json"
    if marker.is_file() and json.loads(marker.read_text()).get("source") == str(src):
        return dest
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.name == "config.json" or f.name.startswith(".") or not f.is_file():
            continue
        if copy:
            shutil.copy2(f, tmp / f.name)
        else:
            os.symlink(f.resolve(), tmp / f.name)
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("type") != "smolvla":
        raise ValueError(f"{ref} is a {cfg.get('type')!r} policy, not smolvla")
    cfg["input_features"] = {}
    (tmp / "config.json").write_text(json.dumps(cfg, indent=4))
    (tmp / ".piper_smolvla_base.json").write_text(json.dumps({"base": ref, "source": str(src)}))
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base", nargs="?", default=DEFAULT_BASE)
    ap.add_argument("dest")
    ap.add_argument("--copy", action="store_true", help="copy the weights instead of symlinking them")
    a = ap.parse_args()
    print(prepare(a.base, Path(a.dest), copy=a.copy))


if __name__ == "__main__":
    main()
