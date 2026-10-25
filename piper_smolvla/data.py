"""The datasets SmolVLA trains on: finding them, checking them, and building one training set from several.

Datasets come from the ACT engine's recorder (ACT Studio's Record tab): LeRobot v3, 15 fps, observation.state
and action = [joint1..joint6 deg, gripper mm], action[t] = state[t+1], one video per camera role. The recorder
refuses to open a dataset without a one-sentence task, and stores it with every frame (meta/tasks.parquet) and
in nervus_dataset.json. SmolVLA is conditioned on that sentence, so it is the instruction to give at run time.

build() is the Mac worker's builder (act/mac/build.py in NervusOS, same author) with one change: each frame
keeps the task it was recorded with (the worker relabels every frame with the dataset's newest task, which is
harmless for ACT, which ignores language). Its rules: the synced recordings are never modified; excluded
episodes are left out, manual trims honoured, and otherwise the idle stillness before the first motion and
after the last is trimmed (18 of the first 50 demonstrations waited more than 2 s before moving, and the policy
learned "wait here"); video is re-encoded as h264 at CRF 18 (LeRobot's default CRF 30 lost 3.8/255 on average).
LeRobot 0.4.4 cannot train on several datasets at once, so several sources are merged here.
"""
from __future__ import annotations

import glob
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .clients import JOINTS

VCODEC = "h264"
CRF = 18
PLACEHOLDER_TASKS = {"", "demonstration", "task", "test", "demo"}


def read_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


# ------------------------------------------------------------------------------------------------ finding
def resolve(ref: str, shared: Path) -> Path:
    """A dataset path (a directory with meta/info.json), or the name of one in <shared>/datasets."""
    p = Path(ref).expanduser()
    for cand in (p, Path(shared).expanduser() / "datasets" / ref):
        if (cand / "meta" / "info.json").is_file():
            return cand.resolve()
    raise FileNotFoundError(f"no LeRobot dataset at {ref!r} or {Path(shared) / 'datasets' / ref}")


def tasks_of(root: Path) -> list[str]:
    try:
        import pandas as pd
        df = pd.read_parquet(root / "meta" / "tasks.parquet")
        return [str(t) for t in df.index]
    except Exception:
        meta = read_json(root / "nervus_dataset.json", {}) or {}
        return [meta["task"]] if meta.get("task") else []


def info(root: Path) -> dict:
    """What training needs to know about one dataset, from its metadata only (no video is read)."""
    root = Path(root)
    inf = json.loads((root / "meta" / "info.json").read_text())
    feats = inf.get("features", {})
    cams = {k.removeprefix("observation.images."): f["shape"] for k, f in feats.items()
            if k.startswith("observation.images.")}
    return {
        "name": root.name, "root": str(root), "fps": inf.get("fps"), "episodes": inf.get("total_episodes", 0),
        "frames": inf.get("total_frames", 0), "cameras": dict(sorted(cams.items())),
        "state_names": (feats.get("observation.state") or {}).get("names"),
        "action_names": (feats.get("action") or {}).get("names"),
        "robot_type": inf.get("robot_type"), "tasks": tasks_of(root),
        "edits": bool(read_json(root / "nervus_edits.json")), "codebase_version": inf.get("codebase_version"),
    }


def check(infos: list[dict], task_override: str | None = None) -> tuple[list[str], list[str]]:
    """(problems that stop training, warnings) for a set of datasets trained together."""
    problems, warnings = [], []
    if not infos:
        return ["no datasets given"], warnings
    ref = infos[0]
    for d in infos:
        if d["state_names"] != ref["state_names"] or d["action_names"] != ref["action_names"]:
            problems.append(f"{d['name']}: state/action names {d['state_names']} differ from {ref['name']}'s")
        if sorted(d["cameras"]) != sorted(ref["cameras"]):
            problems.append(f"{d['name']}: cameras {sorted(d['cameras'])} differ from {ref['name']}'s "
                            f"{sorted(ref['cameras'])} (datasets trained together must share the camera set)")
        if d["fps"] != ref["fps"]:
            problems.append(f"{d['name']}: {d['fps']} fps, {ref['name']} has {ref['fps']}")
        if not d["cameras"]:
            problems.append(f"{d['name']}: no camera features; SmolVLA needs at least one image")
        if d["episodes"] == 0:
            problems.append(f"{d['name']}: no saved episodes")
        if not task_override:
            bad = [t for t in d["tasks"] if t.strip().lower() in PLACEHOLDER_TASKS]
            if not d["tasks"]:
                problems.append(f"{d['name']}: no task text; record with a task sentence or pass --task")
            elif bad:
                warnings.append(f"{d['name']}: task {bad[0]!r} is a placeholder; SmolVLA follows the task text, "
                                "so pass --task \"pick up the red cube\" (or similar) to train on a real one")
    if ref["state_names"] and list(ref["state_names"]) != JOINTS:
        warnings.append(f"state names {ref['state_names']} are not the ACT engine's {JOINTS}; the PiPER robot "
                        "class sends and reads exactly those")
    return problems, warnings


# ------------------------------------------------------------------------------------------------ building
# Adapted from izhangfly/NervusOS-Cambridge/act/mac/build.py (idle_bounds, frames_table, plan_episodes, features_of, build, facts).
def idle_bounds(states: np.ndarray, threshold: float, keep_before: int, keep_after: int) -> tuple[int, int]:
    """Same rule as the Data tab shows (act/engine/nervus_act/datasets.py), so what you see is what trains."""
    if len(states) < 2:
        return 0, len(states)
    thr = np.array([threshold] * 6 + [2.0])[: states.shape[1]]
    moving_lead = (np.abs(states - states[0]) > thr).any(axis=1)
    moving_tail = (np.abs(states - states[-1]) > thr).any(axis=1)
    start = max(0, int(np.argmax(moving_lead)) - keep_before) if moving_lead.any() else 0
    end = min(len(states), len(states) - int(np.argmax(moving_tail[::-1])) + keep_after) if moving_tail.any() else len(states)
    return start, max(start + 1, end)


def frames_table(root: Path):
    import pandas as pd
    files = sorted(glob.glob(str(Path(root) / "data" / "*" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files])
    return df.sort_values("index")


def plan_episodes(root: Path, auto_trim: bool = True) -> list[dict]:
    """[{episode, start, end, frames}] for one source dataset after its edits (nervus_edits.json)."""
    edits = {"auto_trim": auto_trim, "keep_before": 5, "keep_after": 5, "threshold_deg": 1.0, "episodes": {}}
    edits.update(read_json(Path(root) / "nervus_edits.json", {}) or {})
    df = frames_table(root)
    out = []
    for ep, g in df.groupby("episode_index"):
        man = edits["episodes"].get(str(int(ep)), {})
        if man.get("exclude"):
            continue
        S = np.stack(g["observation.state"].values).astype(np.float64)
        if "start" in man or "end" in man:
            s0, s1 = int(man.get("start", 0)), int(man.get("end", len(S)))
        elif edits["auto_trim"] and auto_trim:
            s0, s1 = idle_bounds(S, edits["threshold_deg"], edits["keep_before"], edits["keep_after"])
        else:
            s0, s1 = 0, len(S)
        s0, s1 = max(0, s0), min(len(S), s1)
        if s1 - s0 >= 15:  # at least one second left
            out.append({"episode": int(ep), "start": s0, "end": s1, "frames": len(S)})
    return out


def features_of(root: Path) -> dict:
    inf = json.loads((Path(root) / "meta" / "info.json").read_text())
    out = {}
    for k, f in inf["features"].items():
        if k in ("observation.state", "action") or k.startswith("observation.images."):
            f = dict(f)
            f["shape"] = tuple(f["shape"])  # info.json has lists; LeRobot's frame validation wants tuples
            f.pop("info", None)             # codec info is regenerated on encode
            out[k] = f
    return out


def to_uint8_hwc(chw):
    import torch
    return torch.round(chw.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()


class Cancelled(Exception):
    pass


def build(sources: list[Path], dest: Path, repo_id: str, log=print, task_override: str | None = None,
          auto_trim: bool = True, cancel=None) -> dict:
    """Write one training dataset to dest from several recordings (atomically: dest.partial, then rename)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    feats = features_of(sources[0])
    fps = json.loads((Path(sources[0]) / "meta" / "info.json").read_text())["fps"]
    cams = [k.removeprefix("observation.images.") for k in feats if k.startswith("observation.images.")]
    out = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=feats, root=tmp, robot_type="agilex_piper",
                                use_videos=True, vcodec=VCODEC, streaming_encoding=True)
    enc = getattr(out, "_streaming_encoder", None)
    if enc is not None:
        enc.crf = CRF
    report, t0 = [], time.time()
    for src in sources:
        src = Path(src)
        plan = plan_episodes(src, auto_trim)
        ds = LeRobotDataset(f"local/{src.name}", root=src)
        ranges = {int(e["episode_index"]): (int(e["dataset_from_index"]), int(e["dataset_to_index"]))
                  for e in ds.meta.episodes}
        for p in plan:
            if cancel and cancel():
                raise Cancelled()
            lo, _hi = ranges[p["episode"]]
            task = None
            for gi in range(lo + p["start"], lo + p["end"]):
                x = ds[gi]
                task = task_override or x.get("task") or "demonstration"
                frame = {"observation.state": x["observation.state"].numpy(), "action": x["action"].numpy(), "task": task}
                for c in cams:
                    frame[f"observation.images.{c}"] = to_uint8_hwc(x[f"observation.images.{c}"])
                out.add_frame(frame)
            out.save_episode()
            report.append({"source": src.name, **p, "kept": p["end"] - p["start"], "task": task})
            log(f"build: {src.name} ep {p['episode']}: kept {p['start']}..{p['end']} of {p['frames']} "
                f"({time.time() - t0:.0f} s)")
    if not report:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError("nothing to train on: every episode was excluded or is shorter than one second after trimming")
    out.finalize()
    tmp.rename(dest)
    summary = {"episodes": len(report), "frames": int(sum(r["kept"] for r in report)), "fps": fps, "cameras": cams,
               "tasks": sorted({r["task"] for r in report}), "report": report, "seconds": round(time.time() - t0, 1)}
    (dest / "BUILD_REPORT.json").write_text(json.dumps(summary, indent=2))
    return summary


def facts(root: Path) -> dict:
    """What the robot side needs to run a policy trained on this data (Mac worker build.facts): the demonstrated
    envelope, the 99th percentile of each joint's per-step change (the step limit is a multiple of it), and the
    typical start pose."""
    df = frames_table(root)
    S = np.stack(df["observation.state"].values).astype(np.float64)
    A = np.stack(df["action"].values).astype(np.float64)
    starts = np.stack([np.stack(g["observation.state"].values)[0] for _, g in df.groupby("episode_index")])
    names = JOINTS[: S.shape[1]]
    return {
        "envelope": {j: [float(S[:, i].min()), float(S[:, i].max())] for i, j in enumerate(names)},
        "step_p99": {j: float(np.percentile(np.abs(A[:, i] - S[:, i]), 99)) for i, j in enumerate(names)},
        "start_pose": {j: float(np.median(starts[:, i])) for i, j in enumerate(names)},
    }
