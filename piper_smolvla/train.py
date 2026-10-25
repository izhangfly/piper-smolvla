"""Fine-tuning SmolVLA on PiPER datasets with LeRobot's own trainer, on this Mac or on a rented CUDA box.

Training is `lerobot-train` (never reimplemented) under supervisor.py, which resumes from the last complete
checkpoint after a crash, a hang or a NaN loss, and on every relaunch. A run lives in <work>/runs/<name>/:

    data/              the training set built from the recordings (data.build), unless --no-build
    base/              the starting policy: lerobot/smolvla_base with its input features left to the data
    facts.json         envelope, per-step p99 and start pose of the training data (the robot's safety limits)
    supervisor.json    what the supervisor runs; out/ is lerobot-train's output_dir, out.log its log
    remote.json        where the run lives when it trains on a GPU box

Running `piper-smolvla train` again with the same --name continues the run. Post-training is the same command
with --parent <an earlier SmolVLA policy> and datasets that include the correction episodes.

Policies are published like the ACT worker publishes them (pretrained_model/ files plus nervus_policy.json,
atomic rename) but into <shared>/smolvla/policies/<name>-smolvla_<step>, so the ACT engine, ACT Studio and the
ACT worker never mistake them for ACT policies. The final checkpoint is always published.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import data as D
from .remote import Remote
from .supervisor import pretrained_ok, publish_pretrained

SHARED = Path(os.environ.get("NERVUS_ACT_SHARED", "~/Shared/NervusACT")).expanduser()
WORK = Path(os.environ.get("PIPER_SMOLVLA_WORK", "~/piper-smolvla")).expanduser()
DEFAULT_BASE = "lerobot/smolvla_base"
CONTRACT_VERSION = 1
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,47}$")
SESSION_PREFIX = "psv-"

# Defaults per machine. SmolVLA's authors fine-tuned for 20k steps at batch 64; the Mac's 16 GB fits batch 4.
PROFILES = {
    "mac": {"device": "mps", "batch_size": 4, "num_workers": 2, "steps": 20_000, "save_freq": 2_000, "log_freq": 50},
    "cuda": {"device": "cuda", "batch_size": 64, "num_workers": 8, "steps": 20_000, "save_freq": 5_000, "log_freq": 100},
    "cpu": {"device": "cpu", "batch_size": 2, "num_workers": 0, "steps": 1_000, "save_freq": 500, "log_freq": 10},
}
POSTTRAIN_STEPS = 5_000
# lerobot 0.4.4 writes the scheduler's state last; a checkpoint without it is not complete (supervisor.py).
REQUIRED_EXTRA = ["training_state/scheduler_state.json"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def write_json(p: Path, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, p)


def safe_name(name: str) -> str:
    if not NAME_RE.match(name or ""):
        raise ValueError(f"run names are lowercase letters, digits, '_' and '-', up to 48 characters: {name!r}")
    return name


def resolve_policy(ref: str, shared: Path = SHARED) -> Path:
    """An earlier SmolVLA policy: a directory with config.json, or a name in <shared>/smolvla/policies."""
    for cand in (Path(ref).expanduser(), Path(shared).expanduser() / "smolvla" / "policies" / ref):
        if (cand / "config.json").is_file():
            cfg = read_json(cand / "config.json", {}) or {}
            if cfg.get("type") != "smolvla":
                raise ValueError(f"{cand} is a {cfg.get('type')!r} policy; --parent needs a SmolVLA one")
            return cand.resolve()
    raise FileNotFoundError(f"no SmolVLA policy at {ref!r} or in {Path(shared) / 'smolvla' / 'policies'}")


# ------------------------------------------------------------------------------------------------ the plan
@dataclass
class Plan:
    name: str
    datasets: list[str]                 # resolved dataset roots
    infos: list[dict]
    profile: str
    device: str
    batch_size: int
    num_workers: int
    steps: int
    save_freq: int
    log_freq: int
    base: str | None = DEFAULT_BASE     # a fresh fine-tune starts here ...
    parent: str | None = None           # ... post-training starts from an earlier SmolVLA policy
    task: str | None = None             # replaces every episode's recorded task text
    build: bool = True
    auto_trim: bool = True
    augment: bool = True
    chunk_size: int | None = None
    extra: list[str] = field(default_factory=list)
    work: str = str(WORK)
    shared: str = str(SHARED)
    publish_dir: str = str(SHARED / "smolvla" / "policies")
    publish_every: int | None = None    # also publish every N steps (default: only the final checkpoint)
    remote: str | None = None
    remote_root: str | None = None
    ssh_port: int | None = None

    @property
    def run_dir(self) -> Path:
        return Path(self.work) / "runs" / self.name

    @property
    def role(self) -> str:
        return "candidate" if self.parent else "trained"

    @property
    def prefix(self) -> str:
        return f"{self.name}-smolvla"


def make_plan(datasets: list[str], name: str | None = None, profile: str | None = None, steps: int | None = None,
              batch_size: int | None = None, num_workers: int | None = None, save_freq: int | None = None,
              log_freq: int | None = None, base: str | None = None, parent: str | None = None,
              task: str | None = None, build: bool = True, auto_trim: bool = True, augment: bool = True,
              chunk_size: int | None = None, extra: list[str] | None = None, work: str | None = None,
              shared: str | None = None, publish_dir: str | None = None, publish_every: int | None = None,
              remote: str | None = None, remote_root: str | None = None, ssh_port: int | None = None,
              device: str | None = None) -> Plan:
    shared_p = Path(shared or SHARED).expanduser()
    roots = [D.resolve(d, shared_p) for d in datasets]
    infos = [D.info(r) for r in roots]
    problems, warnings = D.check(infos, task)
    if problems:
        raise ValueError("cannot train on these datasets:\n  " + "\n  ".join(problems))
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if not build and len(roots) > 1:
        raise ValueError("--no-build trains on one dataset as it is; LeRobot 0.4.4 cannot train on several at once")
    if not build and task:
        raise ValueError("--task relabels episodes while the training set is built; it needs the build step")
    profile = profile or ("cuda" if remote else "mac")
    if profile not in PROFILES:
        raise ValueError(f"profile is one of {sorted(PROFILES)}")
    prof = dict(PROFILES[profile])
    if parent and steps is None:
        prof["steps"] = POSTTRAIN_STEPS
    parent_dir = str(resolve_policy(parent, shared_p)) if parent else None
    if parent and chunk_size:
        raise ValueError("--chunk-size can't change when post-training: the parent's chunk size comes with it")
    plan = Plan(
        name=safe_name(name or roots[0].name.replace(".", "-")[:48]), datasets=[str(r) for r in roots], infos=infos,
        profile=profile, device=device or prof["device"], batch_size=batch_size or prof["batch_size"],
        num_workers=prof["num_workers"] if num_workers is None else num_workers, steps=steps or prof["steps"],
        save_freq=save_freq or prof["save_freq"], log_freq=log_freq or prof["log_freq"],
        base=None if parent else (base or DEFAULT_BASE), parent=parent_dir, task=task, build=build,
        auto_trim=auto_trim, augment=augment, chunk_size=chunk_size, extra=list(extra or []),
        work=str(Path(work or WORK).expanduser()), shared=str(shared_p),
        publish_dir=str(Path(publish_dir).expanduser()) if publish_dir else str(shared_p / "smolvla" / "policies"),
        publish_every=publish_every, remote=remote, remote_root=remote_root, ssh_port=ssh_port)
    if plan.save_freq > plan.steps:
        plan.save_freq = plan.steps
    return plan


def lerobot_args(plan: Plan, data_dir: str, policy_dir: str) -> list[str]:
    """lerobot-train's arguments for this plan, except --steps and --output_dir (the supervisor adds those)."""
    args = [f"--policy.path={policy_dir}", f"--policy.device={plan.device}", "--policy.push_to_hub=false",
            f"--dataset.repo_id=local/{plan.name}", f"--dataset.root={data_dir}",
            f"--batch_size={plan.batch_size}", f"--num_workers={plan.num_workers}", f"--save_freq={plan.save_freq}",
            f"--log_freq={plan.log_freq}", f"--job_name={plan.name}-smolvla", "--wandb.enable=false"]
    if plan.augment:
        args.append("--dataset.image_transforms.enable=true")
    if plan.chunk_size:
        args += [f"--policy.chunk_size={plan.chunk_size}", f"--policy.n_action_steps={plan.chunk_size}"]
    return args + list(plan.extra)


# Adapted from izhangfly/NervusOS-Cambridge/act/mac/worker.py (Worker.train publish_meta fields).
def publish_meta(plan: Plan, facts: dict, build_summary: dict | None) -> dict:
    """nervus_policy.json, in the ACT worker's fields plus what SmolVLA adds (the task texts)."""
    first = plan.infos[0]
    tasks = (build_summary or {}).get("tasks") or sorted({t for i in plan.infos for t in i["tasks"]})
    if plan.task:
        tasks = [plan.task]
    return {"policy": "smolvla", "job": plan.name, "datasets": [i["name"] for i in plan.infos],
            "dataset": first["name"], "cameras": sorted(first["cameras"]), "image_sizes": first["cameras"],
            "task": tasks[0] if tasks else None, "tasks": tasks, "fps": first["fps"],
            "units": {"joints": "deg", "gripper": "mm"}, "role": plan.role,
            "parent": Path(plan.parent).name if plan.parent else None, "base": plan.base,
            "contract": CONTRACT_VERSION, "profile": plan.profile, "batch_size": plan.batch_size,
            "target_steps": plan.steps, "build": build_summary and {k: build_summary[k] for k in ("episodes", "frames", "fps")},
            **facts}


def supervisor_config(plan: Plan, out_dir: str, args: list[str], lerobot_train: str, publish: bool,
                      meta: dict) -> dict:
    every = plan.publish_every or plan.steps
    return {"output_dir": out_dir, "steps": plan.steps, "args": args, "lerobot_train": lerobot_train,
            "required_extra": REQUIRED_EXTRA, "poll_s": 15, "hang_timeout_s": 1200, "backoff_base_s": 30,
            "max_restarts": 12, "min_free_gb": 8,
            "publish_dir": plan.publish_dir if publish else None, "publish_prefix": plan.prefix,
            "publish_min_step": every if plan.publish_every else plan.steps, "publish_every": every,
            "publish_meta": meta}


def describe(plan: Plan) -> str:
    lines = [f"run        {plan.name}  ({'remote ' + plan.remote if plan.remote else 'this machine'}, profile {plan.profile})",
             "datasets   " + ", ".join(f"{i['name']} ({i['episodes']} episodes, {i['frames']} frames)" for i in plan.infos),
             "cameras    " + ", ".join(f"{k} {v[1]}x{v[0]}" for k, v in plan.infos[0]["cameras"].items()),
             "task       " + (repr(plan.task) + " (override)" if plan.task else
                               "; ".join(repr(t) for t in sorted({t for i in plan.infos for t in i["tasks"]}))),
             "start from " + (f"{plan.parent} (post-training)" if plan.parent else plan.base),
             f"training   {plan.steps} steps, batch {plan.batch_size}, device {plan.device}, "
             f"checkpoint every {plan.save_freq}",
             f"publish    {plan.publish_dir}/{plan.prefix}_<step>",
             f"work dir   {plan.run_dir}"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ guards
def busy_trainers() -> list[str]:
    """Other training runs on this machine (lerobot-train or a supervisor)."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    me = str(os.getpid())
    hits = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid != me and ("lerobot-train" in cmd or "lerobot_train" in cmd or "train_supervisor.py" in cmd
                          or "piper_smolvla.supervisor" in cmd):
            hits.append(line.strip()[:160])
    return hits


# Adapted from izhangfly/NervusOS-Cambridge/act/mac/train_supervisor.py (Supervisor.mem_pressure).
def mem_free_pct() -> int | None:
    try:
        out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def guard_local(force: bool):
    """The Mac has 16 GB shared with everything else: never train next to another heavy job."""
    if force:
        return
    others = busy_trainers()
    if others:
        raise RuntimeError("another training run is active on this machine; wait for it or pass --force:\n  "
                           + "\n  ".join(others))
    free = mem_free_pct()
    if free is not None and free < 20:
        raise RuntimeError(f"only {free}% of memory is free: close large apps (a local LLM is enough) or pass --force")


# ------------------------------------------------------------------------------------------------ data
def prepare_data(plan: Plan, rebuild: bool = False) -> tuple[Path, dict | None]:
    """The dataset this run trains on, built once per run (a rerun reuses it unless --rebuild)."""
    if not plan.build:
        return Path(plan.datasets[0]), None
    dest = plan.run_dir / "data"
    rep_path = dest / "BUILD_REPORT.json"
    if rebuild or not rep_path.is_file():
        print(f"building the training set in {dest} (edits applied, idle trimmed, h264 CRF 18)", flush=True)
        D.build([Path(d) for d in plan.datasets], dest, f"local/{plan.name}", log=lambda m: print(m, flush=True),
                task_override=plan.task, auto_trim=plan.auto_trim)
    else:
        print(f"reusing the training set in {dest} (--rebuild to build it again)")
    return dest, read_json(rep_path)


def _facts(plan: Plan, data_dir: Path) -> dict:
    p = plan.run_dir / "facts.json"
    f = D.facts(data_dir)
    write_json(p, f)
    return f


# ------------------------------------------------------------------------------------------------ local
def supervisor_pid(run_dir: Path) -> int | None:
    try:
        pid = int((run_dir / "supervisor.pid").read_text())
        os.kill(pid, 0)
    except Exception:
        return None
    try:
        cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    except Exception:
        cmd = ""
    return pid if "piper_smolvla.supervisor" in cmd else None


def train_local(plan: Plan, detach: bool = False, rebuild: bool = False, force: bool = False,
                print_only: bool = False) -> int:
    from . import base as B
    run = plan.run_dir
    print(describe(plan))
    if print_only:
        args = lerobot_args(plan, str(run / "data") if plan.build else plan.datasets[0],
                            plan.parent or str(run / "base"))
        print("\nlerobot-train " + " ".join(shlex.quote(a) for a in args + [f"--steps={plan.steps}",
                                                                              f"--output_dir={run / 'out'}"]))
        return 0
    pid = supervisor_pid(run)
    if pid:
        print(f"this run is already training (supervisor pid {pid}); `piper-smolvla status {plan.name}` follows it")
        return 0
    guard_local(force)
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "plan.json", asdict(plan))
    data_dir, summary = prepare_data(plan, rebuild)
    facts = _facts(plan, data_dir)
    policy_dir = plan.parent or str(B.prepare(plan.base, run / "base"))
    meta = publish_meta(plan, facts, summary)
    lt = str(Path(sys.executable).parent / "lerobot-train")
    cfg = supervisor_config(plan, str(run / "out"), lerobot_args(plan, str(data_dir), policy_dir), lt, True, meta)
    write_json(run / "supervisor.json", cfg)
    with open(run / "supervisor.stdout", "a") as so:
        proc = subprocess.Popen([sys.executable, "-m", "piper_smolvla.supervisor", "--config", str(run / "supervisor.json")],
                                stdout=so, stderr=subprocess.STDOUT, start_new_session=True,
                                env=dict(os.environ, PYTHONUNBUFFERED="1"))
    (run / "supervisor.pid").write_text(str(proc.pid))
    if os.path.exists("/usr/bin/caffeinate"):
        subprocess.Popen(["/usr/bin/caffeinate", "-dimsu", "-w", str(proc.pid)], start_new_session=True)
    print(f"\ntraining under the supervisor (pid {proc.pid}); log {run / 'out.log'}")
    if detach:
        print(f"detached: `piper-smolvla status {plan.name}` shows progress, `piper-smolvla stop {plan.name}` stops it")
        return 0
    return follow_local(plan.name, Path(plan.work), proc)


def follow_local(name: str, work: Path, proc=None) -> int:
    run = work / "runs" / name
    pid = proc.pid if proc else supervisor_pid(run)
    alive = (lambda: proc.poll() is None) if proc else (lambda: supervisor_pid(run) is not None)
    stopping = {"asked": False}

    def ask_stop(*_):
        if not stopping["asked"] and pid:
            stopping["asked"] = True
            print("\nstopping: the supervisor ends the trainer cleanly; run the same command again to resume",
                  flush=True)
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    old = signal.signal(signal.SIGINT, ask_stop)
    try:
        last = None
        while alive():
            st = read_json(run / "out.status.json", {}) or {}
            line = status_line(st)
            if line != last:
                print(line, flush=True)
                last = line
            time.sleep(5)
    finally:
        signal.signal(signal.SIGINT, old)
    st = read_json(run / "out.status.json", {}) or {}
    print(status_line(st))
    if st.get("published"):
        print("published: " + ", ".join(st["published"]))
    return 0 if st.get("state") in ("finished", "stopped_by_user") else 1


def status_line(st: dict) -> str:
    if not st:
        return "starting (no status yet)"
    m = st.get("metrics") or {}
    parts = [st.get("state", "?"), f"step {st.get('step') or 0}/{st.get('target')}"]
    if m.get("loss"):
        parts.append(f"loss {m['loss']}")
    if st.get("step_time_s"):
        parts.append(f"{st['step_time_s']} s/step")
    if st.get("eta_hours") is not None:
        parts.append(f"eta {st['eta_hours']} h")
    if st.get("restarts"):
        parts.append(f"restarts {st['restarts']}")
    if st.get("checkpoints"):
        parts.append(f"checkpoints {','.join(st['checkpoints'][-3:])}")
    return " | ".join(parts)


def stop_local(name: str, work: Path = WORK) -> int:
    pid = supervisor_pid(work / "runs" / name)
    if not pid:
        print(f"{name} is not training on this machine")
        return 1
    os.kill(pid, signal.SIGTERM)
    print(f"asked the supervisor (pid {pid}) to stop; it ends the trainer cleanly. Resume with the same train command.")
    return 0


# ------------------------------------------------------------------------------------------------ remote
def package_dir() -> Path:
    return Path(__file__).resolve().parent


def remote_paths(root: str, name: str) -> dict:
    run = f"{root}/runs/{name}"
    return {"root": root, "run": run, "data": f"{run}/data", "out": f"{run}/out", "base": f"{run}/base",
            "parent": f"{run}/parent", "src": f"{root}/src", "venv": f"{root}/venv",
            "lerobot_train": f"{root}/venv/bin/lerobot-train", "session": SESSION_PREFIX + name}


def train_remote(plan: Plan, rebuild: bool = False, follow: bool = False, print_only: bool = False) -> int:
    rem = Remote(plan.remote, port=plan.ssh_port, dry=print_only)
    print(describe(plan))
    run = plan.run_dir
    if print_only:
        home = "$HOME"
    else:
        probe = rem.ssh('echo "$HOME"; uname -sm; command -v tmux || echo no-tmux; '
                        'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo no-gpu')
        lines = probe.strip().splitlines()
        home = lines[0].strip()
        print(f"remote     {plan.remote}: {' '.join(lines[1:2])}; GPU {'; '.join(lines[3:]) or '?'}")
        if any("no-gpu" in x for x in lines[3:]):
            raise RuntimeError(f"{plan.remote} has no working nvidia-smi: the cuda profile needs an NVIDIA GPU")
    root = plan.remote_root or f"{home}/piper-smolvla"
    rp = remote_paths(root, plan.name)
    policy_dir = rp["parent"] if plan.parent else rp["base"]
    args = lerobot_args(plan, rp["data"], policy_dir)
    if print_only:
        print("\non the GPU box: lerobot-train " + " ".join(shlex.quote(a) for a in args + [
            f"--steps={plan.steps}", f"--output_dir={rp['out']}"]))
        print("commands that would run:")
        data_dir = (run / "data") if plan.build else Path(plan.datasets[0])
    else:
        run.mkdir(parents=True, exist_ok=True)
        write_json(run / "plan.json", asdict(plan))
        data_dir, summary = prepare_data(plan, rebuild)
        meta = publish_meta(plan, _facts(plan, data_dir), summary)
        write_json(run / "publish_meta.json", meta)
        write_json(run / "remote" / "supervisor.json",
                   supervisor_config(plan, rp["out"], args, rp["lerobot_train"], False, meta))
        if plan.base:
            (run / "remote" / "base_ref.txt").write_text(plan.base + "\n")
        print(f"\nsending the run to {plan.remote}:{rp['run']}")
    rem.ssh(f"mkdir -p {shlex.quote(rp['run'])} {shlex.quote(rp['src'])}")
    rem.rsync(str(package_dir()) + "/", f":{rp['src']}/piper_smolvla/", excludes=("__pycache__", "*.pyc"), delete=True)
    rem.rsync(str(run / "remote") + "/", f":{rp['run']}/")
    rem.rsync(str(data_dir) + "/", f":{rp['data']}/", delete=True)
    if plan.parent:
        rem.rsync(str(plan.parent) + "/", f":{rp['parent']}/", copy_links=True)
    out = rem.ssh(f"bash {shlex.quote(rp['src'] + '/piper_smolvla/env/remote_launch.sh')} "
                  f"{shlex.quote(root)} {shlex.quote(plan.name)}")
    if print_only:
        return 0
    write_json(run / "remote.json", {"host": plan.remote, "port": plan.ssh_port, "root": root, "run": rp["run"],
                                     "session": rp["session"], "started": utc_now()})
    if out:
        print(out.strip())
    print(f"training on {plan.remote} in tmux session {rp['session']} "
          f"(ssh {plan.remote} -t tmux attach -t {rp['session']} to watch it)")
    print(f"  piper-smolvla status {plan.name}     progress")
    print(f"  piper-smolvla pull {plan.name}       copy checkpoints back and publish them")
    print(f"  piper-smolvla stop {plan.name}       stop it (the same train command resumes it)")
    if follow:
        return follow_remote(plan.name, Path(plan.work))
    return 0


def _remote_for(name: str, work: Path) -> tuple[Remote, dict]:
    info = read_json(work / "runs" / name / "remote.json")
    if not info:
        raise FileNotFoundError(f"{name} has not been sent to a GPU box from this machine (no remote.json)")
    return Remote(info["host"], port=info.get("port")), info


def remote_status(name: str, work: Path = WORK) -> dict:
    rem, info = _remote_for(name, work)
    q = shlex.quote
    raw = rem.ssh(f"cat {q(info['run'] + '/out.status.json')} 2>/dev/null || echo '{{}}'; echo; "
                  f"if [ -f {q(info['run'] + '/supervisor.pid')} ] && kill -0 $(cat {q(info['run'] + '/supervisor.pid')}) "
                  f"2>/dev/null; then echo RUNNING; else echo NOT-RUNNING; fi")
    body, _, flag = raw.strip().rpartition("\n")
    try:
        st = json.loads(body or "{}")
    except ValueError:
        st = {}
    st["supervisor_running"] = flag.strip() == "RUNNING"
    return st


def pull(name: str, work: Path = WORK, with_state: bool = False, publish: bool = True) -> list[str]:
    """Copy the run's status, logs and checkpoints back from the GPU box, then publish what is complete.
    training_state/ (optimizer moments, the larger half) stays on the box unless with_state."""
    rem, info = _remote_for(name, work)
    run = work / "runs" / name
    (run / "remote").mkdir(parents=True, exist_ok=True)
    for f in ("out.status.json", "out.supervisor.log", "remote.log", "out.log"):
        try:
            rem.rsync(f":{info['run']}/{f}", str(run / "remote" / f))
        except RuntimeError:
            pass
    excludes = () if with_state else ("training_state",)
    try:
        rem.rsync(f":{info['run']}/out/checkpoints/", str(run / "out" / "checkpoints") + "/", excludes=excludes)
    except RuntimeError as e:
        if "No such file" not in str(e):
            raise
        return []
    return publish_run(name, work) if publish else []


def publish_run(name: str, work: Path = WORK) -> list[str]:
    run = work / "runs" / name
    plan = read_json(run / "plan.json", {}) or {}
    meta = read_json(run / "publish_meta.json") or (read_json(run / "supervisor.json", {}) or {}).get("publish_meta")
    status = read_json(run / "remote" / "out.status.json", {}) or read_json(run / "out.status.json", {}) or {}
    target = int(plan.get("steps") or 0)
    every = plan.get("publish_every") or target
    dest_root = Path(plan.get("publish_dir") or SHARED / "smolvla" / "policies")
    out = []
    ck_root = run / "out" / "checkpoints"
    for p in sorted(ck_root.glob("[0-9]*")) if ck_root.is_dir() else []:
        if p.is_symlink() or not pretrained_ok(p / "pretrained_model"):
            continue
        step = int(p.name)
        if not (step == target or (plan.get("publish_every") and step % every == 0)):
            continue
        pname = f"{name}-smolvla_{p.name}"
        m = dict(meta or {}, step=step, loss=(status.get("metrics") or {}).get("loss"))
        if publish_pretrained(p / "pretrained_model", dest_root, pname, m):
            print(f"published {dest_root / pname}")
        out.append(pname)
    return out


def follow_remote(name: str, work: Path = WORK, every_s: int = 120) -> int:
    print(f"following {name} on the GPU box; Ctrl+C stops following (training carries on)")
    try:
        while True:
            st = remote_status(name, work)
            print(status_line(st), flush=True)
            if not st.get("supervisor_running"):
                published = pull(name, work)
                print("finished" if st.get("state") == "finished" else f"the supervisor is not running ({st.get('state')})")
                if published:
                    print("published: " + ", ".join(published))
                return 0 if st.get("state") == "finished" else 1
            time.sleep(every_s)
    except KeyboardInterrupt:
        print("\nstopped following; the run continues on the GPU box")
        return 0


def stop_remote(name: str, work: Path = WORK) -> int:
    rem, info = _remote_for(name, work)
    pidf = shlex.quote(info["run"] + "/supervisor.pid")
    out = rem.ssh(f"if [ -f {pidf} ] && kill -0 $(cat {pidf}) 2>/dev/null; then kill -TERM $(cat {pidf}) && echo STOPPING; "
                  f"else echo NOT-RUNNING; fi")
    print("asked the supervisor to stop; resume with the same train command" if "STOPPING" in out else
          f"{name} is not training on {info['host']}")
    return 0
