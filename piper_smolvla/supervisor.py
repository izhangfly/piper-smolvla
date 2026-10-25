#!/usr/bin/env python3
# Adapted from izhangfly/NervusOS-Cambridge/act/mac/train_supervisor.py.
"""Unattended supervisor for a LeRobot `lerobot-train` run, on the Mac or on a rented GPU box.

Copied from the NervusOS ACT training worker (act/mac/train_supervisor.py, same author), where it ran the
Piper-ACT trainings (100k steps, a kill -9 crash drill included) without a restart. It starts the run, watches
it, and recovers from crashes, hangs and NaN losses by resuming from the last *complete* checkpoint. Every
launch resumes if a complete checkpoint exists, which is what makes `piper-smolvla train` restartable: run it
again with the same --name and it carries on. It uses the standard library only, so the GPU box needs nothing
but the training venv.

Changes from the ACT copy: a checkpoint is complete only when the files named in the config's
`required_extra` also exist (SmolVLA has a learning-rate scheduler, and lerobot 0.4.4 writes its state last),
publishing goes through publish_pretrained() (shared with `piper-smolvla pull`), and the ACT-specific finish
note is gone.

    python -m piper_smolvla.supervisor --config supervisor.json            # foreground
    python -m piper_smolvla.supervisor --config supervisor.json --daemon   # detach

Files it writes, all inside the run's output_dir parent (never inside the dataset):
    <run>.log                 raw lerobot-train stdout/stderr, appended across restarts
    <run>.supervisor.log      what the supervisor decided and why
    <run>.status.json         machine-readable snapshot, rewritten every poll
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# The config may name the lerobot-train executable (the worker passes the venv it runs in);
# otherwise it sits next to the Python running this file.
LEROBOT_TRAIN = Path(sys.executable).parent / "lerobot-train"

# Files that must exist for a checkpoint to be resumable (verified against a real 0.4.4
# checkpoint: pretrained_model/ + training_state/). A config can add more with "required_extra".
REQUIRED = [
    "pretrained_model/config.json",
    "pretrained_model/model.safetensors",
    "pretrained_model/train_config.json",
    "training_state/optimizer_state.safetensors",
    "training_state/optimizer_param_groups.json",
    "training_state/rng_state.safetensors",
    "training_state/training_step.json",
]

# Failure signatures -> remedy applied on the NEXT (resumed) launch. Order matters: first match wins.
SIGNATURES = [
    ("bad_ckpt", re.compile(r"SafetensorError|incomplete metadata|HeaderTooLarge|FileNotFoundError.*(training_state|pretrained_model)", re.I)),
    ("nan_loss", re.compile(r"loss:nan|loss:inf", re.I)),
    ("mps_oom", re.compile(r"MPS backend out of memory|out of memory", re.I)),
    ("mps_unimpl", re.compile(r"not currently implemented for the MPS device", re.I)),
    ("video_decode", re.compile(r"Could not load libtorchcodec|torchcodec\S*Error|decod(e|ing)\S* .*error|violate the tolerance|FrameTimestampError", re.I)),
    ("worker_died", re.compile(r"DataLoader worker .* (exited|killed)|worker.*unexpectedly|BrokenPipeError|received signal", re.I)),
    ("hub_401", re.compile(r"401 Client Error|RepositoryNotFoundError", re.I)),
    ("output_exists", re.compile(r"already exists and resume is False", re.I)),
    ("bad_arg", re.compile(r"unrecognized arguments|error: argument|DecodingError|ParsingError", re.I)),
]
FATAL = {"hub_401", "bad_arg", "output_exists"}  # restarting cannot fix these

TQDM = re.compile(r"(\d+)/(\d+) \[")
TQDM_RATE = re.compile(r"Training:[^\r\n]*?\d+/\d+ \[[^\]]*?([\d.]+)(s/step|step/s)\]")
LOSS = re.compile(r"step:(\S+) .*?loss:([0-9.]+|nan|inf) grdn:([0-9.]+|nan|inf) lr:(\S+) updt_s:([0-9.]+) data_s:([0-9.]+)")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def safetensors_ok(f):
    """A truncated safetensors file (kill mid-save) is non-empty but unloadable: check header vs size."""
    try:
        size = f.stat().st_size
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            if size < 8 or 8 + n > size:
                return False
            hdr = json.loads(fh.read(n))
        end = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
        return 8 + n + end == size
    except Exception:
        return False


def pretrained_ok(pm: Path) -> bool:
    """A pretrained_model/ directory LeRobot can load: config, weights and every processor state file whole."""
    try:
        json.loads((pm / "config.json").read_text())
        if not safetensors_ok(pm / "model.safetensors"):
            return False
        for pj in ("policy_preprocessor.json", "policy_postprocessor.json"):
            for st in json.loads((pm / pj).read_text()).get("steps", []):
                if st.get("state_file") and not safetensors_ok(pm / st["state_file"]):
                    return False
        return True
    except Exception:
        return False


def publish_pretrained(src: Path, dest_root: Path, name: str, meta: dict | None) -> bool:
    """Copy one checkpoint's pretrained_model/ to dest_root/name, with nervus_policy.json (the facts the robot
    side needs, and where the policy came from) written before an atomic rename, so a synced folder never shows
    a half-copied policy under its real name. False if it was already there."""
    dest = Path(dest_root) / name
    if (dest / "model.safetensors").is_file():
        return False
    tmp = Path(dest_root) / f".{name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, tmp)
    if meta is not None:
        meta = dict(meta, created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        (tmp / "nervus_policy.json").write_text(json.dumps(meta, indent=2))
    tmp.rename(dest)
    return True


class Supervisor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out = Path(cfg["output_dir"]).expanduser()
        if not self.out.is_absolute():
            self.out = Path.cwd() / self.out
        self.name = self.out.name
        self.logdir = self.out.parent
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.train_log = self.logdir / f"{self.name}.log"
        self.sup_log = self.logdir / f"{self.name}.supervisor.log"
        self.status_path = self.logdir / f"{self.name}.status.json"
        self.target = int(cfg["steps"])
        self.remedies = {"env": {}, "args": {}}
        self.restarts = 0
        self.history = []  # (time, reason, step)
        self.proc = None
        self.stop_requested = False
        self.last_metrics = {}

    # ---------- logging ----------
    def say(self, msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with open(self.sup_log, "a") as f:
            f.write(line + "\n")

    # ---------- checkpoints ----------
    def complete_ckpts(self):
        d = self.out / "checkpoints"
        if not d.is_dir():
            return []
        good = []
        for p in sorted(d.iterdir()):
            if p.is_symlink() or not p.is_dir() or not p.name.isdigit():
                continue
            if self.ckpt_valid(p):
                good.append(p)
        return good

    safetensors_ok = staticmethod(safetensors_ok)

    def ckpt_valid(self, p):
        try:
            for r in REQUIRED + list(self.cfg.get("required_extra", [])):
                f = p / r
                if not f.is_file():
                    return False
                if f.suffix == ".json":
                    json.loads(f.read_text())
                elif not self.safetensors_ok(f):
                    return False
            pm = p / "pretrained_model"
            for pj in ("policy_preprocessor.json", "policy_postprocessor.json"):
                for st in json.loads((pm / pj).read_text()).get("steps", []):
                    if st.get("state_file") and not self.safetensors_ok(pm / st["state_file"]):
                        return False
            # optimizer_param_groups.json is the LAST file lerobot 0.4.4 writes without a scheduler, and
            # scheduler_state.json the last with one (SmolVLA): the config lists it in required_extra.
            return json.loads((p / "training_state/training_step.json").read_text())["step"] == int(p.name)
        except Exception:
            return False

    def ckpt_step(self, p):
        try:
            return int(json.loads((p / "training_state/training_step.json").read_text())["step"])
        except Exception:
            return int(p.name)

    def repoint_last(self, p):
        """Make checkpoints/last point at a complete checkpoint (a kill mid-save can leave it wrong)."""
        last = self.out / "checkpoints" / "last"
        try:
            if last.is_symlink() and os.readlink(last) == p.name:
                return
            tmp = last.with_name(".last.tmp")
            if tmp.is_symlink() or tmp.exists():
                tmp.unlink()
            tmp.symlink_to(p.name)
            os.replace(tmp, last)  # atomic, unlike lerobot's own unlink()+symlink_to()
            self.say(f"repointed checkpoints/last -> {p.name}")
        except OSError as e:
            self.say(f"WARN could not repoint last: {e}")

    def quarantine_incomplete(self):
        d = self.out / "checkpoints"
        if not d.is_dir():
            return
        good = {p.name for p in self.complete_ckpts()}
        for p in d.iterdir():
            if p.is_dir() and not p.is_symlink() and p.name.isdigit() and p.name not in good:
                dest = self.logdir / f"{self.name}.incomplete_ckpt_{p.name}_{int(time.time())}"
                shutil.move(str(p), str(dest))
                self.say(f"quarantined incomplete checkpoint {p.name} -> {dest.name}")

    # ---------- disk ----------
    def disk_guard(self):
        free_gb = shutil.disk_usage(self.logdir).free / 1e9
        floor = float(self.cfg.get("min_free_gb", 6))
        if free_gb >= floor:
            return free_gb
        # Only the newest training_state is needed to resume; pretrained_model/ is what gets evaluated.
        ck = self.complete_ckpts()
        for p in ck[:-1]:
            ts = p / "training_state"
            if ts.is_dir():
                shutil.rmtree(ts)
                self.say(f"disk low ({free_gb:.1f} GB): removed {p.name}/training_state (model kept)")
                free_gb = shutil.disk_usage(self.logdir).free / 1e9
                if free_gb >= floor:
                    break
        return free_gb

    # ---------- policies hand-back ----------
    def publish(self):
        dest_root = self.cfg.get("publish_dir")
        if not dest_root:
            return
        min_step = int(self.cfg.get("publish_min_step", 0))
        for p in self.complete_ckpts():
            s = self.ckpt_step(p)
            # The final checkpoint always goes out (when past min_step): it is often the one that works.
            if s < min_step or (s % int(self.cfg.get("publish_every", 1)) != 0 and s != self.target):
                continue
            name = f"{self.cfg['publish_prefix']}_{p.name}"
            meta = self.cfg.get("publish_meta")
            if meta is not None:
                meta = dict(meta, step=s, loss=self.last_metrics.get("loss"))
            if publish_pretrained(p / "pretrained_model", Path(dest_root).expanduser(), name, meta):
                self.say(f"published {p.name}/pretrained_model -> {dest_root}/{name}")
            if name not in getattr(self, "published", []):
                self.published = getattr(self, "published", []) + [name]

    # ---------- launch ----------
    def trainer(self) -> str:
        return str(Path(self.cfg.get("lerobot_train") or LEROBOT_TRAIN).expanduser())

    def build_cmd(self):
        ck = self.complete_ckpts()
        if ck:
            last = ck[-1]
            self.repoint_last(last)
            cmd = [self.trainer(),
                   f"--config_path={last}/pretrained_model/train_config.json",
                   "--resume=true", f"--steps={self.target}"]
            mode = f"resume from {last.name} (step {self.ckpt_step(last)})"
        else:
            if self.out.exists():
                # Crashed before the first checkpoint: nothing to resume, and lerobot refuses an
                # existing output_dir without resume. Keep the evidence, start clean.
                dest = self.logdir / f"{self.name}.precheckpoint_crash_{int(time.time())}"
                shutil.move(str(self.out), str(dest))
                self.say(f"no complete checkpoint; moved {self.out.name} aside -> {dest.name}")
            cmd = [self.trainer()] + list(self.cfg["args"]) + [
                f"--steps={self.target}", f"--output_dir={self.out}"]
            mode = "fresh start"
        for k, v in self.remedies["args"].items():
            cmd = [c for c in cmd if not c.startswith(f"--{k}=")] + [f"--{k}={v}"]
        return cmd, mode

    def launch(self):
        cmd, mode = self.build_cmd()
        env = os.environ.copy()
        env.update({k: str(v) for k, v in self.cfg.get("env", {}).items()})
        env.update(self.remedies["env"])
        env["PYTHONUNBUFFERED"] = "1"
        self.say(f"LAUNCH #{self.restarts} ({mode}) remedies={self.remedies}")
        self.say("  " + " ".join(cmd))
        self.log_offset = self.train_log.stat().st_size if self.train_log.exists() else 0
        with open(self.train_log, "a") as f:
            f.write(f"\n===== supervisor launch #{self.restarts} {now()} — {mode} =====\n")
        self.logf = open(self.train_log, "a")
        self.proc = subprocess.Popen(cmd, cwd=self.logdir, env=env, stdout=self.logf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.launch_time = time.time()
        self.last_growth = time.time()
        self.last_size = self.train_log.stat().st_size
        self.base_step = self.ckpt_step(self.complete_ckpts()[-1]) if "resume" in mode else 0

    def kill(self, why):
        if not self.proc or self.proc.poll() is not None:
            return
        self.say(f"stopping trainer pid {self.proc.pid}: {why}")
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
            self.proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=30)
            except Exception:
                pass

    # ---------- observe ----------
    def tail(self, nbytes=200_000):
        try:
            with open(self.train_log, "rb") as f:
                size = f.seek(0, 2)
                start = max(self.log_offset, size - nbytes)
                f.seek(start)
                return f.read().decode("utf-8", "replace")
        except FileNotFoundError:
            return ""

    def progress(self, text):
        step = None
        m = TQDM.findall(text)
        if m:
            step = self.base_step + int(m[-1][0])
        r = TQDM_RATE.findall(text)
        if r:
            v, unit = float(r[-1][0]), r[-1][1]
            self.step_time = v if unit == "s/step" else (1.0 / v if v > 0 else None)
        lm = LOSS.findall(text)
        if lm:
            s, loss, grdn, lr, updt, data = lm[-1]
            self.last_metrics = {"loss": loss, "grdn": grdn, "lr": lr, "updt_s": float(updt), "data_s": float(data)}
        return step

    def classify(self, text):
        # Classify from the traceback only. The startup config dump contains words like
        # 'torchcodec' and would otherwise be misread as a decode failure.
        i = text.rfind("Traceback (most recent call last)")
        tailtext = text[i:] if i >= 0 else text[-3000:]
        for name, rx in SIGNATURES:
            if rx.search(tailtext):
                return name
        return "unknown" if i >= 0 else "killed_no_traceback"

    @staticmethod
    def swap_used_gb():
        try:
            out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"used = ([\d.]+)M", out)
            return round(float(m.group(1)) / 1024, 1) if m else None
        except Exception:
            return None

    def mem_pressure(self):
        try:
            out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=10).stdout
            m = re.search(r"free percentage:\s*(\d+)%", out)
            return int(m.group(1)) if m else None
        except Exception:
            return None

    def write_status(self, state, step=None, extra=None):
        eta_h = None
        if step and self.last_metrics.get("updt_s"):
            per = self.last_metrics["updt_s"] + self.last_metrics["data_s"]
            eta_h = round((self.target - step) * per / 3600, 2)
        st = {
            "time": now(), "state": state, "run": self.name, "step": step, "target": self.target,
            "pct": round(100 * step / self.target, 1) if step else None, "eta_hours": eta_h,
            "metrics": self.last_metrics, "restarts": self.restarts, "history": self.history[-20:],
            "remedies": self.remedies, "pid": self.proc.pid if self.proc else None,
            "free_gb": round(shutil.disk_usage(self.logdir).free / 1e9, 1),
            "mem_free_pct": self.mem_pressure(),
            "swap_used_gb": self.swap_used_gb(),
            "step_time_s": round(self.step_time, 3) if getattr(self, "step_time", None) else None,
            "checkpoints": [p.name for p in self.complete_ckpts()],
            "published": getattr(self, "published", []),
        }
        if extra:
            st.update(extra)
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2))
        tmp.replace(self.status_path)

    # ---------- remedy policy ----------
    def remedy(self, reason):
        n = sum(1 for _, r, _ in self.history if r == reason)
        if reason == "bad_ckpt":
            ck = self.complete_ckpts()
            if ck:
                dest = self.logdir / f"{self.name}.bad_ckpt_{ck[-1].name}_{int(time.time())}"
                shutil.move(str(ck[-1]), str(dest))
                self.say(f"resume failed loading {ck[-1].name}: set aside -> {dest.name}")
        elif reason == "mps_unimpl":
            self.remedies["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        elif reason == "video_decode" and n >= 1:
            self.remedies["args"]["dataset.video_backend"] = "pyav"
        elif reason == "worker_died":
            self.remedies["args"]["num_workers"] = 2 if n <= 1 else 0
        elif reason == "mps_oom":
            # 1st: a fresh process usually clears allocator fragmentation. 2nd: fewer workers (less
            # unified-memory pressure). Batch size is NOT changed automatically — it alters the recipe.
            if n >= 1:
                self.remedies["args"]["num_workers"] = 2
        elif reason == "hang" and n >= 1:
            self.remedies["args"]["num_workers"] = 0
        elif reason == "nan_loss" and n >= 1:
            # Resuming the same checkpoint twice into NaN: fall back one checkpoint.
            ck = self.complete_ckpts()
            if len(ck) >= 2:
                dest = self.logdir / f"{self.name}.nan_ckpt_{ck[-1].name}_{int(time.time())}"
                shutil.move(str(ck[-1]), str(dest))
                self.say(f"NaN twice: set aside {ck[-1].name}, resuming from {ck[-2].name}")

    # ---------- main loop ----------
    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop_requested", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop_requested", True))
        max_restarts = int(self.cfg.get("max_restarts", 12))
        hang_s = int(self.cfg.get("hang_timeout_s", 900))
        poll = int(self.cfg.get("poll_s", 30))
        self.say(f"supervisor start pid={os.getpid()} target={self.target} out={self.out}")
        self.quarantine_incomplete()
        self.launch()
        step = None
        while True:
            time.sleep(poll)
            if self.stop_requested:
                self.kill("supervisor asked to stop")
                self.write_status("stopped_by_user", step)
                self.say("stopped by request")
                return 0
            text = self.tail()
            step = self.progress(text) or step
            size = self.train_log.stat().st_size
            if size != self.last_size:
                self.last_size, self.last_growth = size, time.time()
            free = self.disk_guard()
            try:
                self.publish()
            except Exception as e:
                self.say(f"WARN publish failed: {e}")

            rc = self.proc.poll()
            reason = None
            if rc is None:
                if re.search(r"loss:(nan|inf)", text[-5000:]):
                    reason = "nan_loss"
                    self.kill("NaN/inf loss")
                elif time.time() - self.last_growth > hang_s:
                    reason = "hang"
                    self.kill(f"log silent for {hang_s}s")
                else:
                    self.write_status("training", step, {"free_gb_guard": round(free, 1)})
                    continue
            else:
                ck = self.complete_ckpts()
                done = ck and self.ckpt_step(ck[-1]) >= self.target
                if rc == 0 and done:
                    self.publish()
                    self.write_status("finished", self.ckpt_step(ck[-1]))
                    self.say(f"FINISHED: step {self.ckpt_step(ck[-1])}, checkpoints {[p.name for p in ck]}")
                    return 0
                reason = self.classify(self.tail())
                self.say(f"trainer exited rc={rc} reason={reason}")

            self.history.append((now(), reason, step))
            self.restarts += 1
            if reason in FATAL:
                self.write_status(f"fatal:{reason}", step)
                self.say(f"FATAL {reason}: restarting cannot fix this — needs a human. Last output:\n{self.tail()[-3000:]}")
                return 2
            if self.restarts > max_restarts:
                self.write_status("gave_up", step)
                self.say(f"GAVE UP after {self.restarts - 1} restarts. History: {self.history}")
                return 3
            recent = [h for h in self.history if h[1] == reason]
            if len(recent) >= 4:
                self.write_status(f"gave_up:{reason}", step)
                self.say(f"GAVE UP: '{reason}' happened {len(recent)} times; not a transient.")
                return 3
            self.remedy(reason)
            self.quarantine_incomplete()
            backoff = min(300, int(self.cfg.get("backoff_base_s", 30)) * self.restarts)
            self.write_status(f"restarting:{reason}", step)
            self.say(f"restart #{self.restarts} in {backoff}s after '{reason}'")
            time.sleep(backoff)
            self.launch()


def daemonize(logfile):
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    sys.stdout.flush()
    fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--daemon", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    sup = Supervisor(cfg)
    if a.daemon:
        daemonize(str(sup.logdir / f"{sup.name}.supervisor.stdout"))
        if os.path.exists("/usr/bin/caffeinate"):
            # Keep a Mac awake for exactly as long as the supervisor lives.
            subprocess.Popen(["/usr/bin/caffeinate", "-dimsu", "-w", str(os.getpid())], start_new_session=True)
        (sup.logdir / f"{sup.name}.supervisor.pid").write_text(str(os.getpid()))
    sys.exit(sup.run())


if __name__ == "__main__":
    main()
