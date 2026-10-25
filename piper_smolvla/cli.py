"""piper-smolvla: fine-tune SmolVLA on PiPER demonstrations, serve it, and run it on the arm.

    piper-smolvla datasets                                   what's recorded, with each dataset's task text
    piper-smolvla train DATASET... [--remote user@host]      fine-tune lerobot/smolvla_base (Mac or GPU box)
    piper-smolvla posttrain --parent POLICY DATASET...       fine-tune an earlier policy further (corrections)
    piper-smolvla status|pull|stop|publish NAME              follow, fetch, stop or re-publish a run
    piper-smolvla serve [--tunnel ROBOT_HOST]                the policy server (on the Mac)
    piper-smolvla run --policy NAME --task "..." [--dry-run] the robot client (on the arm's host)
    piper-smolvla sim                                        a simulated driver and cameras, to try `run`
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DESCRIPTION = __doc__


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    from .train import PROFILES, SHARED, WORK
    ap = argparse.ArgumentParser(prog="piper-smolvla", description=DESCRIPTION,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"piper-smolvla {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    # ---- datasets ----
    d = sub.add_parser("datasets", help="list datasets and their task text")
    d.add_argument("datasets", nargs="*", help="names in <shared>/datasets or paths (default: all in the shared folder)")
    d.add_argument("--shared", default=str(SHARED))
    d.add_argument("--json", action="store_true")

    # ---- train / posttrain ----
    def train_opts(p, post: bool):
        p.add_argument("datasets", nargs="+", help="LeRobot datasets: names in <shared>/datasets, or paths")
        p.add_argument("--name", help="run name (default: the first dataset's); the same name resumes the run")
        p.add_argument("--profile", choices=sorted(PROFILES), help="mac (MPS, batch 4) or cuda (batch 64); "
                       "default mac, or cuda with --remote")
        p.add_argument("--steps", type=int, help="training steps (default 20000; 5000 when post-training)")
        p.add_argument("--batch-size", type=int)
        p.add_argument("--num-workers", type=int)
        p.add_argument("--save-freq", type=int, help="checkpoint every N steps")
        p.add_argument("--log-freq", type=int)
        p.add_argument("--device", help="override the profile's device (mps, cuda, cpu)")
        if post:
            p.add_argument("--parent", required=True, help="the SmolVLA policy to continue from (a name in "
                           "<shared>/smolvla/policies, or a path)")
        else:
            p.add_argument("--base", default="lerobot/smolvla_base", help="starting weights (Hub id or path)")
            p.add_argument("--parent", help="continue from an earlier SmolVLA policy instead (post-training)")
        p.add_argument("--task", help="train every episode on this task sentence instead of the recorded one")
        p.add_argument("--no-build", action="store_true", help="train on one dataset as it is (no edits, no trim, "
                       "no re-encode)")
        p.add_argument("--no-trim", action="store_true", help="keep the idle stillness at each episode's ends")
        p.add_argument("--no-augment", action="store_true", help="no image augmentation")
        p.add_argument("--rebuild", action="store_true", help="build the training set again")
        p.add_argument("--chunk-size", type=int, help="actions per chunk (default: the base's 50)")
        p.add_argument("--publish-dir", help="where policies are published (default <shared>/smolvla/policies)")
        p.add_argument("--publish-every", type=int, help="also publish every N steps (default: final only)")
        p.add_argument("--work-dir", default=str(WORK), help="runs live in <work-dir>/runs/<name>")
        p.add_argument("--shared", default=str(SHARED), help="the NervusACT shared folder")
        p.add_argument("--remote", metavar="USER@HOST", help="train on this GPU box over ssh (tmux, resumable)")
        p.add_argument("--remote-dir", help="the run's root on the GPU box (default ~/piper-smolvla)")
        p.add_argument("--ssh-port", type=int)
        p.add_argument("--follow", action="store_true", help="with --remote: wait, then pull and publish")
        p.add_argument("--detach", action="store_true", help="local: start and return (status/stop follow it)")
        p.add_argument("--force", action="store_true", help="local: train even if another job or low memory says no")
        p.add_argument("--print-only", action="store_true", help="show the plan and commands; run nothing")
        p.epilog = "Arguments after the options that look like --key=value are passed to lerobot-train."

    train_opts(sub.add_parser("train", help="fine-tune SmolVLA on datasets"), post=False)
    train_opts(sub.add_parser("posttrain", help="fine-tune an earlier SmolVLA policy on new (correction) data"),
               post=True)

    for name, hlp in (("status", "progress of a run"), ("pull", "copy a remote run's checkpoints back and publish"),
                      ("stop", "stop a run (the train command resumes it)"),
                      ("publish", "publish a run's complete checkpoints again")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("name")
        p.add_argument("--work-dir", default=str(WORK))
        if name == "pull":
            p.add_argument("--with-state", action="store_true", help="also copy training_state/ (optimizer)")
            p.add_argument("--no-publish", action="store_true")

    # ---- serve ----
    s = sub.add_parser("serve", help="run the policy server (LeRobot async inference) on this machine")
    s.add_argument("--host", default="127.0.0.1", help="listen address (default loopback; see --tunnel)")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--fps", type=int, default=15, help="the policy's control rate (the datasets' 15)")
    s.add_argument("--tunnel", metavar="ROBOT_HOST", help="ssh -R to the robot host, which then reaches the server "
                   "at its own 127.0.0.1:PORT")
    s.add_argument("--tunnel-port", type=int, help="port on the robot host (default: --port)")
    s.add_argument("--policies-dir", default=str(SHARED / "smolvla" / "policies"),
                   help="where policy names are looked up")

    # ---- run ----
    r = sub.add_parser("run", help="run a policy on the arm (robot client)")
    r.add_argument("--server", default="127.0.0.1:8080", help="the policy server, host:port")
    r.add_argument("--policy", required=True, help="a name in <shared>/smolvla/policies, a path on the server, "
                   "or a Hub id")
    r.add_argument("--task", required=True, help="the instruction, e.g. \"pick up the cube\" (use the "
                   "demonstrations' sentence)")
    r.add_argument("--dry-run", action="store_true", help="check and plan everything, send nothing to the arm")
    r.add_argument("--dry-run-seconds", type=float, default=10.0)
    r.add_argument("--max-duration", type=float, default=90.0, help="a live run stops after this many seconds")
    r.add_argument("--driver", default="http://127.0.0.1:8770", help="the PiPER driver")
    r.add_argument("--engine", default="http://127.0.0.1:8795", help="the ACT engine (camera frames)")
    r.add_argument("--cameras", type=_csv, help="camera roles, e.g. wrist,scene (default: the policy's)")
    r.add_argument("--image-width", type=int)
    r.add_argument("--image-height", type=int)
    r.add_argument("--fps", type=int, help="control rate (default: the training data's)")
    r.add_argument("--policy-device", default="mps", help="where the server runs the model (mps, cuda, cpu)")
    r.add_argument("--actions-per-chunk", type=int)
    r.add_argument("--chunk-threshold", type=float, default=0.5, help="ask for the next chunk when this share "
                   "of the current one is left")
    r.add_argument("--aggregate", default="piper_gripper_latest",
                   help="how overlapping chunks blend (LeRobot's names also work: weighted_average, latest_only)")
    r.add_argument("--goto-start", action="store_true", help="first move slowly to the demonstrations' start pose")
    r.add_argument("--facts", help="nervus_policy.json for the safety limits (default: the policy's)")
    r.add_argument("--speed", type=int, default=40, help="firmware MOVE J speed percent for the run")
    r.add_argument("--gripper-effort", type=int, default=500)
    r.add_argument("--step-limit-x", type=float, default=1.5)
    r.add_argument("--lead-limit-x", type=float, default=4.0)
    r.add_argument("--envelope-margin", type=float, default=5.0)
    r.add_argument("--stagnation-s", type=float, default=8.0, help="stop when the arm stays still this long (0 = off)")
    r.add_argument("--shared", default=str(SHARED))
    r.add_argument("--yes", action="store_true", help="don't ask before moving the arm")

    # ---- sim ----
    m = sub.add_parser("sim", help="a simulated PiPER driver and ACT-engine cameras on loopback")
    m.add_argument("--driver-port", type=int, default=0)
    m.add_argument("--camera-port", type=int, default=0)
    m.add_argument("--cameras", type=_csv, default=["wrist", "scene"])
    m.add_argument("--width", type=int, default=640)
    m.add_argument("--height", type=int, default=480)
    return ap


# ------------------------------------------------------------------------------------------------ commands
def cmd_datasets(a) -> int:
    from . import data as D
    shared = Path(a.shared).expanduser()
    refs = a.datasets or sorted(p.name for p in (shared / "datasets").glob("*") if (p / "meta" / "info.json").is_file())
    infos = []
    for ref in refs:
        try:
            infos.append(D.info(D.resolve(ref, shared)))
        except Exception as e:
            print(f"{ref}: {e}", file=sys.stderr)
    if a.json:
        print(json.dumps(infos, indent=2))
        return 0
    if not infos:
        print(f"no datasets in {shared / 'datasets'}: record some with ACT Studio's Record tab")
        return 0
    for i in infos:
        cams = ", ".join(f"{k} {v[1]}x{v[0]}" for k, v in i["cameras"].items())
        print(f"{i['name']}: {i['episodes']} episodes, {i['frames']} frames at {i['fps']} fps; cameras {cams}"
              + ("; has edits" if i["edits"] else ""))
        for t in i["tasks"] or ["(no task text)"]:
            print(f"    task: {t!r}")
    return 0


def cmd_train(a, extra: list[str]) -> int:
    from . import train as T
    plan = T.make_plan(
        a.datasets, name=a.name, profile=a.profile, steps=a.steps, batch_size=a.batch_size, num_workers=a.num_workers,
        save_freq=a.save_freq, log_freq=a.log_freq, base=getattr(a, "base", None), parent=a.parent, task=a.task,
        build=not a.no_build, auto_trim=not a.no_trim, augment=not a.no_augment, chunk_size=a.chunk_size,
        extra=extra, work=a.work_dir, shared=a.shared, publish_dir=a.publish_dir, publish_every=a.publish_every,
        remote=a.remote, remote_root=a.remote_dir, ssh_port=a.ssh_port, device=a.device)
    if a.remote:
        return T.train_remote(plan, rebuild=a.rebuild, follow=a.follow, print_only=a.print_only)
    return T.train_local(plan, detach=a.detach, rebuild=a.rebuild, force=a.force, print_only=a.print_only)


def cmd_status(a) -> int:
    from . import train as T
    work = Path(a.work_dir).expanduser()
    run = work / "runs" / a.name
    if not run.is_dir():
        print(f"no run {a.name!r} in {work / 'runs'}")
        return 1
    if (run / "remote.json").is_file():
        st = T.remote_status(a.name, work)
        where = (T.read_json(run / "remote.json") or {}).get("host")
        print(f"{a.name} on {where}: {T.status_line(st)}" + ("" if st.get("supervisor_running") else " (not running)"))
    else:
        st = T.read_json(run / "out.status.json", {}) or {}
        alive = T.supervisor_pid(run) is not None
        print(f"{a.name}: {T.status_line(st)}" + ("" if alive else " (not running)"))
    if st.get("published"):
        print("published: " + ", ".join(st["published"]))
    return 0


def cmd_serve(a) -> int:
    from .serve import serve
    return serve(host=a.host, port=a.port, fps=a.fps, policies_dir=Path(a.policies_dir), tunnel=a.tunnel,
                 tunnel_port=a.tunnel_port)


def run_options(a):
    from .run import RunOptions
    return RunOptions(server=a.server, policy=a.policy, task=a.task, dry_run=a.dry_run, driver_url=a.driver,
                      camera_url=a.engine, cameras=a.cameras, image_width=a.image_width, image_height=a.image_height,
                      fps=a.fps, policy_device=a.policy_device, actions_per_chunk=a.actions_per_chunk,
                      chunk_size_threshold=a.chunk_threshold, aggregate=a.aggregate, max_duration_s=a.max_duration,
                      dry_run_s=a.dry_run_seconds, goto_start=a.goto_start, facts=a.facts, speed_pct=a.speed,
                      gripper_effort=a.gripper_effort, step_limit_x=a.step_limit_x, lead_limit_x=a.lead_limit_x,
                      envelope_margin=a.envelope_margin, stagnation_s=a.stagnation_s, shared=a.shared)


def cmd_run(a) -> int:
    from . import run as R
    o = run_options(a)
    plan = R.plan_run(o)
    mode = "DRY RUN (nothing is sent to the arm)" if o.dry_run else "LIVE (the arm will move)"
    print(f"{mode}: policy {plan['send_ref']} on {o.server}, task {o.task!r}, cameras {plan['cameras']} "
          f"{plan['image_size'][1]}x{plan['image_size'][0]}, {plan['fps']} Hz, chunk {plan['actions_per_chunk']}")
    checks = R.preflight(o, plan)
    for level, msg in checks:
        print(f"  [{level:5}] {msg}")
    if any(level == "error" for level, _ in checks):
        print("not running: fix the errors above")
        return 2
    if not o.dry_run:
        if not a.yes:
            if not sys.stdin.isatty():
                print("a live run needs --yes when there is no terminal to confirm in")
                return 2
            if input("The arm will move. Someone is next to it with the E-stop in reach? [y/N] ").strip().lower() != "y":
                print("not running")
                return 1
        if o.goto_start:
            sp = (plan["facts"] or {}).get("start_pose")
            if not sp:
                print("--goto-start needs the policy's nervus_policy.json (start_pose)")
                return 2
            print("moving to the start pose (4 s at 20 % speed)", flush=True)
            R.goto_start(o, sp, fps=plan["fps"])
        for n in (3, 2, 1):
            print(f"starting in {n}...", flush=True)
            time.sleep(1)
    cfg = R.client_config(o, plan)
    rep = R.run_client(cfg, o.dry_run_s if o.dry_run else o.max_duration_s)
    print(json.dumps(rep, indent=2))
    print(f"\n{'dry run' if o.dry_run else 'run'} ended: {rep['stop_reason']}; planned motion up to "
          f"{rep['max_plan_deg']} deg: {rep['verdict']}")
    if o.dry_run:
        return 0 if rep["actions"] else 1
    return 0


def cmd_sim(a) -> int:
    from .sim import SimCameraServer, SimDriver
    drv = SimDriver(port=a.driver_port)
    cams = SimCameraServer(drv.arm, roles=tuple(a.cameras), port=a.camera_port, width=a.width, height=a.height)
    print(f"simulated PiPER driver  {drv.url}\nsimulated cameras       {cams.url}  ({', '.join(a.cameras)}, "
          f"{a.width}x{a.height})", flush=True)
    print(f"try: piper-smolvla run --driver {drv.url} --engine {cams.url} --policy NAME --task \"...\" --dry-run")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    cams.close()
    drv.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    a, unknown = ap.parse_known_args(argv)
    extra = []
    if unknown:
        if a.cmd not in ("train", "posttrain"):
            ap.error(f"unrecognized arguments: {' '.join(unknown)}")
        bad = [u for u in unknown if not (u.startswith("--") and "=" in u)]
        if bad:
            ap.error(f"unrecognized arguments: {' '.join(bad)} (lerobot-train arguments are written --key=value)")
        extra = unknown
    try:
        if a.cmd == "datasets":
            return cmd_datasets(a)
        if a.cmd in ("train", "posttrain"):
            return cmd_train(a, extra)
        if a.cmd == "status":
            return cmd_status(a)
        if a.cmd == "pull":
            from .train import pull
            published = pull(a.name, Path(a.work_dir).expanduser(), with_state=a.with_state, publish=not a.no_publish)
            print("published: " + (", ".join(published) if published else "nothing new"))
            return 0
        if a.cmd == "stop":
            from . import train as T
            work = Path(a.work_dir).expanduser()
            remote = (work / "runs" / a.name / "remote.json").is_file()
            return T.stop_remote(a.name, work) if remote else T.stop_local(a.name, work)
        if a.cmd == "publish":
            from .train import publish_run
            out = publish_run(a.name, Path(a.work_dir).expanduser())
            print("published: " + (", ".join(out) if out else "nothing (no complete checkpoint selected)"))
            return 0
        if a.cmd == "serve":
            return cmd_serve(a)
        if a.cmd == "run":
            return cmd_run(a)
        if a.cmd == "sim":
            return cmd_sim(a)
    except (ValueError, FileNotFoundError, RuntimeError, ConnectionError) as e:
        print(f"piper-smolvla {a.cmd}: {e}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
