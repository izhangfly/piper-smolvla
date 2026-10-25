# PiPER SmolVLA

Fine-tune and run [SmolVLA](https://huggingface.co/lerobot/smolvla_base), LeRobot's small vision-language-action
policy, on an [AgileX PiPER](https://global.agilex.ai/products/piper) arm. You show the arm a task by hand, train
SmolVLA on those demonstrations (on an Apple-silicon Mac or a rented CUDA GPU), and run it by telling the arm what
to do: `--task "pick up the red cube"`.

Everything heavy is LeRobot's own code: `lerobot-train` trains, and LeRobot's asynchronous inference
(`PolicyServer` where the model runs, `RobotClient` next to the arm) runs the policy. This project adds what sits
around it on a real PiPER: a LeRobot robot class for the arm, safety limits on every command, a training
supervisor that resumes after crashes, remote GPU training over ssh, and a dry run that checks everything without
moving the arm.

```text
 robot host (Ubuntu, at the arm)                          Mac (Apple silicon)                  GPU box (optional)
 ┌───────────────────────────────────────┐               ┌─────────────────────────────┐       ┌──────────────────┐
 │ piper-smolvla run  (LeRobot client)   │   gRPC, over   │ piper-smolvla serve         │       │ lerobot-train    │
 │   robot type piper_http               │◄─ ssh tunnel ─►│  (LeRobot PolicyServer,     │       │ under supervisor │
 │      │ HTTP, loopback                 │                │   SmolVLA on MPS)           │       │ in tmux          │
 │ PiPER driver :8770 ── CAN ── PiPER    │                │ piper-smolvla train ────────┼─ssh──►│                  │
 │ ACT engine   :8795 ── wrist + scene   │                │  (MPS, or --remote)         │rsync  │                  │
 └───────────────────────────────────────┘                └─────────────────────────────┘       └──────────────────┘
        ▲ datasets (ACT Studio's recorder) and published policies travel in ~/Shared/NervusACT (Syncthing)
```

It was built for, and tested on, one PiPER on a Surface Pro 9 running Ubuntu 24.04, with training and inference on
a 16 GB M-series Mac. It is part of [NervusOS](https://github.com/izhangfly/NervusOS-Cambridge), whose PiPER
driver and ACT engine it talks to.

## Hardware and the services it expects

| Part | What this project assumes |
|---|---|
| Arm | AgileX PiPER (6 joints + parallel gripper) on CAN, driven by the [NervusOS PiPER driver](https://github.com/izhangfly/NervusOS-Cambridge/tree/main/hardware/piper) (`piper_webctl.py`, `127.0.0.1:8770`; a copy with its systemd units is in `driver/` of the standalone repository). Nothing here touches CAN: the robot class is an HTTP client of the driver, claiming the arm as client `smolvla`, so it never fights the other apps that use the arm (one controller at a time). |
| Cameras | Served by the [NervusOS ACT engine](https://github.com/izhangfly/NervusOS-Cambridge/tree/main/act) as `GET /api/camera/<role>.jpg` (`127.0.0.1:8795`): `wrist` (Orbbec Dabai DC1 RGB) and `scene` (a third-person USB webcam), 640x480. The engine opens them by `/dev/v4l/by-id` name and releases them when idle; the first request wakes them. |
| Robot host | Any Linux machine running those two services. It only runs the client: CPU torch, no GPU. |
| Training and inference | An Apple-silicon Mac (16 GB fits batch 4 on MPS; one inference takes about 0.5 s), and optionally an NVIDIA GPU box reachable over ssh. |

Units are the driver's and the datasets': joints in degrees, the gripper opening in millimetres. Observation and
action are `[joint1..joint6, gripper]`, the ACT engine's dataset contract (`~/Shared/NervusACT/CONTRACT.json`).

Without NervusOS, the robot class works with anything that serves the same small HTTP API: `GET /api/state` and
`POST /api/stream`, `/api/release`, `/api/speed` for the arm, `GET /api/camera/<role>.jpg` for frames. See
`piper_smolvla/clients.py`, and `piper_smolvla/sim.py` for a complete simulated implementation.

## Install

```sh
git clone https://github.com/izhangfly/piper-smolvla && cd piper-smolvla
```

**On the Mac** (training and the policy server), in a venv of its own:

```sh
uv venv ~/.venvs/smolvla --python 3.11
uv pip install --python ~/.venvs/smolvla/bin/python -e ".[smolvla,test]"
source ~/.venvs/smolvla/bin/activate && piper-smolvla --help
```

`piper_smolvla/env/macos-arm64.txt` pins the exact versions it was developed with (LeRobot 0.4.4, torch 2.10,
transformers 4.57); `uv pip install -r` it for the identical environment.

**On the robot host** only the client is needed, with CPU torch (this is how the Surface was set up):

```sh
python3 -m venv ~/.venvs/smolvla
~/.venvs/smolvla/bin/pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
~/.venvs/smolvla/bin/pip install -e .
```

**A GPU box** needs nothing installed: `train --remote` sets it up from the pinned `piper_smolvla/env/linux-cuda.txt`.

## 1. Record demonstrations

Record with ACT Studio's **Record** tab (it drives the ACT engine's recorder): open or create a dataset, give it
a one-sentence task, put the arm in drag-teach, and record takes. The recorder refuses a dataset without a task,
stores the sentence with every frame (`meta/tasks.parquet`) and in `nervus_dataset.json`, and writes a LeRobot v3
dataset (15 fps, action = the next measured state) to `~/Shared/NervusACT/datasets/<name>`. SmolVLA is conditioned
on that sentence, so word it the way you will ask at run time, and use different sentences for different tasks.
Trims and exclusions from ACT Studio's **Data** tab are applied when the training set is built.

```sh
piper-smolvla datasets                  # every dataset with its episodes, cameras and task text
```

Any LeRobot v3 dataset with the same state and action layout works too: pass its path instead of a name.

## 2. Train on the Mac

One or more datasets go in; datasets trained together must share their cameras. LeRobot 0.4.4 trains on one
dataset at a time, so they are merged into one training set first, with the same rules as the ACT worker: edits
applied, idle stillness at both ends of each take trimmed, video re-encoded as h264 at CRF 18. Each episode keeps
its recorded task (or `--task "..."` replaces them all). Training starts from `lerobot/smolvla_base`: MPS, batch 4,
20,000 steps, a checkpoint every 2,000.

```sh
piper-smolvla train cube_v1 cube_v2 --name cube
piper-smolvla train cube_v1 --name cube --print-only     # show the plan and the lerobot-train command only
piper-smolvla status cube                                # step, loss, step time, ETA
piper-smolvla stop cube                                  # ends it cleanly; the same train command resumes it
```

It refuses to start next to another training run or when less than 20 % of memory is free (`--force` overrides),
keeps the Mac awake while it trains, and follows the run in the terminal (Ctrl+C stops it cleanly; `--detach`
returns at once). Running the same command again resumes from the last complete checkpoint, and the supervisor does
the same by itself after a crash, a hang or a NaN loss.

Arguments written `--key=value` after the options go straight to `lerobot-train`, for example
`--policy.optimizer_lr=5e-5`. Other options: `--steps`, `--batch-size`, `--save-freq`, `--chunk-size`,
`--no-trim`, `--no-augment`, `--no-build` (train on one dataset exactly as it is), `--publish-every N`.

## 3. Train on a rented GPU

The `cuda` profile is batch 64 for 20,000 steps, about two hours on one RTX 4090:

```sh
piper-smolvla train cube_v1 cube_v2 --name cube --remote ubuntu@gpu-box [--ssh-port 2222] [--follow]
piper-smolvla status cube                # step, loss, ETA, read from the box
piper-smolvla pull cube                  # copy checkpoints back (without optimizer state) and publish them
piper-smolvla stop cube                  # the same train command resumes it
```

The training set is built on the Mac and sent with rsync together with this package and the run's config. On the
box, a tmux session (`psv-cube`; `ssh gpu-box -t tmux attach -t psv-cube` to watch) installs the pinned
environment with `uv` (once; again only when the lock changes), installs ffmpeg if it can (otherwise video is
decoded with PyAV), downloads the base model there, checks that enough GPU memory is free, and runs `lerobot-train`
under the supervisor. If the box reboots or the session dies, run the train command again: it resumes. Hosts, keys
and jump hosts come from `~/.ssh/config`. `--follow` waits for the end, then pulls and publishes.

## 4. Post-train on corrections

Start from an earlier SmolVLA policy and train on datasets that include correction episodes (record them in ACT
Studio from the states where the policy went wrong). Same machinery, locally or with `--remote`; 5,000 steps by
default.

```sh
piper-smolvla posttrain --parent cube-smolvla_020000 cube_v1 cube_fixes [--remote ubuntu@gpu-box]
```

## What a run leaves behind

Runs live in `~/piper-smolvla/runs/<name>/` (`--work-dir`): the built training set, the prepared base,
`facts.json`, `supervisor.json`, LeRobot's `out/` and its log `out.log`. The final checkpoint (and, with
`--publish-every N`, every Nth) is published the way the ACT worker publishes policies, but to its own folder, so the
ACT engine never mistakes it for an ACT policy:

```text
~/Shared/NervusACT/smolvla/policies/<name>-smolvla_<step>/
    config.json  model.safetensors  policy_preprocessor*.json  policy_postprocessor*.json  train_config.json
    nervus_policy.json     cameras, task texts, fps, units, the training data's envelope, per-step 99th
                           percentiles and median start pose (the robot side's safety limits), its origin
```

**Why a prepared base.** LeRobot 0.4.4 keeps a pretrained policy's input features when fine-tuning
([huggingface/lerobot#4517](https://github.com/huggingface/lerobot/issues/4517)). `smolvla_base` declares three
256x256 cameras named `camera1..3` and a 6-value state, so a plain fine-tune either fails the camera check or saves a
config that makes LeRobot's policy server squash 640x480 frames to 256x256 at run time. `piper_smolvla/base.py`
gives the trainer a copy of the base (weights symlinked) whose config leaves the input features to the dataset, so
the fine-tuned policy's config describes exactly what the robot sends.

## 5. Serve on the Mac

LeRobot's policy protocol is unauthenticated gRPC carrying pickled objects: anyone who can reach the port can run
code on the server. So the server listens on loopback by default, and a reverse ssh tunnel brings it to the robot
host, where it appears at 127.0.0.1:8080:

```sh
piper-smolvla serve --tunnel surface     # `surface` is the robot host's ssh name; the tunnel reopens if it drops
# trusted LAN only: piper-smolvla serve --host 0.0.0.0   (then run --server <mac>:8080 on the robot host)
```

Policies are asked for by name and found in the Mac's own `~/Shared/NervusACT/smolvla/policies` (`--policies-dir`),
so the two machines' home paths may differ. The model loads when the first client connects.

## 6. Run on the robot host

Always a dry run first. It checks the driver, the cameras (waiting for the engine to wake them), the policy's
cameras and sizes, the safety facts and the server, then runs the real LeRobot client for ten seconds and plans
everything without sending a single command to the arm:

```sh
piper-smolvla run --policy cube-smolvla_020000 --task "pick up the red cube" --dry-run
```

It ends with how far the policy planned to move. As in the ACT engine's scene check, a policy that plans to sit
still (under 8 degrees) is seeing a scene unlike its demonstrations: fix the light or the camera, not the policy.

Then, with the motors enabled (Robot Arm app or ACT Studio; this client never enables them) and someone at the arm:

```sh
piper-smolvla run --policy cube-smolvla_020000 --task "pick up the red cube" [--goto-start] [--max-duration 90]
```

It asks for confirmation (`--yes` skips it), counts down, and runs for at most `--max-duration` seconds. Press
**s** then Enter, or Ctrl+C, to stop; the arm holds where it is, motors on, and control is released.
`--goto-start` first moves slowly (4 s at 20 % speed) to the demonstrations' median start pose. The policy is
named as published, or given as a path on the server or a Hugging Face Hub id; `--cameras wrist` runs a policy
with fewer cameras. Joints from overlapping chunks are blended with LeRobot's weighted average, the gripper taking
the newest chunk only (`--aggregate`).

**Try it without hardware.** `piper-smolvla sim` starts a simulated driver and simulated cameras on loopback and
prints the `--driver` and `--engine` URLs to pass to `run`.

## Safety

Every action passes through `piper_smolvla/safety.py` before it reaches the driver. The rules are the NervusOS ACT
engine's, learned on this arm, plus some from other PiPER SmolVLA projects:

- Targets are clamped to the demonstrated range plus 5 degrees (mm for the gripper) and to the joint limits the
  driver reports; the driver clamps to the manual's limits again.
- A per-step limit on the command (1.5x the demonstrations' 99th-percentile step; 3x that for the gripper), counted
  from the arm's measured pose on the first step, and a lead limit (4x) on how far a command may run ahead of the arm.
- The grasp latch keeps a gripper that has closed on an object at "measured minus 1.5 mm" and lets go only after
  eight open requests in a row; grip effort 500 (1000 crushed a printed cube). The gripper is never averaged across
  overlapping chunks (the mean of open and closed is half-open).
- Stale arm feedback or camera frames hold the last command; stale for more than a second ends the run.
- The run ends if the arm is more than 30 degrees from its last command (blocked, pushed or faulted), if it enters
  drag-teach (the policy lets go and sends nothing more, so the arm is never stiffened in someone's hands), on a
  hard fault, if another app takes the arm, on a non-finite action, or if the arm stays still for 8 s.
- A live run refuses to start when the motors are off, the arm is in drag-teach or emergency-stopped, or another
  client holds it. On every exit the arm holds its measured pose, motors on, and control is released.
- Without `nervus_policy.json` (a policy from elsewhere), the limits fall back to the driver's joint limits and
  4 degrees as every joint's demonstrated step, as in the ACT engine.

Keep the E-stop in reach, start with slow, short runs, and do a dry run after every new checkpoint or task text.

## Tests

```sh
python -m pytest          # about a minute; no hardware, no model, no network
```

They run the robot class against the simulated driver and cameras (observation/action round trip, clamps, dry runs
that never send a command, refusals, staleness, hand-over, losing control, stagnation), the whole run path through
LeRobot's real `RobotClient` and gRPC against a policy server whose model is replaced by a fixed motion, the
dataset build on tiny real LeRobot datasets, the GPU-box scripts with stand-ins for the GPU and `lerobot-train`
(including a resume), and the command line. A guard fails any test that tries to reach the real driver or ACT
engine ports.

## Files

| Path | What it is |
|---|---|
| `piper_smolvla/config.py`, `robot.py` | The LeRobot robot (`--robot.type=piper_http`) |
| `piper_smolvla/clients.py` | HTTP clients for the driver and the camera endpoint |
| `piper_smolvla/safety.py` | Clamps, limits, grasp latch, stops |
| `piper_smolvla/data.py` | Dataset checks, the training-set build, the facts |
| `piper_smolvla/base.py` | The prepared SmolVLA base |
| `piper_smolvla/train.py`, `supervisor.py`, `remote.py` | Training locally and remotely, publishing |
| `piper_smolvla/env/` | Pinned environments and the GPU box's two scripts |
| `piper_smolvla/serve.py`, `run.py`, `keyboard.py` | The policy server and the robot client |
| `piper_smolvla/sim.py` | Simulated driver and cameras |
| `piper_smolvla/cli.py` | `piper-smolvla` |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `another app is controlling the arm` | Robot Arm, Hand Teleop or the ACT engine holds it: close it or press Stop there. |
| `the arm's feedback is stale` | The arm is idle in standby: press its drag-teach button once (light on), then Enable. |
| `camera ... 404` after 8 s | The camera isn't plugged in or the engine can't open it; `curl -o x.jpg 127.0.0.1:8795/api/camera/wrist.jpg`. |
| Dry run: `no actions arrived` | Look at the server's output: usually the run's cameras differ from the policy's (`--cameras`). |
| Dry run: `would likely sit still` | The scene, light or camera position differs from the demonstrations. |
| `remote port forwarding failed` from `serve --tunnel` | Port 8080 is still held on the robot host by an old session; it retries every 10 s, or use `--tunnel-port`. |
| Remote: `torch cannot use the GPU` | The box's NVIDIA driver is too old for the CUDA 12.8 wheels in `linux-cuda.txt`. |
| Training crawls on the Mac | Memory pressure: close large apps. `status` shows the step time. |

## Credits

- [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0) and [SmolVLA](https://huggingface.co/lerobot/smolvla_base)
  (*SmolVLA: Smol Models for Vision-Language-Action*, Hugging Face, 2025): training, the policy, async inference.
- [AgRoboticsResearch/lerobot_robot_piper](https://github.com/AgRoboticsResearch/lerobot_robot_piper): the robot plugin layout.
- [dk2472780158-ctrl/piper-dual-arm-smolvla-towel-folding](https://github.com/dk2472780158-ctrl/piper-dual-arm-smolvla-towel-folding):
  SmolVLA on real PiPER arms; the tracking stop and hold-on-exit rules.
- [jokeru8/piper_lerobot](https://github.com/jokeru8/piper_lerobot): the stop hotkey.
- [zhitaoqiu/Piper-SmolVla](https://github.com/zhitaoqiu/Piper-SmolVla): the stagnation stop and the GPU precheck.
- [zhaoyufei122/piper-smolvla](https://github.com/zhaoyufei122/piper-smolvla): remote 4090 fine-tuning settings.
- [lykycy123/lerobot-piper](https://github.com/lykycy123/lerobot-piper), [Kane1440/lerobot_piper2](https://github.com/Kane1440/lerobot_piper2),
  [WilbertYuan/lerobot_piper3](https://github.com/WilbertYuan/lerobot_piper3): earlier PiPER LeRobot integrations.
- The [NervusOS](https://github.com/izhangfly/NervusOS-Cambridge) ACT engine and Mac worker, where the safety rules,
  the dataset build and the training supervisor were worked out on this arm.

[NOTICE](NOTICE) lists exactly what came from where; copied code carries an "Adapted from" comment.

## Author and licence

Ian Zhang ([izhangfly](https://github.com/izhangfly)). Apache License 2.0, see [LICENSE](LICENSE).
