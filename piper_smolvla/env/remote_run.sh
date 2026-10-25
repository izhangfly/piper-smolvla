#!/usr/bin/env bash
# One training run on the GPU box (inside tmux, started by remote_launch.sh):
#   1. the Python environment from the pinned lock (linux-cuda.txt), reinstalled only when the lock changes;
#   2. ffmpeg for torchcodec's video decoding, or PyAV decoding where ffmpeg can't be installed;
#   3. the starting policy (lerobot/smolvla_base with its input features left to the data), if not there;
#   4. lerobot-train under the supervisor, which resumes from the last complete checkpoint on every launch.
#     remote_run.sh <root> <run name>
set -uo pipefail
ROOT="$1"
NAME="$2"
RUN="$ROOT/runs/$NAME"
VENV="$ROOT/venv"
LOCK="$ROOT/src/piper_smolvla/env/linux-cuda.txt"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$ROOT/hf}"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
mkdir -p "$RUN"
exec > >(tee -a "$RUN/remote.log") 2>&1
say() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }
fail() { say "FAILED: $*"; exit 1; }
as_root() {
  if [ "$(id -u)" = 0 ]; then "$@"; elif sudo -n true 2>/dev/null; then sudo "$@"; else return 1; fi
}

say "run $NAME on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "no NVIDIA GPU (nvidia-smi)"
# Adapted from zhitaoqiu/Piper-SmolVla/scripts/run_train_4090_smolvla_twoobj200.sh (GPU memory precheck, allocator).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-8000}"
FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [ -n "$FREE_MB" ] && [ "$FREE_MB" -lt "$MIN_FREE_GPU_MB" ]; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
  fail "only ${FREE_MB} MB of GPU memory is free (need ${MIN_FREE_GPU_MB}; set MIN_FREE_GPU_MB to override)"
fi

# 1. environment
WANT="$(sha256sum "$LOCK" | cut -d' ' -f1)"
if [ ! -x "$VENV/bin/lerobot-train" ] || [ "$(cat "$VENV/.lock.sha256" 2>/dev/null)" != "$WANT" ]; then
  say "installing the environment from $(basename "$LOCK")"
  if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user --quiet uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  fi
  command -v uv >/dev/null 2>&1 || fail "could not install uv"
  [ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.11 || fail "uv venv"
  uv pip install --python "$VENV/bin/python" -r "$LOCK" || fail "installing $LOCK"
  echo "$WANT" > "$VENV/.lock.sha256"
fi
"$VENV/bin/python" -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok); sys.exit(0 if ok else 1)" \
  || fail "torch cannot use the GPU (driver too old for CUDA 12.8? see nvidia-smi above)"

# 2. ffmpeg (torchcodec links against the system's); without it, decode with PyAV, which bundles its own
if ! command -v ffmpeg >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  say "installing ffmpeg"
  (as_root apt-get update -qq && as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg) >/dev/null 2>&1 || true
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  say "no ffmpeg: decoding video with PyAV"
  "$VENV/bin/python" - "$RUN/supervisor.json" <<'EOF'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
if "--dataset.video_backend=pyav" not in cfg["args"]:
    cfg["args"].append("--dataset.video_backend=pyav")
    json.dump(cfg, open(p, "w"), indent=2)
EOF
fi

# 3. the starting policy
if [ -f "$RUN/base_ref.txt" ] && [ ! -f "$RUN/base/config.json" ]; then
  say "preparing the base policy $(cat "$RUN/base_ref.txt")"
  "$VENV/bin/python" -m piper_smolvla.base "$(cat "$RUN/base_ref.txt")" "$RUN/base" || fail "preparing the base policy"
fi

# 4. train
cd "$RUN"
say "training (the supervisor resumes from the last complete checkpoint)"
echo $$ > "$RUN/supervisor.pid"
exec "$VENV/bin/python" -m piper_smolvla.supervisor --config "$RUN/supervisor.json"
