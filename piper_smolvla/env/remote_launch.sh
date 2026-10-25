#!/usr/bin/env bash
# Starts one training run on the GPU box in a tmux session, or reports that it is already running.
# `piper-smolvla train --remote` runs this over ssh after copying the run over:
#     remote_launch.sh <root> <run name>
set -euo pipefail
ROOT="$1"
NAME="$2"
RUN="$ROOT/runs/$NAME"
SESSION="psv-$NAME"
SCRIPT="$ROOT/src/piper_smolvla/env/remote_run.sh"

if [ -f "$RUN/supervisor.pid" ] && kill -0 "$(cat "$RUN/supervisor.pid")" 2>/dev/null; then
  echo "already running (supervisor pid $(cat "$RUN/supervisor.pid"))"
  exit 0
fi

as_root() {
  if [ "$(id -u)" = 0 ]; then "$@"; elif sudo -n true 2>/dev/null; then sudo "$@"; else return 1; fi
}
if ! command -v tmux >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  (as_root apt-get update -qq && as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux) >/dev/null 2>&1 || true
fi

if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  # The shell stays open after the run ends, so `tmux attach` still shows how it ended.
  tmux new-session -d -s "$SESSION" "bash '$SCRIPT' '$ROOT' '$NAME'; echo; echo '[run ended; this shell stays for inspection]'; exec bash"
  echo "started in tmux session $SESSION"
else
  nohup setsid bash "$SCRIPT" "$ROOT" "$NAME" >/dev/null 2>&1 < /dev/null &
  echo "started with nohup (no tmux here); the log is $RUN/remote.log"
fi
