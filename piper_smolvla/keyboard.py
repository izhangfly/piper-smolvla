# Copyright 2026 jokeru8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Ian Zhang (izhangfly) for piper-smolvla: taken from piper_lerobot's
# src/lerobot/robots/piper_follower/keyboard_stop.py; the "h" (smooth home) key is dropped, because this
# project only ever holds the arm where it is, and the callback takes no argument.

"""Interactive stop hotkey for a policy run: press s (then Enter if the terminal is line-buffered) to stop the
run and hold the arm where it is. Ctrl+C does the same."""

from __future__ import annotations

import logging
import select
import sys
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


# Adapted from jokeru8/piper_lerobot/src/lerobot/robots/piper_follower/keyboard_stop.py.
class KeyboardStopWatcher:
    def __init__(self, on_stop: Callable[[], None]) -> None:
        self._on_stop = on_stop
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if not sys.stdin.isatty():
            logger.info("stdin is not a TTY; the s (stop and hold) hotkey is disabled")
            return False
        self._thread = threading.Thread(target=self._loop, name="piper-keyboard-stop", daemon=True)
        self._thread.start()
        logger.info("press s then Enter to stop and hold the arm (or Ctrl+C)")
        return True

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            except (OSError, ValueError):
                break
            if not ready:
                continue
            try:
                key = sys.stdin.read(1).lower()
            except OSError:
                break
            if key == "s":
                self._on_stop()
                break
