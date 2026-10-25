#!/usr/bin/env python3
"""PiPER Control: the arm driver's joint and Cartesian panel (panel.html) as a desktop app.

Opening it starts the driver (nervus-piper.service; the desktop user may start it, see the polkit rule) if it
isn't running, waits until it answers on 127.0.0.1:8770, then shows the panel in its own window. Closing the
window stops any held jog key. The motors are left as they are (a disabled PiPER drops under gravity), and the
panel's heartbeat lapses, so the driver stops streaming within 1.5 s.

    piper-control [--url http://127.0.0.1:8770]

Needs python3-gi and gir1.2-webkit2-4.1 (both on Ubuntu 24.04 desktops).
"""
import argparse
import json
import subprocess
import threading
import time
import urllib.request

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("WebKit2", "4.1")
from gi.repository import GLib, Gtk, WebKit2  # noqa: E402

START_TIMEOUT_S = 25


def answering(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/api/state?lite=1", timeout=1.0) as r:
            return r.status == 200
    except OSError:
        return False


def post(url: str, path: str, body: dict):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=1.0).read()
    except OSError:
        pass


class Window(Gtk.Window):
    def __init__(self, url: str):
        super().__init__(title="PiPER Control")
        self.url = url
        self.set_default_size(1280, 900)
        self.set_icon_name("nervus-piper-control")
        self.stack = Gtk.Stack()
        self.message = Gtk.Label(label="Starting the arm driver…")
        self.message.set_line_wrap(True)
        self.stack.add_named(self.message, "message")
        self.web = WebKit2.WebView()
        self.stack.add_named(self.web, "panel")
        self.add(self.stack)
        self.connect("destroy", self.on_close)
        threading.Thread(target=self.bring_up, daemon=True).start()

    def bring_up(self):
        if not answering(self.url):
            # nervus-piper.service is a system unit the desktop user may start without a password.
            r = subprocess.run(["systemctl", "start", "nervus-piper.service"], capture_output=True, text=True)
            if r.returncode != 0:
                GLib.idle_add(self.fail, "Could not start the arm driver:\n" + (r.stderr.strip() or "unknown error")
                              + "\n\nCheck: systemctl status nervus-piper")
                return
        end = time.time() + START_TIMEOUT_S
        while time.time() < end:
            if answering(self.url):
                GLib.idle_add(self.show_panel)
                return
            time.sleep(0.5)
        GLib.idle_add(self.fail, "The arm driver started but isn't answering on " + self.url +
                      ".\nIs the CAN adapter plugged in and the arm powered?\n\nCheck: journalctl -u nervus-piper -f")

    def show_panel(self):
        self.web.load_uri(self.url + "/")
        self.stack.set_visible_child_name("panel")
        return False

    def fail(self, text: str):
        self.message.set_text(text)
        return False

    def on_close(self, *_):
        # Release any jog key that was held when the window closed; the heartbeat deadman does the rest.
        post(self.url, "/api/jogstop", {"client": "panel"})
        Gtk.main_quit()


def main():
    ap = argparse.ArgumentParser(description="PiPER Control: the arm driver's panel as a desktop app")
    ap.add_argument("--url", default="http://127.0.0.1:8770")
    args = ap.parse_args()
    if not args.url.startswith(("http://127.0.0.1", "http://localhost")):
        raise SystemExit("the driver listens on loopback only")
    Window(args.url).show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
