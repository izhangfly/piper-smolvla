"""The policy server: LeRobot's async-inference PolicyServer, where the model runs (the Mac, on MPS).

Two additions over `python -m lerobot.async_inference.policy_server`:

* Policy names. A client may ask for a policy by name; the server looks it up in its own copy of the shared
  folder (<shared>/smolvla/policies/<name>), so the robot host and the Mac don't need the same home path.
* --tunnel HOST. LeRobot's protocol is unauthenticated gRPC carrying pickled Python objects, so anyone who can
  reach the port can run code on this machine. By default the server listens on loopback only, and --tunnel
  opens `ssh -R` to the robot host so its client reaches the server at 127.0.0.1:<port> there. Binding to the
  network (--host 0.0.0.0) is possible, for a trusted LAN only.
"""
from __future__ import annotations

import pickle  # nosec: the protocol is LeRobot's; see the module docstring for why it stays on loopback
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

from .run import LOG_DIR, SHARED, in_dir

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def resolve_name(ref: str, policies_dir: Path) -> str:
    """A bare policy name that exists in policies_dir becomes its path; anything else is passed on unchanged
    (a path on this machine, or a Hub id)."""
    cand = Path(policies_dir).expanduser() / ref
    if ref and "/" not in ref and (cand / "config.json").is_file():
        return str(cand)
    return ref


def make_server_class():
    with in_dir(LOG_DIR):
        from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.transport import services_pb2  # type: ignore

    class PiperPolicyServer(PolicyServer):
        policies_dir = SHARED / "smolvla" / "policies"

        def SendPolicyInstructions(self, request, context):  # noqa: N802
            spec = pickle.loads(request.data)  # nosec
            ref = getattr(spec, "pretrained_name_or_path", "")
            resolved = resolve_name(ref, self.policies_dir)
            if resolved != ref:
                self.logger.info(f"policy {ref!r} -> {resolved}")
                spec.pretrained_name_or_path = resolved
                request = services_pb2.PolicySetup(data=pickle.dumps(spec))
            return super().SendPolicyInstructions(request, context)

    return PiperPolicyServer


def open_tunnel(host: str, port: int, remote_port: int | None = None) -> subprocess.Popen:
    """ssh -R: the robot host's 127.0.0.1:<remote_port> reaches this server's 127.0.0.1:<port>."""
    rp = remote_port or port
    argv = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", "-o", "BatchMode=yes",
            "-R", f"127.0.0.1:{rp}:127.0.0.1:{port}", host]
    print("tunnel: " + shlex.join(argv), flush=True)
    proc = subprocess.Popen(argv)
    time.sleep(2.0)
    if proc.poll() is not None:
        raise RuntimeError(f"the ssh tunnel to {host} failed (is port {rp} free there, and does `ssh {host}` work?)")
    return proc


# Adapted from huggingface/lerobot/src/lerobot/async_inference/policy_server.py (serve).
def serve(host: str = "127.0.0.1", port: int = 8080, fps: int = 15, policies_dir: Path | None = None,
          tunnel: str | None = None, tunnel_port: int | None = None, obs_queue_timeout: float = 2.0) -> int:
    import grpc
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.transport import services_pb2_grpc  # type: ignore

    if host not in LOOPBACK:
        print(f"WARNING: listening on {host}:{port}. LeRobot's policy protocol is unauthenticated and unpickles what "
              "it receives: anyone who can reach this port can run code here. Use it on a trusted network only, or "
              "prefer --tunnel.", file=sys.stderr, flush=True)
    Server = make_server_class()
    if policies_dir:
        Server.policies_dir = Path(policies_dir).expanduser()
    cfg = PolicyServerConfig(host=host, port=port, fps=fps, inference_latency=1.0 / fps,
                             obs_queue_timeout=obs_queue_timeout)
    policy_server = Server(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    if server.add_insecure_port(f"{host}:{port}") == 0:
        raise RuntimeError(f"could not listen on {host}:{port} (in use?)")
    server.start()
    print(f"policy server on {host}:{port} ({fps} fps); policies by name from {Server.policies_dir}", flush=True)
    tun = open_tunnel(tunnel, port, tunnel_port) if tunnel else None
    if tun:
        print(f"on {tunnel}: piper-smolvla run --server 127.0.0.1:{tunnel_port or port} --policy NAME --task \"...\"",
              flush=True)
    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())
    try:
        while not done.wait(1.0):
            if tun and tun.poll() is not None:
                print("the ssh tunnel closed; reopening it", flush=True)
                try:
                    tun = open_tunnel(tunnel, port, tunnel_port)
                except RuntimeError as e:
                    print(e, flush=True)
                    time.sleep(10)
    finally:
        if tun and tun.poll() is None:
            tun.terminate()
        policy_server.stop()
        server.stop(grace=2).wait(5)
        print("policy server stopped", flush=True)
    return 0
