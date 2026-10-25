"""The whole run path without a model: LeRobot's RobotClient and gRPC transport, the piper_http robot against the
simulated driver and cameras, and a policy server that is LeRobot's own PolicyServer with the model swapped for
"move joint1 by half a degree per step"."""
import socket
import threading
import time
from concurrent import futures

import pytest
import torch

from piper_smolvla.clients import JOINTS


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def fake_server():
    import grpc
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.helpers import TimedAction
    from lerobot.transport import services_pb2, services_pb2_grpc

    from piper_smolvla.serve import make_server_class
    import pickle

    Base = make_server_class()  # the name-resolving server the CLI runs
    seen = {"specs": [], "obs": []}

    class Fake(Base):
        def SendPolicyInstructions(self, request, context):  # noqa: N802
            spec = pickle.loads(request.data)
            seen["specs"].append(spec)
            self.lerobot_features = spec.lerobot_features
            self.actions_per_chunk = spec.actions_per_chunk
            self.device = "cpu"
            return services_pb2.Empty()

        def _predict_action_chunk(self, observation_t):
            raw = observation_t.get_observation()
            seen["obs"].append({k: (v.shape if hasattr(v, "shape") else v) for k, v in raw.items()})
            base = torch.tensor([float(raw[j]) for j in JOINTS])
            chunk = []
            for i in range(self.actions_per_chunk):
                a = base.clone()
                a[0] += 0.5 * (i + 1)
                chunk.append(a)
            self.last_processed_obs = observation_t
            return [TimedAction(timestamp=observation_t.get_timestamp() + i / 15, timestep=observation_t.get_timestep() + i,
                                action=a) for i, a in enumerate(chunk)]

    port = free_port()
    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    ps = Fake(PolicyServerConfig(host="127.0.0.1", port=port, fps=15, inference_latency=0.02, obs_queue_timeout=1))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(ps, srv)
    srv.add_insecure_port(f"127.0.0.1:{port}")
    srv.start()
    yield f"127.0.0.1:{port}", seen
    ps.stop()
    srv.stop(grace=0.5)


def options(sim, server, **kw):
    from piper_smolvla.run import RunOptions
    drv, cams = sim
    return RunOptions(server=server, policy="sim-policy", task="move joint1", driver_url=drv.url, camera_url=cams.url,
                      cameras=["wrist", "scene"], policy_device="cpu", actions_per_chunk=20, **kw)


def run(o, seconds):
    from piper_smolvla.run import client_config, plan_run, run_client
    pl = plan_run(o)
    return run_client(client_config(o, pl), seconds)


def test_dry_run_plans_but_never_streams(sim, fake_server):
    drv, _ = sim
    server, seen = fake_server
    rep = run(options(sim, server, dry_run=True), 3.0)
    assert rep["actions"] > 10 and rep["sent"] == 0
    assert "limit" in rep["stop_reason"]
    assert drv.posts["/api/stream"] == 0 and drv.posts["/api/speed"] == 0 and drv.posts["/api/release"] == 0
    spec = seen["specs"][0]
    assert spec.policy_type == "smolvla" and spec.pretrained_name_or_path == "sim-policy"
    assert spec.lerobot_features["observation.state"]["names"] == JOINTS
    obs = seen["obs"][0]
    assert obs["task"] == "move joint1" and obs["wrist"] == (480, 640, 3) and obs["scene"] == (480, 640, 3)


def test_live_run_moves_then_holds_and_releases(sim, fake_server):
    drv, _ = sim
    server, _ = fake_server
    j0 = drv.arm.joints[0]
    rep = run(options(sim, server), 3.0)
    assert rep["sent"] > 10 and drv.posts["/api/stream"] == rep["sent"] + 1   # + the hold at the end
    assert drv.posts["/api/release"] == 1 and drv.arm.owner is None
    assert drv.arm.joints[0] > j0 + 3                                         # it moved, a little
    assert rep["clamps"]["steps"] == rep["sent"]


def test_safety_stop_ends_the_run(sim, fake_server):
    from .conftest import post
    drv, _ = sim
    server, _ = fake_server
    threading.Timer(1.5, lambda: post(drv.url, "/sim/teach", {"on": True})).start()
    t = time.monotonic()
    rep = run(options(sim, server), 20.0)
    assert time.monotonic() - t < 10
    assert "drag-teach" in rep["stop_reason"]
    assert drv.posts["/api/release"] == 0                     # handed over: nothing that stiffens the arm


def test_unreachable_server(sim):
    with pytest.raises(RuntimeError, match="policy server"):
        run(options(sim, f"127.0.0.1:{free_port()}"), 2.0)
    drv, _ = sim
    assert drv.posts["/api/stream"] == 0
