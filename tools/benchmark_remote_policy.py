#!/usr/bin/env python3
"""Benchmark remote MolmoAct2 latency without camera or robot hardware."""
from __future__ import annotations
import argparse, os, sys, time, uuid
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from molmoact_so101.model.backends import RemotePolicyBackend, token_from_env
from molmoact_so101.model.protocol import PolicyRequest, encode_jpeg

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-url", required=True); p.add_argument("--head-image", required=True); p.add_argument("--side-image", required=True)
    p.add_argument("--prompt", default="pick up the cube and place it inside the white plastic bowl")
    p.add_argument("--state", default="0,0,0,0,0,0", help="six model-frame joint values; zeros are fine for latency-only tests")
    p.add_argument("--requests", type=int, default=20); p.add_argument("--timeout", type=float, default=30.0); p.add_argument("--api-token-env", default="MOLMOACT_API_TOKEN"); p.add_argument("--no-cuda-graph", action="store_true")
    a = p.parse_args()
    if a.requests < 2: p.error("--requests must be at least 2 to report p50/p95")
    state = np.fromstring(a.state, sep=",", dtype=np.float32)
    if state.shape != (6,) or not np.isfinite(state).all(): p.error("--state must contain six finite comma-separated values")
    images = [encode_jpeg(Image.open(path).convert("RGB")) for path in (a.head_image, a.side_image)]
    backend = RemotePolicyBackend(a.server_url, token_from_env(a.api_token_env), a.timeout); rtts=[]; model_times=[]
    for index in range(a.requests):
        request = PolicyRequest(str(uuid.uuid4()), a.prompt, state, time.monotonic(), images, ("head", "side"), cuda_graph=not a.no_cuda_graph)
        started = time.perf_counter(); actions = backend.predict(request); rtt = (time.perf_counter() - started) * 1000.0; rtts.append(rtt)
        model_ms = backend.last_diagnostics.get("policy_ms")
        if model_ms is not None: model_times.append(float(model_ms))
        print(f"{index + 1:02d}/{a.requests}: rtt={rtt:.1f}ms server_model={model_ms}ms shape={actions.shape}")
    q = lambda values, p: float(np.percentile(np.asarray(values, dtype=np.float64), p))
    print(f"RTT ms: p50={q(rtts,50):.1f} p95={q(rtts,95):.1f} max={max(rtts):.1f}")
    if model_times: print(f"Model ms: p50={q(model_times,50):.1f} p95={q(model_times,95):.1f}")
    print("No camera, robot, or motor command was initialized by this benchmark.")

if __name__ == "__main__": main()
