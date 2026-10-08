"""GPU-only FastAPI inference service. It has no robot-control code."""
from __future__ import annotations
import os, time
import numpy as np
from PIL import Image
from fastapi import FastAPI, Header, HTTPException
from molmoact_so101.model.protocol import PolicyRequest, PolicyResponse, decode_jpeg

app = FastAPI(title="MolmoAct2 inference", docs_url=None, redoc_url=None)
_policy = None

def _authorized(value: str | None) -> bool:
    token = os.environ.get("MOLMOACT_API_TOKEN")
    return bool(token) and value == f"Bearer {token}"

def _get_policy():
    global _policy
    if _policy is None:
        from molmoact_so101.model.policy import MolmoActPolicy, REPO_ID
        dtype = os.environ.get("MOLMOACT_DTYPE", "bfloat16")
        _policy = MolmoActPolicy.from_pretrained(os.environ.get("MOLMOACT_MODEL_ID", REPO_ID),
            device=os.environ.get("MOLMOACT_DEVICE", "cuda"), dtype=dtype,
            apply_patches=dtype != "float32")
    return _policy

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.get("/readyz")
def readyz(authorization: str | None = Header(default=None)):
    if not _authorized(authorization): raise HTTPException(401, "unauthorized")
    try: _get_policy()
    except Exception as exc: raise HTTPException(503, f"model unavailable: {type(exc).__name__}")
    return {"ready": True}

@app.post("/v1/warmup")
def warmup(authorization: str | None = Header(default=None)):
    """Warm CUDA graphs without accepting observations from or controlling a robot."""
    if not _authorized(authorization): raise HTTPException(401, "unauthorized")
    try:
        policy = _get_policy()
        count = int(os.environ.get("MOLMOACT_WARMUP_REQUESTS", "3"))
        if count < 1 or count > 10: raise ValueError("MOLMOACT_WARMUP_REQUESTS must be 1..10")
        blank = Image.new("RGB", (640, 480))
        timings = []
        for _ in range(count):
            started = time.perf_counter()
            policy.predict_chunk([blank, blank], np.zeros(6, dtype=np.float32), "move the arm", cuda_graph=True)
            timings.append(round((time.perf_counter() - started) * 1000.0, 1))
        return {"warmed": count, "inference_ms": timings}
    except ValueError as exc: raise HTTPException(422, str(exc))
    except Exception as exc: raise HTTPException(503, f"warmup failed: {type(exc).__name__}")

@app.post("/v1/predict")
def predict(payload: dict, authorization: str | None = Header(default=None)):
    if not _authorized(authorization): raise HTTPException(401, "unauthorized")
    try:
        request = PolicyRequest.from_wire(payload)
        images = [decode_jpeg(v) for v in request.images_jpeg]
        started = time.perf_counter()
        actions = _get_policy().predict_chunk(images, request.state, request.prompt,
            num_steps=request.num_steps, cuda_graph=request.cuda_graph)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return PolicyResponse(request.request_id, actions, time.monotonic(), elapsed_ms).to_wire()
    except ValueError as exc: raise HTTPException(422, str(exc))
    except Exception as exc: raise HTTPException(503, f"inference failed: {type(exc).__name__}")
