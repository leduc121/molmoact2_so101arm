"""GPU-only FastAPI inference service. It has no robot-control code."""
from __future__ import annotations
import os, time
import numpy as np
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
        dtype = os.environ.get("MOLMOACT_DTYPE", "float16")
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

@app.post("/v1/predict")
def predict(payload: dict, authorization: str | None = Header(default=None)):
    if not _authorized(authorization): raise HTTPException(401, "unauthorized")
    try:
        request = PolicyRequest.from_wire(payload)
        images = [decode_jpeg(v) for v in request.images_jpeg]
        actions = _get_policy().predict_chunk(images, request.state, request.prompt,
            num_steps=request.num_steps, cuda_graph=request.cuda_graph)
        return PolicyResponse(request.request_id, actions, time.monotonic()).to_wire()
    except ValueError as exc: raise HTTPException(422, str(exc))
    except Exception as exc: raise HTTPException(503, f"inference failed: {type(exc).__name__}")
