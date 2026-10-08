"""Local and HTTP policy backends; neither backend controls robot hardware."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from .protocol import PolicyRequest, PolicyResponse, decode_jpeg, validate_actions


class PolicyBackend(ABC):
    @abstractmethod
    def predict(self, request: PolicyRequest) -> np.ndarray: ...


class LocalPolicyBackend(PolicyBackend):
    def __init__(self, policy): self.policy = policy
    def predict(self, request: PolicyRequest) -> np.ndarray:
        request.validate()
        images = [decode_jpeg(value) for value in request.images_jpeg]
        return validate_actions(self.policy.predict_chunk(images=images, state=request.state,
                                prompt=request.prompt, num_steps=request.num_steps,
                                cuda_graph=request.cuda_graph))


class RemotePolicyBackend(PolicyBackend):
    """A stateless client: every request is fresh and responses are ID checked."""
    def __init__(self, server_url: str, token: str | None, timeout: float):
        if not server_url.startswith(("https://", "http://")):
            raise ValueError("server URL must start with http:// or https://")
        self.url, self.token, self.timeout = server_url.rstrip("/") + "/v1/predict", token, timeout
        self._last_response_server_time = -float("inf")

    def predict(self, request: PolicyRequest) -> np.ndarray:
        request.validate()
        headers = {"Content-Type": "application/json"}
        if self.token: headers["Authorization"] = f"Bearer {self.token}"
        raw = json.dumps(request.to_wire()).encode("utf-8")
        try:
            with urllib.request.urlopen(urllib.request.Request(self.url, raw, headers), timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            raise RuntimeError(f"remote inference failed: {exc}") from exc
        parsed = PolicyResponse.from_wire(data)
        if parsed.request_id != request.request_id:
            raise RuntimeError("remote response request_id mismatch")
        if parsed.server_monotonic <= self._last_response_server_time:
            raise RuntimeError("duplicate or out-of-order remote response")
        self._last_response_server_time = parsed.server_monotonic
        return validate_actions(parsed.actions)


def token_from_env(name: str | None) -> str | None:
    return os.environ.get(name) if name else None
