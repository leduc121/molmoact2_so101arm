"""Versioned, validated wire contract for local and remote policy backends."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

PROTOCOL_VERSION = "1"
JOINT_COUNT = 6


def validate_state(state: np.ndarray) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32)
    if arr.shape != (JOINT_COUNT,) or not np.isfinite(arr).all():
        raise ValueError(f"state must be finite shape ({JOINT_COUNT},), got {arr.shape}")
    return arr


def validate_actions(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != JOINT_COUNT or not np.isfinite(arr).all():
        raise ValueError(f"actions must be finite shape (T, {JOINT_COUNT}), got {arr.shape}")
    return arr


def encode_jpeg(image: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_jpeg(encoded: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB")
    except Exception as exc:
        raise ValueError("invalid JPEG image payload") from exc


@dataclass(frozen=True)
class PolicyRequest:
    request_id: str
    prompt: str
    state: np.ndarray
    observation_monotonic: float
    images_jpeg: Sequence[str]
    slots: tuple[str, str] = ("head", "side")
    num_steps: int = 10
    cuda_graph: bool = False
    protocol_version: str = PROTOCOL_VERSION

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION or not self.request_id or len(self.images_jpeg) != 2:
            raise ValueError("request needs protocol version, id, and exactly two images")
        if len(self.slots) != 2 or not self.prompt or not np.isfinite(self.observation_monotonic):
            raise ValueError("invalid request metadata")
        validate_state(self.state)

    def to_wire(self) -> dict:
        self.validate()
        return {"protocol_version": self.protocol_version, "request_id": self.request_id,
                "prompt": self.prompt, "state": self.state.tolist(),
                "observation_monotonic": self.observation_monotonic,
                "images_jpeg": list(self.images_jpeg), "slots": list(self.slots),
                "num_steps": self.num_steps, "cuda_graph": self.cuda_graph}

    @classmethod
    def from_wire(cls, data: dict) -> "PolicyRequest":
        result = cls(data["request_id"], data["prompt"], np.asarray(data["state"], dtype=np.float32),
                     float(data["observation_monotonic"]), data["images_jpeg"], tuple(data.get("slots", [])),
                     int(data.get("num_steps", 10)), bool(data.get("cuda_graph", False)),
                     data.get("protocol_version", ""))
        result.validate()
        return result


@dataclass(frozen=True)
class PolicyResponse:
    request_id: str
    actions: np.ndarray
    server_monotonic: float
    protocol_version: str = PROTOCOL_VERSION

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION or not self.request_id or not np.isfinite(self.server_monotonic):
            raise ValueError("invalid response metadata")
        validate_actions(self.actions)

    def to_wire(self) -> dict:
        self.validate()
        return {"protocol_version": self.protocol_version, "request_id": self.request_id,
                "actions": self.actions.tolist(), "server_monotonic": self.server_monotonic}

    @classmethod
    def from_wire(cls, data: dict) -> "PolicyResponse":
        result = cls(data["request_id"], np.asarray(data["actions"], dtype=np.float32),
                     float(data["server_monotonic"]), data.get("protocol_version", ""))
        result.validate()
        return result
