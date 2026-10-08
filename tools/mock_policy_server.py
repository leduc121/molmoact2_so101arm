#!/usr/bin/env python3
"""Authenticated CPU-only mock of the remote policy API; never loads a model."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from molmoact_so101.model.protocol import PolicyRequest, PolicyResponse


class Handler(BaseHTTPRequestHandler):
    token = ""

    def log_message(self, *_args):
        pass  # Do not accidentally print Authorization headers.

    def _json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {self.token}"

    def do_GET(self):
        if self.path == "/healthz":
            return self._json(HTTPStatus.OK, {"ok": True, "mock": True})
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/predict":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        if not self._authorized():
            return self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            request = PolicyRequest.from_wire(json.loads(self.rfile.read(size)))
            # Decode/validate images in the same way a real server does.
            from molmoact_so101.model.protocol import decode_jpeg
            [decode_jpeg(image) for image in request.images_jpeg]
            actions = np.tile(request.state, (2, 1)).astype(np.float32)
            self._json(HTTPStatus.OK, PolicyResponse(request.request_id, actions, time.monotonic()).to_wire())
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--token-env", default="MOLMOACT_API_TOKEN")
    args = parser.parse_args()
    Handler.token = os.environ.get(args.token_env, "")
    if not Handler.token:
        parser.error(f"set {args.token_env} before starting mock server")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock policy server listening at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
