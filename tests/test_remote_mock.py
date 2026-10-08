import sys
import io
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch
import numpy as np
from PIL import Image
from molmoact_so101.model.backends import RemotePolicyBackend
from molmoact_so101.model.protocol import PolicyRequest, encode_jpeg
from tools.mock_policy_server import Handler

class RemoteMockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Handler.token = "test-token"; cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(timeout=1)
    def request(self, request_id="id-1"):
        image = encode_jpeg(Image.new("RGB", (8, 8)))
        return PolicyRequest(request_id, "mock task", np.zeros(6), time.monotonic(), [image, image])
    def test_authenticated_roundtrip(self):
        actions = RemotePolicyBackend(self.base_url, "test-token", 2).predict(self.request())
        self.assertEqual(actions.shape, (2, 6)); self.assertTrue(np.isfinite(actions).all())
    def test_bad_token_is_rejected(self):
        with self.assertRaises(RuntimeError): RemotePolicyBackend(self.base_url, "wrong", 2).predict(self.request())
    def test_cli_dry_run_default_and_remote_url_requirement(self):
        import inference
        argv = ["inference.py", "--prompt", "x", "--head-cam", "/dev/video4", "--side-cam", "/dev/video2"]
        with patch.object(sys, "argv", argv): self.assertTrue(inference.args().dry_run)
        with patch.object(sys, "argv", argv + ["--policy-backend", "remote"]), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit): inference.args()
