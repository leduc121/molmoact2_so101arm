import time
import unittest
import numpy as np
from PIL import Image
from molmoact_so101.model.protocol import PolicyRequest, PolicyResponse, decode_jpeg, encode_jpeg, validate_actions

class ProtocolTests(unittest.TestCase):
    def request(self):
        image = encode_jpeg(Image.new("RGB", (4, 4)))
        return PolicyRequest("request-1", "task", np.zeros(6), time.monotonic(), [image, image])
    def test_wire_roundtrip(self): self.assertEqual(PolicyRequest.from_wire(self.request().to_wire()).slots, ("head", "side"))
    def test_reject_bad_actions_and_state(self):
        with self.assertRaises(ValueError): validate_actions(np.zeros((1, 5)))
        with self.assertRaises(ValueError): validate_actions(np.array([[np.nan] * 6]))
        with self.assertRaises(ValueError): PolicyRequest("", "x", np.zeros(6), 1, []).validate()
    def test_jpeg_roundtrip(self): self.assertEqual(decode_jpeg(encode_jpeg(Image.new("RGB", (3, 2)))).size, (3, 2))
    def test_response_rejects_nonfinite_actions(self):
        with self.assertRaises(ValueError): PolicyResponse("request-1", np.full((1, 6), np.inf), time.monotonic()).validate()
    def test_response_roundtrips_inference_timing(self):
        value = PolicyResponse("request-1", np.zeros((2, 6)), time.monotonic(), 12.5)
        self.assertEqual(PolicyResponse.from_wire(value.to_wire()).server_inference_ms, 12.5)
