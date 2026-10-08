import unittest
from molmoact_so101.setup.usb_camera import CameraConfig, camera_source

class CameraConfigTests(unittest.TestCase):
    def test_camera_source(self):
        self.assertEqual(camera_source("4"), 4)
        self.assertEqual(camera_source("/dev/v4l/by-id/cam"), "/dev/v4l/by-id/cam")
        with self.assertRaises(ValueError): camera_source("camera")
    def test_config_validation(self):
        with self.assertRaises(ValueError): CameraConfig(0, rotation=45)
