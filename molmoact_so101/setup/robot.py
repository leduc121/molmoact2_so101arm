"""SO-101 follower-arm driver.

Camera capture deliberately lives in ``usb_camera.py``: this project uses two
generic UVC/V4L2 RGB cameras and has no RealSense or depth-camera dependency.
"""
import threading
import time

import numpy as np


JOINT_COUNT = 6
MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
_MAX_JOINT_DELTA = 10.0  # degrees per worker iteration — inner rate limiter


class FollowerArm:
    """Drives the SO-101 follower arm via LeRobot's SOFollower driver.

    All serial I/O is handled by a single background worker thread that
    alternates write / read on the half-duplex Feetech bus. Call set_target()
    and get_state() from any thread; they touch only lock-protected slots.
    """

    def __init__(self, port: str = "/dev/ttyACM0", simulate: bool = False):
        self.simulate = simulate
        self._target = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._state  = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._target_lock  = threading.Lock()
        self._state_lock   = threading.Lock()
        self._torque_lock  = threading.Lock()
        self._torque_desired = True
        self._stop   = threading.Event()
        self._thread = None

        if not simulate:
            try:
                from lerobot.robots.so_follower import SOFollower
                from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
                self.robot = SOFollower(SOFollowerRobotConfig(
                    port=port, id="so_follower", use_degrees=True
                ))
                self.robot.connect()
                print(f"[Follower] Connected on {port}")
                obs  = self.robot.get_observation()
                init = np.array([obs[f"{n}.pos"] for n in MOTOR_NAMES], dtype=np.float32)
                self._target = init.copy()
                self._state  = init.copy()
                self._thread = threading.Thread(target=self._worker_loop, daemon=True)
                self._thread.start()
            except Exception as e:
                print(f"[Follower] Could not connect: {e} — falling back to simulation")
                self.simulate = True
        else:
            print("[Follower] Simulation mode")

    def _worker_loop(self):
        torque_actual = True
        last_written  = self._target.copy()
        while not self._stop.is_set():
            with self._torque_lock:
                desired = self._torque_desired
            if desired != torque_actual:
                try:
                    if desired:
                        self.robot.bus.enable_torque()
                        print("[Follower] Torque enabled")
                    else:
                        self.robot.bus.disable_torque()
                        print("[Follower] Torque disabled")
                    torque_actual = desired
                except Exception as e:
                    print(f"[Follower] torque transition error: {e}")

            if torque_actual:
                with self._target_lock:
                    target = self._target.copy()
                delta = np.clip(target - last_written, -_MAX_JOINT_DELTA, _MAX_JOINT_DELTA)
                last_written = last_written + delta
                try:
                    action = {f"{n}.pos": float(last_written[i])
                              for i, n in enumerate(MOTOR_NAMES)}
                    self.robot.send_action(action)
                except Exception as e:
                    print(f"[Follower] send_action error: {e}")

            try:
                obs   = self.robot.get_observation()
                state = np.array([obs[f"{n}.pos"] for n in MOTOR_NAMES], dtype=np.float32)
                with self._state_lock:
                    self._state = state
            except Exception as e:
                print(f"[Follower] get_observation error: {e}")

    def set_target(self, target: np.ndarray):
        if self.simulate:
            with self._target_lock:
                delta = np.clip(target - self._target, -_MAX_JOINT_DELTA, _MAX_JOINT_DELTA)
                self._target = self._target + delta
            with self._state_lock:
                self._state = self._target.copy()
            return
        with self._target_lock:
            self._target = target.astype(np.float32, copy=True)

    def get_state(self) -> np.ndarray:
        with self._state_lock:
            return self._state.copy()

    def request_torque(self, on: bool):
        if self.simulate:
            return
        with self._torque_lock:
            self._torque_desired = on

    def disconnect(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.simulate:
            try:
                self.robot.bus.disable_torque()
            except Exception:
                pass
            self.robot.disconnect()
