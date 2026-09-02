"""Stereolabs ZED camera driver.

Requires the ``pyzed`` SDK (ZED SDK 5.x) — install the ``.run`` from
https://www.stereolabs.com/developers/release and make sure ``pyzed`` is
importable in the camera virtual environment.

See https://www.stereolabs.com/docs for API reference.
"""

import time
from typing import Any

import numpy as np
import pyzed.sl as sl
try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


class ZEDConfig:
    """Configuration for the ZED camera."""

    resolution: Any = sl.RESOLUTION.HD720  # pyzed.sl.RESOLUTION.HD720 by default
    depth_mode: Any = sl.DEPTH_MODE.NEURAL_LIGHT  # pyzed.sl.DEPTH_MODE.PERFORMANCE by default
    fps: int = 30
    mount_position: str = CameraMountPosition.EGO_VIEW.value
    sdk_verbose: int = 1
    depth_minimum_distance: float = -1.0  # -1 = auto
    depth_maximum_distance: float = -1.0  # -1 = auto


class ZEDSensor(Sensor, SensorServer):
    """Sensor for Stereolabs ZED depth cameras.

    Publishes left RGB (``ego_view``) and depth in millimetre uint16
    (``ego_view_depth``), matching the contract established by
    :class:`RealSenseSensor` so existing exporters / viewers work unchanged.
    """

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: ZEDConfig = ZEDConfig(),
        device_id: str | None = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):


        self._sl = sl

        # ---- resolution ----------------------------------------------------
        if config.resolution is not None:
            resolution = config.resolution
        else:
            resolution = sl.RESOLUTION.HD720

        # ---- depth mode ----------------------------------------------------
        if config.depth_mode is not None:
            depth_mode = config.depth_mode
        else:
            depth_mode = sl.DEPTH_MODE.PERFORMANCE

        # ---- init parameters -----------------------------------------------
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.camera_fps = config.fps
        init_params.depth_mode = depth_mode
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        init_params.sdk_verbose = config.sdk_verbose
        init_params.depth_minimum_distance = config.depth_minimum_distance
        init_params.depth_maximum_distance = config.depth_maximum_distance
        init_params.camera_disable_self_calib = False
        init_params.depth_stabilization = 30
        init_params.open_timeout_sec = 10.0

        if device_id is not None:
            try:
                serial_number = int(device_id)
                init_params.set_from_serial_number(serial_number)
                print(f"[ZED] Requested serial number: {serial_number}")
            except ValueError:
                print(
                    f"[ZED] device_id '{device_id}' is not a serial number; "
                    f"falling back to first available device"
                )

        # ---- open camera ---------------------------------------------------
        self._camera = sl.Camera()
        status = self._camera.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")

        # ---- calibration ---------------------------------------------------
        camera_info = self._camera.get_camera_information()
        cam_config = camera_info.camera_configuration
        calib = cam_config.calibration_parameters
        left = calib.left_cam

        self._rgb_width = cam_config.resolution.width
        self._rgb_height = cam_config.resolution.height

        self.calibration: dict[str, Any] = {
            "schema_version": 1,
            "camera_type": "zed",
            "serial_number": camera_info.serial_number,
            "camera_model": str(camera_info.camera_model),
            "color_intrinsics": {
                "width": self._rgb_width,
                "height": self._rgb_height,
                "fx": left.fx,
                "fy": left.fy,
                "ppx": left.cx,
                "ppy": left.cy,
                "distortion_model": str(left.disto),
                "distortion_coefficients": list(left.disto),
            },
            "depth_intrinsics": {
                "width": self._rgb_width,
                "height": self._rgb_height,
                "fx": left.fx,
                "fy": left.fy,
                "ppx": left.cx,
                "ppy": left.cy,
                "distortion_model": str(left.disto),
                "distortion_coefficients": list(left.disto),
            },
        }
        print("[ZED] Publishing left RGB + depth (uint16 mm) with calibration metadata")

        # ---- Mat buffers (reused every frame) ------------------------------
        self._mat_rgb = sl.Mat()
        self._mat_depth = sl.Mat()

        self._runtime = sl.RuntimeParameters()

        self._zed_config = config
        self._run_as_server = run_as_server
        self.mount_position = mount_position

        if self._run_as_server:
            self.start_server(port)

        print(
            f"Done initializing ZED sensor: "
            f"serial={camera_info.serial_number}, "
            f"resolution={self._rgb_width}x{self._rgb_height}, "
            f"fps={config.fps}"
        )

    # ------------------------------------------------------------------
    def read(self) -> dict[str, Any] | None:
        """Grab one frame pair (left RGB + depth) and return it in the
        canonical per-camera dict format."""
        sl = self._sl

        if self._camera.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            print("WARNING! ZED grab failed")
            return None

        try:
            self._camera.retrieve_image(
                self._mat_rgb, sl.VIEW.LEFT, sl.MEM.CPU
            )
            self._camera.retrieve_measure(
                self._mat_depth, sl.MEASURE.DEPTH_U16_MM, sl.MEM.CPU
            )
        except Exception as e:
            print(f"ERROR! Failed to retrieve ZED images: {e}")
            return None

        rgb = self._mat_rgb.get_data()
        depth = self._mat_depth.get_data()

        # ZED returns BGRA (4 channels) by default — drop alpha
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        # depth may be a * x 1 shape — squeeze to 2D
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]

        if rgb.size == 0 or depth.size == 0:
            print("WARNING! Empty ZED image")
            return None

        now = time.time()
        return {
            "timestamps": {
                self.mount_position: now,
                f"{self.mount_position}_depth": now,
            },
            "images": {
                self.mount_position: np.ascontiguousarray(rgb),
                f"{self.mount_position}_depth": np.ascontiguousarray(depth),
            },
            "metadata": {"zed_calibration": self.calibration},
        }

    # ------------------------------------------------------------------
    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(
            timestamps=data["timestamps"],
            images=data["images"],
            metadata=data["metadata"],
        )
        return serialized_msg.serialize()

    # ------------------------------------------------------------------
    def observation_space(self):
        if gym is None:
            return None
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._rgb_height, self._rgb_width, 3),
                    dtype=np.uint8,
                ),
                "depth_image": gym.spaces.Box(
                    low=0,
                    high=65535,
                    shape=(self._rgb_height, self._rgb_width, 1),
                    dtype=np.uint16,
                ),
            }
        )

    # ------------------------------------------------------------------
    def close(self):
        if self._run_as_server:
            self.stop_server()
        self._camera.close()

    # ------------------------------------------------------------------
    def run_server(self):
        if not self._run_as_server:
            raise ValueError("run_as_server must be True to call run_server()")
        while True:
            read_result = self.read()
            if read_result is None:
                continue
            self.send_message({self.mount_position: self.serialize(read_result)})
