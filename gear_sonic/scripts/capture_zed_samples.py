"""Capture a few synchronized color/depth frames directly from a ZED camera.

Example:
    python gear_sonic/scripts/capture_zed_samples.py \
        --resolution HD720 --fps 30 --num-frames 5

The script saves a lossless color PNG, a raw uint16 depth PNG (millimetres),
a colorized depth preview, and metadata describing the camera and frames.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np


def _enum_value(enum_class, name: str, option_name: str):
    value = getattr(enum_class, name.upper(), None)
    if value is None:
        available = sorted(name for name in dir(enum_class) if name.isupper())
        raise ValueError(
            f"Unsupported {option_name} '{name}'. Available values: {', '.join(available)}"
        )
    return value


def _depth_preview(depth_mm: np.ndarray, max_depth_mm: int) -> np.ndarray:
    """Convert raw millimetre depth to a BGR visualization."""
    valid = (depth_mm > 0) & (depth_mm < max_depth_mm)
    normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        depth_mm[valid].astype(np.float32) * (255.0 / max_depth_mm), 0, 255
    ).astype(np.uint8)
    preview = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def _grab_or_raise(camera, runtime_params, success_code, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status = None
    while time.monotonic() < deadline:
        last_status = camera.grab(runtime_params)
        if last_status == success_code:
            return
        time.sleep(0.01)
    raise RuntimeError(f"No ZED frame received within {timeout_seconds:.0f}s: {last_status}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save synchronized ZED color and raw depth sample frames."
    )
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frame pairs to save.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: zed_samples/<timestamp>).",
    )
    parser.add_argument(
        "--resolution",
        default="HD1080",
        help="ZED resolution enum, for example HD2K, HD1080, HD720, or VGA.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Camera capture FPS.")
    parser.add_argument(
        "--depth-mode",
        default="NEURAL_LIGHT",
        help="ZED depth mode, for example NEURAL_LIGHT, NEURAL, or NEURAL_PLUS.",
    )
    parser.add_argument("--serial", type=int, default=None, help="Optional ZED serial number.")
    parser.add_argument(
        "--warmup-frames", type=int, default=30, help="Successful frames to discard before saving."
    )
    parser.add_argument(
        "--interval", type=float, default=0.5, help="Seconds between saved frame pairs."
    )
    parser.add_argument(
        "--preview-max-mm",
        type=int,
        default=5000,
        help="Maximum depth represented in the colorized preview.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be greater than zero")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames cannot be negative")
    if args.interval < 0:
        raise ValueError("--interval cannot be negative")
    if args.preview_max_mm <= 0:
        raise ValueError("--preview-max-mm must be greater than zero")

    # Import after argument parsing so ``--help`` also works outside the ZED environment.
    import pyzed.sl as sl

    resolution = _enum_value(sl.RESOLUTION, args.resolution, "resolution")
    depth_mode = _enum_value(sl.DEPTH_MODE, args.depth_mode, "depth mode")
    output_dir = args.output or Path("zed_samples") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution
    init_params.camera_fps = args.fps
    init_params.depth_mode = depth_mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init_params.sdk_verbose = 1
    init_params.open_timeout_sec = 10.0
    if args.serial is not None:
        init_params.set_from_serial_number(args.serial)

    camera = sl.Camera()
    print(
        f"Opening ZED: resolution={args.resolution.upper()}, fps={args.fps}, "
        f"depth_mode={args.depth_mode.upper()}"
    )
    status = camera.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {status}")

    color_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    try:
        camera_info = camera.get_camera_information()
        camera_config = camera_info.camera_configuration
        left = camera_config.calibration_parameters.left_cam

        print(
            f"Opened serial={camera_info.serial_number}, model={camera_info.camera_model}, "
            f"actual_resolution={camera_config.resolution.width}x{camera_config.resolution.height}"
        )
        print(f"Warming up with {args.warmup_frames} frames...")
        warmed_up = 0
        while warmed_up < args.warmup_frames:
            _grab_or_raise(camera, runtime_params, sl.ERROR_CODE.SUCCESS)
            warmed_up += 1

        frames: list[dict[str, object]] = []
        for index in range(args.num_frames):
            _grab_or_raise(camera, runtime_params, sl.ERROR_CODE.SUCCESS)

            camera.retrieve_image(color_mat, sl.VIEW.LEFT, sl.MEM.CPU)
            camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH_U16_MM, sl.MEM.CPU)

            color_bgra = np.array(color_mat.get_data(), copy=True)
            depth_mm = np.array(depth_mat.get_data(), dtype=np.uint16, copy=True)
            if depth_mm.ndim == 3:
                depth_mm = depth_mm[:, :, 0]

            # ZED VIEW.LEFT is BGRA; OpenCV writes BGR correctly after alpha is removed.
            color_bgr = np.ascontiguousarray(color_bgra[:, :, :3])
            depth_mm = np.ascontiguousarray(depth_mm)
            preview = _depth_preview(depth_mm, args.preview_max_mm)

            stem = f"frame_{index:03d}"
            color_name = f"{stem}_color.png"
            depth_name = f"{stem}_depth_mm.png"
            preview_name = f"{stem}_depth_preview.png"
            _write_image(output_dir / color_name, color_bgr)
            _write_image(output_dir / depth_name, depth_mm)
            _write_image(output_dir / preview_name, preview)

            timestamp_ns = camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            valid_depth = depth_mm[(depth_mm > 0) & (depth_mm < 65535)]
            frames.append(
                {
                    "index": index,
                    "camera_timestamp_ns": int(timestamp_ns),
                    "color": color_name,
                    "depth_mm": depth_name,
                    "depth_preview": preview_name,
                    "valid_depth_min_mm": int(valid_depth.min()) if valid_depth.size else None,
                    "valid_depth_max_mm": int(valid_depth.max()) if valid_depth.size else None,
                }
            )
            print(f"Saved {index + 1}/{args.num_frames}: {color_name}, {depth_name}")

            if index + 1 < args.num_frames and args.interval:
                time.sleep(args.interval)

        metadata = {
            "camera": {
                "serial_number": int(camera_info.serial_number),
                "model": str(camera_info.camera_model),
                "resolution": {
                    "width": int(camera_config.resolution.width),
                    "height": int(camera_config.resolution.height),
                },
                "requested_fps": args.fps,
                "requested_resolution": args.resolution.upper(),
                "depth_mode": args.depth_mode.upper(),
                "depth_unit": "millimetre",
                "left_intrinsics": {
                    "fx": float(left.fx),
                    "fy": float(left.fy),
                    "cx": float(left.cx),
                    "cy": float(left.cy),
                    "h_fov_deg": float(left.h_fov),
                    "v_fov_deg": float(left.v_fov),
                    "distortion_coefficients": [float(value) for value in left.disto],
                },
            },
            "frames": frames,
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Done. Files saved to: {output_dir.resolve()}")
    finally:
        camera.close()


if __name__ == "__main__":
    main()
