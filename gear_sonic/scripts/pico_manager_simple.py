# Pico SMPL stream server for body tracking visualization

"""

# Recommended Command Line Arguments:
    # With VR3 PT visualization (by --vis_vr3pt) and optional SMPL body visualization (by --vis_smpl)
    # If you want to enable waist tracking in the VR3 PT visualization, please add --waist_tracking
    python pico_manager_thread_server.py --manager \
        --vis_vr3pt --vis_smpl \
        --waist_tracking

    # VR3 PT visualization only (without SMPL body) — lower latency
    python pico_manager_thread_server.py --manager --vis_vr3pt

# DEBUG VR3 PT VISUALIZATION:
    # A standalone test mode that captures one live frame and visualizes it.
    python pico_manager_thread_server.py --vr3pt_live

# TIMING COMPARISON:
    # The visualizer automatically reports timing every 5 seconds when running:
    #   [Vis Timing] vr3pt: X.XXms | smpl: X.XXms | render: X.XXms | vr3pt_only: X.XXms | both(vr3pt+smpl): X.XXms

"""

from collections import defaultdict, deque
from enum import Enum, IntEnum
import os
import subprocess
import threading
import time

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation as sRot
import torch
import zmq

from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)

try:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        build_command_message,
        build_planner_message,
        pack_pose_message,
    )
except ImportError:

    def build_command_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_command_message unavailable")

    def build_planner_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_planner_message unavailable")

    def pack_pose_message(*args, **kwargs) -> bytes:
        raise RuntimeError("pack_pose_message unavailable")


try:
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
except ImportError:
    print("Warning: gear_sonic.isaac_utils.rotations not available.")
    remove_smpl_base_rot = None
    smpl_root_ytoz_up = None

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

try:
    from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
        G1GripperInverseKinematicsSolver,
    )
except ImportError:
    print("Warning: G1GripperInverseKinematicsSolver not available.")
    G1GripperInverseKinematicsSolver = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import VR3PtPoseVisualizer
except ImportError:
    print("Warning: VR3PtPoseVisualizer not available (pyvista may not be installed).")
    VR3PtPoseVisualizer = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import get_g1_key_frame_poses
except ImportError:
    print("Warning: get_g1_key_frame_poses not available (pyvista may not be installed).")
    get_g1_key_frame_poses = None


class LocomotionMode(IntEnum):
    """Locomotion mode enum — simplified to 3 modes."""

    IDLE = 0
    SLOW_WALK = 1
    IDLE_SQUAT = 4


class StreamMode(Enum):
    """Stream mode enum — simplified: PLANNER + VR_3PT only."""

    OFF = 0
    PLANNER = 2          # value 2 preserved for data exporter compatibility
    PLANNER_VR_3PT = 5   # value 5 preserved for data exporter compatibility


### Parse 3 point pose from SMPL
#
# OFFSETS: Rotation corrections applied to each keypoint to align SMPL joint frames
# with the desired robot/visualization coordinate convention.
#
# Index mapping (based on [0, 22, 23, 12].index(joint_id)):
#   - OFFSETS[0]: Root/Pelvis (joint 0)
#   - OFFSETS[1]: Left Wrist (joint 22)
#   - OFFSETS[2]: Right Wrist (joint 23)
#   - OFFSETS[3]: Neck (joint 12) - more stable than Head (joint 15) for body tracking
#
# Scipy euler rotation convention:
#   - Lowercase "xyz" = EXTRINSIC rotations (about the FIXED/ORIGINAL frame's axes)
#   - Uppercase "XYZ" = INTRINSIC rotations (about the ROTATING body's axes)
#
# For EXTRINSIC "xyz" with angles [a, b, c]:
#   All rotations are about the ORIGINAL frame's axes (before any rotation):
#     R_total = R_z(c) @ R_y(b) @ R_x(a)   (matrix multiplication order)
#   Applied as: first rotate 'a' about original X, then 'b' about original Y, then 'c' about original Z
#
# For INTRINSIC "XYZ" with angles [a, b, c]:
#   Each rotation is about the CURRENT (rotated) frame's axis:
#     R_total = R_x(a) @ R_y(b) @ R_z(c)   (matrix multiplication order)
#   Applied as: first rotate 'a' about X, then 'b' about NEW Y, then 'c' about NEW Z
#
OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Root: yaw -90° about fixed Z
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),  # L-Wrist: roll +90° about fixed X
    sRot.from_euler(
        "xyz", [-90, 0, 180], degrees=True
    ),  # R-Wrist: roll -90° about fixed X, then yaw 180° about fixed Z
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Neck: yaw -90° about fixed Z
]


def _compute_rel_transform(pose, world_frame, scalar_first=True):
    """
    Transform a pose from Unity coordinate frame to robot coordinate frame.

    Args:
        pose: np.ndarray shape (7,) - [x, y, z, qx, qy, qz, qw] in Unity frame
        world_frame: np.ndarray shape (7,) - reference frame to compute relative transform
        scalar_first: bool - if True, quaternion is [qw, qx, qy, qz]; if False, [qx, qy, qz, qw]

    Returns:
        rel_pos: np.ndarray (3,) - position in robot frame
        rel_rot: np.ndarray (4,) - quaternion [qw, qx, qy, qz] in robot frame

    Coordinate transform matrix Q converts Unity (Y-up, left-handed) to Robot (Z-up, right-handed):
        Unity:  X-right, Y-up, Z-forward
        Robot:  X-forward, Y-left, Z-up
    """
    world_frame = world_frame.copy()

    # Q transforms Unity coordinates to Robot coordinates
    # Unity [x, y, z] -> Robot [-x, z, y]
    Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
    pose[:3] = Q @ pose[:3]
    world_frame[:3] = Q @ world_frame[:3]
    rot_base = sRot.from_quat(world_frame[3:], scalar_first=scalar_first).as_matrix()
    rot = sRot.from_quat(pose[3:], scalar_first=scalar_first).as_matrix()
    rel_rot = sRot.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    rel_pos = sRot.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
    return rel_pos, rel_rot.as_quat(scalar_first=True)


def _process_3pt_pose(smpl_pose_np):
    """
    Extract 3-point VR pose (L-Wrist, R-Wrist, Neck) from full SMPL body joint poses.

    NOTE: We use Neck (joint 12) instead of Head (joint 15) because:
      - Neck is more rigidly coupled to the torso
      - Head has high DoF (looking around) which doesn't reflect body pose
      - Neck provides more stable tracking for upper body orientation

    Args:
        smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints, each [x, y, z, qx, qy, qz, qw]
                      in Unity frame (scalar-last quaternion format)

    Returns:
        vr_3pt_pose: np.ndarray shape (3, 7) - 3 keypoints in robot frame
                     Each row is [x, y, z, qw, qx, qy, qz] (scalar-FIRST quaternion format)
                     Row 0: Left Wrist (SMPL joint 22)
                     Row 1: Right Wrist (SMPL joint 23)
                     Row 2: Neck (SMPL joint 12)

                     IMPORTANT: Positions and orientations are RELATIVE TO ROOT (pelvis).

    Processing Steps:
        1. Transform all 24 joints from Unity frame to robot frame
        2. Extract 4 keypoints: Root(0), L-Wrist(22), R-Wrist(23), Neck(12)
        3. Apply per-joint rotation OFFSETS to align joint frames
        4. Make L-Wrist, R-Wrist, Neck relative to Root (both position and orientation)
        5. Return only the 3 non-root keypoints

    Note: Position calibration (wrist offsets, neck kinematic chain) is done in
          ThreePointPose.apply_calibration() to ensure consistency with calibrated
          orientations.
    """

    # Defensive copy: _compute_rel_transform modifies pose[:3] in-place, which would
    # corrupt the caller's array (e.g. PicoReader._latest) and cause wrong results
    # if the same sample is processed more than once.
    smpl_pose_np = smpl_pose_np.copy()

    # =========================================================================
    # STEP 1: Transform all joints from Unity frame to robot frame
    # =========================================================================
    # Input: smpl_pose_np[i] = [x, y, z, qx, qy, qz, qw] in Unity frame (scalar-last)
    # Output: body_poses[i] = [x, y, z, qw, qx, qy, qz] in robot frame (scalar-first)
    body_poses = np.zeros((smpl_pose_np.shape[0], 7), dtype=np.float32)
    for i in range(smpl_pose_np.shape[0]):
        pos, orn = _compute_rel_transform(
            smpl_pose_np[i], [0, 0, 0, 0, 0, 0, 1], scalar_first=False
        )
        body_poses[i, :3] = pos  # Position in robot frame
        body_poses[i, 3:] = orn  # Quaternion [qw, qx, qy, qz] in robot frame

    # =========================================================================
    # STEP 2 & 3: Extract 4 keypoints and apply rotation OFFSETS
    # =========================================================================
    # We only care about these SMPL joint indices:
    #   - Joint 0:  Root/Pelvis (reference frame)
    #   - Joint 22: Left Wrist
    #   - Joint 23: Right Wrist
    #   - Joint 12: Neck (more stable than Head joint 15)
    #
    # kp_poses maps these to indices 0, 1, 2, 3 respectively
    positions = np.array([[p[0], p[1], p[2]] for p in body_poses])
    kp_poses = np.zeros((4, 7), dtype=np.float32)

    for i, pose in enumerate(body_poses):
        if i not in [0, 22, 23, 12]:
            continue  # Skip joints we don't care about

        pos = positions[i]

        # Map SMPL joint index to our keypoint index (0-3)
        # rel_i: 0=Root, 1=L-Wrist, 2=R-Wrist, 3=Neck
        rel_i = [0, 22, 23, 12].index(i)

        # Extract quaternion and apply rotation offset
        # pose[3:7] is [qw, qx, qy, qz] (scalar-first from _compute_rel_transform)
        quat = np.array([pose[3], pose[4], pose[5], pose[6]])

        # Apply offset: new_rotation = original_rotation * OFFSET
        # This post-multiplies the offset (intrinsic rotation)
        rot_quat = (sRot.from_quat(quat, scalar_first=True) * OFFSETS[rel_i]).as_quat(
            scalar_first=False
        )

        kp_poses[rel_i, 3:] = rot_quat  # Store as scalar-last temporarily for scipy compatibility
        kp_poses[rel_i, :3] = pos

    # =========================================================================
    # STEP 4: Make positions and orientations RELATIVE TO ROOT
    # =========================================================================
    # This transforms everything into the root's local coordinate frame.
    # After this step:
    #   - Root's position would be (0,0,0) and orientation identity (but we don't return root)
    #   - Other keypoints are expressed relative to root
    root_pos = kp_poses[0, :3].copy()
    root_quat = kp_poses[0, 3:].copy()  # Still scalar-last for scipy

    for i in range(1, 4):
        # Position: subtract root position, then rotate by inverse of root orientation
        kp_poses[i, :3] = sRot.from_quat(root_quat).inv().apply(kp_poses[i, :3] - root_pos)

        # Orientation: compute relative rotation (root_inv * keypoint_rot)
        # Result stored as scalar-FIRST [qw, qx, qy, qz]
        kp_poses[i, 3:] = (
            sRot.from_quat(root_quat).inv() * sRot.from_quat(kp_poses[i, 3:])
        ).as_quat(scalar_first=True)

    # =========================================================================
    # STEP 5: Return only L-Wrist, R-Wrist, Neck (skip Root)
    # =========================================================================
    # NOTE: Position and orientation calibration (including neck position via kinematic
    #       chain) is done in ThreePointPose.apply_calibration() to ensure consistency
    #       between calibrated orientation and computed neck position.
    # kp_poses[1:] = indices 1, 2, 3 = L-Wrist, R-Wrist, Neck
    # Each row: [x, y, z, qw, qx, qy, qz] relative to root, scalar-first quaternion
    return kp_poses[1:]

# =============================================================================
# VR 3-Point Pose Visualization Functions
# =============================================================================


def run_vr3pt_visualizer_test():
    """
    Standalone test for VR 3-point pose visualizer using PyVista.
    Run this to verify the reference frames are displayed correctly.
    """
    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Visualizer Test (PyVista)")
    print("=" * 60)
    print("\nExpected reference frames (all with RGB axes for XYZ):")
    print("  1. WHITE ball at origin (0, 0, 0) - World frame")
    print("  2. CYAN ball at (0, 0, 0.35) - Looking forward (identity)")
    print("  3. MAGENTA ball at (0, 0.4, 0.25) - Looking left (yaw +90°)")
    print("  4. YELLOW ball at (0.4, 0, 0.15) - Looking down (pitch +90°)")
    print("\nClose the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_static()


def run_vr3pt_live_visualizer():
    """
    Live visualizer for real VR 3-point pose data from Pico.
    Captures one frame from Pico and displays it alongside reference frames.
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use live visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Live Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Capturing VR 3-point pose...")

    # Capture body poses and compute vr_3pt_pose
    body_poses = xrt.get_body_joints_pose()
    body_poses_np = np.array(body_poses)

    # Process to get 3-point pose (L-Wrist, R-Wrist, Neck)
    vr_3pt_pose = _process_3pt_pose(body_poses_np)

    print(f"\nCaptured vr_3pt_pose shape: {vr_3pt_pose.shape}")
    print(f"  L-Wrist: pos={vr_3pt_pose[0, :3]}, quat_wxyz={vr_3pt_pose[0, 3:]}")
    print(f"  R-Wrist: pos={vr_3pt_pose[1, :3]}, quat_wxyz={vr_3pt_pose[1, 3:]}")
    print(f"  Neck:    pos={vr_3pt_pose[2, :3]}, quat_wxyz={vr_3pt_pose[2, 3:]}")

    print("\nDisplaying visualization...")
    print("Close the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_with_vr_pose(vr_3pt_pose)


def run_vr3pt_realtime_visualizer(update_hz: int = 10):
    """
    Real-time visualizer for VR 3-point pose data from Pico.
    Continuously updates the visualization with live data.

    Args:
        update_hz: Update rate in Hz (default 10)
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use realtime visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Real-time Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Starting real-time visualization...")
    print(f"Update rate: {update_hz} Hz")
    print("Close the window or press 'q' to exit.")
    print("=" * 60)

    # Use the VR3PtPoseVisualizer for real-time visualization with G1 robot
    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.create_realtime_plotter(interactive=True)

    try:
        while visualizer.is_open:
            # Get new data from Pico
            body_poses = xrt.get_body_joints_pose()
            body_poses_np = np.array(body_poses)
            vr_3pt_pose = _process_3pt_pose(body_poses_np)

            # Update visualization
            visualizer.update_vr_poses(vr_3pt_pose)
            visualizer.render()

            time.sleep(1.0 / update_hz)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        visualizer.close()

def generate_finger_data(hand: str, trigger: float, grip: float) -> np.ndarray:
    """
    Generate finger position data from Pico controller button states.

    Args:
        hand: "left" or "right"
        trigger: Trigger button value (0-1)
        grip: Grip button value (0-1)

    Returns:
        Array of shape [25, 4, 4] representing fingertip positions
    """
    fingertips = np.zeros([25, 4, 4])

    thumb = 0
    middle = 10
    # Control thumb based on shoulder button state (index 4 is thumb tip)
    fingertips[4 + thumb, 0, 3] = 1.0  # open thumb
    if trigger > 0.5:
        fingertips[4 + middle, 0, 3] = 1.0  # close middle

    return fingertips

# Joystick deadzone threshold
JOYSTICK_DEADZONE = 0.15


class YawAccumulator:
    """Accumulates yaw heading angle based on joystick input."""

    def __init__(self, yaw_gain: float = 1.5, deadzone: float = JOYSTICK_DEADZONE):
        self.yaw_gain = yaw_gain
        self.deadzone = deadzone
        self.reset()

    def reset(self):
        """Reset facing direction to default (1,0,0)."""
        self.heading = [1.0, 0.0, 0.0]
        self.yaw_angle_rad = 0.0
        self.dyaw = 0.0
        print("YawAccumulator: reset yaw angle to 0.0")

    def yaw_angle(self) -> float:
        """Get current yaw angle in radians."""
        return self.yaw_angle_rad

    def yaw_angle_change(self) -> float:
        """Get current yaw angle change in radians."""
        return self.dyaw

    def update(self, rx: float, dt: float) -> list[float]:
        """
        Update facing direction based on right stick x-axis input.

        Args:
            rx: Right stick x-axis value (-1 to 1)
            dt: Time delta in seconds

        Returns:
            Facing direction as [x, y, 0.0]
        """
        self.dyaw = self.yaw_gain * (-rx) * dt
        if abs(rx) >= self.deadzone:
            self.yaw_angle_rad += self.dyaw
            self.heading = [np.cos(self.yaw_angle_rad), np.sin(self.yaw_angle_rad), 0.0]
        return self.heading

def init_hand_ik_solvers():
    """Initialize hand IK solvers if available."""
    if G1GripperInverseKinematicsSolver is not None:
        left_solver = G1GripperInverseKinematicsSolver(side="left")
        right_solver = G1GripperInverseKinematicsSolver(side="right")
        print("Hand IK solvers initialized")
        return left_solver, right_solver
    print("Warning: Hand IK solvers not available")
    return None, None


def get_controller_inputs():
    """Fetch controller button/trigger states from XRoboToolkit."""
    left_trigger = xrt.get_left_trigger()
    right_trigger = xrt.get_right_trigger()
    left_grip = xrt.get_left_grip()
    right_grip = xrt.get_right_grip()
    left_menu_button = xrt.get_left_menu_button()
    return left_menu_button, left_trigger, right_trigger, left_grip, right_grip


def get_controller_axes():
    """Fetch joystick axes (lx, ly, rx, ry). Falls back to zeros if not available."""
    if xrt is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        left_axis = xrt.get_left_axis()  # expected [x, y]
        right_axis = xrt.get_right_axis()  # expected [x, y]
        lx = float(left_axis[0]) if len(left_axis) >= 1 else 0.0
        ly = float(left_axis[1]) if len(left_axis) >= 2 else 0.0
        rx = float(right_axis[0]) if len(right_axis) >= 1 else 0.0
        ry = float(right_axis[1]) if len(right_axis) >= 2 else 0.0
        return lx, ly, rx, ry
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def get_menu_buttons():
    """Fetch both menu buttons (left, right). Falls back to False if not available."""
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_menu_button")
    right = _safe_btn("get_right_menu_button")
    return left, right


def get_axis_clicks():
    """Fetch both axis click buttons (left, right). Falls back to False if not available."""
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_axis_click")
    right = _safe_btn("get_right_axis_click")
    return left, right


def get_face_buttons():
    """Fetch primary face buttons A and X. Returns (a_pressed, x_pressed)."""
    if xrt is None:
        return False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        x_pressed = bool(xrt.get_X_button())
        return a_pressed, x_pressed
    except Exception:
        return False, False


def get_abxy_buttons():
    """Fetch A,B,X,Y face buttons as booleans (a,b,x,y)."""
    if xrt is None:
        return False, False, False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        b_pressed = bool(xrt.get_B_button())
        x_pressed = bool(xrt.get_X_button())
        y_pressed = bool(xrt.get_Y_button())
        return a_pressed, b_pressed, x_pressed, y_pressed
    except Exception:
        return False, False, False, False


def compute_hand_joints_from_inputs(
    left_solver, right_solver, left_trigger, left_grip, right_trigger, right_grip
) -> tuple[np.ndarray, np.ndarray]:
    """Compute left/right hand joints using IK solvers, or zeros if unavailable."""
    if left_solver is not None and right_solver is not None:
        left_finger_data = generate_finger_data("left", left_trigger, left_grip)
        right_finger_data = generate_finger_data("right", right_trigger, right_grip)
        left_hand_joints = left_solver({"position": left_finger_data})
        right_hand_joints = right_solver({"position": right_finger_data})
    else:
        left_hand_joints = np.zeros((1, 7), dtype=np.float32)
        right_hand_joints = np.zeros((1, 7), dtype=np.float32)
    return left_hand_joints, right_hand_joints


class PicoReader:
    """
    Background reader that pulls Pico/XRT data as fast as possible and computes dt/FPS.
    """

    def __init__(self, max_queue_size: int = 15):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_t = None
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest = None
        self._lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            if not xrt.is_body_data_available():
                time.sleep(0.001)
                continue
            stamp_ns = xrt.get_time_stamp_ns()
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                time.sleep(0.000001)
                continue
            # Compute device-based dt/fps using timestamp deltas (ns -> s)
            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns
            t_realtime = time.time()
            t_monotonic = time.monotonic()
            try:
                body_poses = xrt.get_body_joints_pose()

                sample = {
                    "body_poses_np": np.array(body_poses),
                    "timestamp_realtime": t_realtime,
                    "timestamp_monotonic": t_monotonic,
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample
                now = time.time()
                if now - last_report >= 5.0:
                    print(
                        f"[PicoReader] dt_ts: {device_dt*1000.0:.2f} ms, fps: {self._fps_ema:.2f}"
                    )
                    last_report = now
            except Exception as e:
                print(f"[PicoReader] read error: {e}")


class ThreePointPose:
    """
    Encapsulates everything around calculating 3-point pose from SMPL input.

    This includes:
    - Processing SMPL poses to extract 3-point VR pose (L-Wrist, R-Wrist, Neck)
    - Calibration logic to align VR poses with G1 robot
    - Optional visualization of 3-point poses

    Calibration is done in two steps:
    1. Neck orientation: Captures initial neck orientation to align subsequent poses as upright
    2. Wrist positions: Aligns wrist positions to match G1 robot key frame positions
    """

    # Kinematic chain constants for neck position (matches VR3PtPoseVisualizer)
    TORSO_LINK_OFFSET_Z = 0.05  # meters from root to torso_link
    NECK_LINK_LENGTH = 0.35  # meters from torso_link to neck along neck's local Z

    def __init__(
        self,
        enable_vis_vr3pt: bool = False,
        with_g1_robot: bool = True,
        enable_waist_tracking: bool = False,
        enable_smpl_vis: bool = False,
        log_prefix: str = "ThreePointPose",
        robot_model=None,
    ):
        """
        Initialize 3-point pose processor.

        Args:
            enable_vis_vr3pt: Whether to enable VR 3pt pose visualization (requires display)
            with_g1_robot: Whether to include G1 robot in visualization
            enable_waist_tracking: Whether to enable waist tracking in visualization
            enable_smpl_vis: Whether to render SMPL body joints in the VR3pt visualizer
            log_prefix: Prefix for log messages
            robot_model: Optional pre-instantiated RobotModel. If None, will create one.
                        Used for FK-based calibration (no display required).
        """
        self.log_prefix = log_prefix
        self.with_g1_robot = with_g1_robot
        self.enable_waist_tracking = enable_waist_tracking
        self.enable_smpl_vis = enable_smpl_vis

        # Robot model for FK-based calibration (headless, no display required)
        self._robot_model = robot_model
        if self._robot_model is None:
            from gear_sonic.data.robot_model.instantiation.g1 import (
                instantiate_g1_robot_model,
            )

            self._robot_model = instantiate_g1_robot_model()
            print(f"[{log_prefix}] Robot model loaded for FK calibration")

        # Optional visualization (requires display + PyVista)
        self.vr3pt_visualizer = None
        if enable_vis_vr3pt:
            if VR3PtPoseVisualizer is None:
                raise ImportError(
                    "VR3PtPoseVisualizer could not be imported but --vis_vr3pt was requested. "
                    "Ensure pyvista is installed: pip install pyvista"
                )
            self.vr3pt_visualizer = VR3PtPoseVisualizer(
                axis_length=0.08,
                ball_radius=0.015,
                with_g1_robot=with_g1_robot,
                robot_model=self._robot_model,
                enable_waist_tracking=enable_waist_tracking,
                enable_smpl_vis=enable_smpl_vis,
            )
            self.vr3pt_visualizer.create_realtime_plotter(interactive=True)
            g1_str = " with G1 robot" if with_g1_robot else ""
            waist_str = " + waist tracking" if enable_waist_tracking else ""
            smpl_str = " + SMPL body" if enable_smpl_vis else ""
            print(f"[{log_prefix}] VR 3pt pose visualization enabled{g1_str}{waist_str}{smpl_str}")

        # Calibration state — triggered explicitly by calibrate_now() or reset_with_measured_q()
        self._calibration_pending = False
        self._calibration_neck_quat_inv: np.ndarray | None = None  # inv(initial neck quat)
        self._calibration_lwrist_offset: np.ndarray | None = None  # position offset
        self._calibration_rwrist_offset: np.ndarray | None = None
        self._calibration_lwrist_rot_offset: sRot | None = None  # orientation offset
        self._calibration_rwrist_rot_offset: sRot | None = None
        # Override robot q for FK during recalibration (e.g. measured joints for VR 3PT)
        self._override_robot_q: np.ndarray | None = None

    @property
    def is_pending(self) -> bool:
        """Check if calibration is pending."""
        return self._calibration_pending

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration has been captured."""
        return self._calibration_neck_quat_inv is not None

    def process_smpl_pose(
        self,
        smpl_pose_np: np.ndarray,
        smpl_joints_local: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Process SMPL pose to extract and calibrate 3-point VR pose.

        Args:
            smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints
            smpl_joints_local: Optional np.ndarray shape (24, 3) - SMPL local joint
                               positions for body visualization. If provided and SMPL
                               visualization is enabled, the joint spheres are updated.

        Returns:
            vr_3pt_pose: np.ndarray shape (3, 7) - Calibrated 3-point pose
                         [L-Wrist, R-Wrist, Neck], each row [x, y, z, qw, qx, qy, qz]
        """
        # Extract raw 3-point pose from SMPL
        vr_3pt_pose_raw = _process_3pt_pose(smpl_pose_np)

        # Capture calibration on first valid frame (or after reset)
        if self._calibration_pending:
            self._capture_calibration(vr_3pt_pose_raw)

        # Apply calibration to get the final pose
        vr_3pt_pose = self._apply_calibration(vr_3pt_pose_raw)

        if self.vr3pt_visualizer is not None:
            self.vr3pt_visualizer.update_from_vr_pose(vr_3pt_pose, waist_scale=1.0)
            if smpl_joints_local is not None:
                self.vr3pt_visualizer.update_smpl_joints(smpl_joints_local)
            self.vr3pt_visualizer.render()

        return vr_3pt_pose

    def close(self) -> None:
        """Close and cleanup visualizer resources."""
        if self.vr3pt_visualizer is not None:
            try:
                self.vr3pt_visualizer.close()
            except Exception as e:
                print(f"[{self.log_prefix}] Warning: Error closing VR3pt visualizer: {e}")

    def calibrate_now(self, body_poses_np: np.ndarray) -> bool:
        """Calibrate using current SMPL frame against FK of all-zero body joints.
        Operator should be in zero-reference pose when calling this."""
        try:
            vr_3pt_pose_raw = _process_3pt_pose(body_poses_np)
            self._override_robot_q = np.zeros(29, dtype=np.float64)
            self._capture_calibration(vr_3pt_pose_raw)
            print(f"[{self.log_prefix}] Calibration completed (zero-pose reference)")
            return True
        except Exception as e:
            print(f"[{self.log_prefix}] Calibration failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _capture_calibration(self, vr_3pt_pose: np.ndarray) -> None:
        """Capture calibration offsets from vr_3pt_pose against G1 FK reference.
        If neck calibration already exists (e.g. from calibrate_now), it is preserved
        to avoid jumps from SMPL noise during recalibration."""

        # Step 1: Neck orientation — only capture if not already set
        if self._calibration_neck_quat_inv is None:
            neck_quat_wxyz = vr_3pt_pose[2, 3:].copy()
            neck_rot = sRot.from_quat(neck_quat_wxyz, scalar_first=True)
            self._calibration_neck_quat_inv = neck_rot.inv().as_quat(scalar_first=True)
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        # Step 2: Rotate VR wrist positions/orientations by neck inverse
        lwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[0, :3].copy())
        rwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[1, :3].copy())
        lwrist_rot_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
        rwrist_rot_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)

        # Step 3: Get G1 FK reference poses
        if self._robot_model is None:
            raise RuntimeError(
                "Robot model is required for calibration but was not loaded. "
                "Ensure the G1 robot model and URDF are available."
            )
        if get_g1_key_frame_poses is None:
            raise RuntimeError(
                "get_g1_key_frame_poses could not be imported. "
                "Ensure gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer is available."
            )

        # Convert 29-DOF override to full model config if needed
        if self._override_robot_q is not None:
            robot_q = self._robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=self._override_robot_q[:29]
            )
        else:
            robot_q = None
        g1_poses = get_g1_key_frame_poses(self._robot_model, q=robot_q)

        g1_lwrist_pos = g1_poses["left_wrist"]["position"]
        g1_rwrist_pos = g1_poses["right_wrist"]["position"]
        g1_lwrist_rot = sRot.from_quat(
            g1_poses["left_wrist"]["orientation_wxyz"], scalar_first=True
        )
        g1_rwrist_rot = sRot.from_quat(
            g1_poses["right_wrist"]["orientation_wxyz"], scalar_first=True
        )

        # Compute position offsets: calibrated = neck_corrected - offset
        self._calibration_lwrist_offset = lwrist_pos_corrected - g1_lwrist_pos
        self._calibration_rwrist_offset = rwrist_pos_corrected - g1_rwrist_pos

        # Compute orientation offsets: calibrated = rot_offset * neck_corrected
        self._calibration_lwrist_rot_offset = g1_lwrist_rot * lwrist_rot_corrected.inv()
        self._calibration_rwrist_rot_offset = g1_rwrist_rot * rwrist_rot_corrected.inv()

        self._calibration_pending = False
        self._override_robot_q = None

        # Log summary
        source = "override q" if g1_lwrist_pos.any() else "default/zero"
        print(
            f"[{self.log_prefix}] Calibration captured (FK ref: {source}):\n"
            f"  L-Wrist pos offset: [{self._calibration_lwrist_offset[0]:.4f}, "
            f"{self._calibration_lwrist_offset[1]:.4f}, {self._calibration_lwrist_offset[2]:.4f}]\n"
            f"  R-Wrist pos offset: [{self._calibration_rwrist_offset[0]:.4f}, "
            f"{self._calibration_rwrist_offset[1]:.4f}, {self._calibration_rwrist_offset[2]:.4f}]"
        )

    def _apply_calibration(self, vr_3pt_pose: np.ndarray) -> np.ndarray:
        """Apply stored calibration offsets to raw VR 3-point pose."""
        if self._calibration_neck_quat_inv is None:
            return vr_3pt_pose

        calibrated = vr_3pt_pose.copy()
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        # Neck orientation: calibrated = inv(initial) * current
        neck_rot = sRot.from_quat(vr_3pt_pose[2, 3:], scalar_first=True)
        calibrated[2, 3:] = (calib_inv_rot * neck_rot).as_quat(scalar_first=True)

        # Wrist positions: rotate by neck inverse, then subtract offset
        if self._calibration_lwrist_offset is not None:
            calibrated[0, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[0, :3]) - self._calibration_lwrist_offset
            )
        if self._calibration_rwrist_offset is not None:
            calibrated[1, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[1, :3]) - self._calibration_rwrist_offset
            )

        # Wrist orientations: rot_offset * (neck_inv * current)
        if self._calibration_lwrist_rot_offset is not None:
            lw_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
            calibrated[0, 3:] = (self._calibration_lwrist_rot_offset * lw_corrected).as_quat(
                scalar_first=True
            )
        if self._calibration_rwrist_rot_offset is not None:
            rw_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)
            calibrated[1, 3:] = (self._calibration_rwrist_rot_offset * rw_corrected).as_quat(
                scalar_first=True
            )

        # Neck position via kinematic chain: root → torso_link (+Z) → neck (along calibrated Z)
        neck_z = sRot.from_quat(calibrated[2, 3:], scalar_first=True).apply([0, 0, 1])
        calibrated[2, :3] = (
            np.array([0, 0, self.TORSO_LINK_OFFSET_Z]) + self.NECK_LINK_LENGTH * neck_z
        ).astype(np.float32)

        return calibrated

    def _clear_calibration(self):
        """Clear all calibration state."""
        self._calibration_neck_quat_inv = None
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._override_robot_q = None

    def reset(self) -> None:
        """Reset calibration. Next process_smpl_pose() call will recalibrate."""
        self._clear_calibration()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Calibration reset, will re-calibrate on next frame")

    def reset_with_measured_q(self, body_q_measured: np.ndarray) -> None:
        """Recalibrate wrist offsets using measured robot joints (29 DOFs).
        Preserves neck calibration to avoid jumps from SMPL noise.
        Next process_smpl_pose() will recompute wrist offsets against FK of these joints."""
        # Preserve neck calibration — only clear wrist offsets
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._override_robot_q = body_q_measured.copy()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Wrist recalibration pending (neck preserved, measured q)")


class FeedbackReader:
    """Reads feedback from robot via ZMQ for VR_3PT recalibration.

    Simplified: FROZEN_UPPER_BODY branch removed, only keeps
    full_body_q_measured for VR 3PT recalibration use.
    """

    def __init__(self, zmq_feedback_host: str = "localhost", zmq_feedback_port: int = 5557):
        self.poller = ZMQPoller(host=zmq_feedback_host, port=zmq_feedback_port, topic="g1_debug")
        self.full_body_q_measured: np.ndarray | None = None

    def poll_feedback(self):
        """Poll for feedback once, and update internal state."""
        self.full_body_q_measured = self._read_full_body_q()

    def _read_full_body_q(self) -> np.ndarray | None:
        data = self.poller.get_data()
        if data is None:
            print("[PlannerLoop] No feedback data received")
            return None
        unpacked = msgpack.unpackb(data, raw=False)
        if "body_q_measured" in unpacked:
            body_q_swizzled = unpacked["body_q_measured"]
            full_body_q = np.array(body_q_swizzled, dtype=np.float64)
            print("[PlannerLoop] Received measured body_q for VR_3PT recalibration")
            return full_body_q
        print("[PlannerLoop] body_q_measured not in feedback data")
        return None



class PlannerStreamer:
    """Encapsulates the planner control loop state and logic — simplified.

    Changes from original:
      - Only 3 locomotion modes: IDLE, SLOW_WALK, IDLE_SQUAT
      - A+B toggles IDLE <-> SLOW_WALK, X+Y toggles IDLE <-> IDLE_SQUAT
      - Right stick ry controls SQUAT height (cumulative offset)
      - FROZEN_UPPER_BODY branch removed
      - poll_hz = 10 Hz
    """

    def __init__(
        self,
        socket,
        reader: PicoReader,
        three_point: ThreePointPose,
        poll_hz: int = 10,
        zmq_feedback_host: str = "localhost",
        zmq_feedback_port: int = 5557,
    ):
        self.socket = socket
        self.reader = reader
        self.three_point = three_point
        self.feedback_reader = FeedbackReader(
            zmq_feedback_host=zmq_feedback_host, zmq_feedback_port=zmq_feedback_port
        )

        self.dt = 1.0 / max(1, poll_hz)
        # Current locomotion mode, default IDLE
        self.mode = LocomotionMode.IDLE
        # Toggle edge detection for A+B and X+Y
        self.prev_ab = False
        self.prev_xy = False
        # Persistent facing buffer (unit vector on XY plane)
        self.yaw_accumulator = YawAccumulator()
        self.last_send = time.time()
        self.last_xrt_timestamp = None

        # Height control — cumulative accumulator for SQUAT mode
        self.height_accumulator = 0.79  # standing default height (meters)

        # Hand IK solvers for trigger-controlled hand open/close in VR 3PT mode
        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()

    def reset_yaw(self):
        """Called when entering planner mode. Resets state for fresh start."""
        self.yaw_accumulator.reset()

    def seed_button_state(self):
        """Seed prev_ab/prev_xy with current button state to prevent
        spurious toggles on the first run_once() call after entering PLANNER mode."""
        a, b, x, y = get_abxy_buttons()
        self.prev_ab = bool(a) and bool(b)
        self.prev_xy = bool(x) and bool(y)

    def recalibrate_for_vr3pt(self):
        """
        Recalibrate VR 3-point pose tracking using the robot's current measured joints.

        Polls the g1_debug feedback to get the robot's actual joint state, then
        schedules recalibration so VR tracking aligns with the robot's current pose.
        This prevents sudden jumps when entering VR 3PT mode from PLANNER mode.
        """
        self.feedback_reader.poll_feedback()
        if self.feedback_reader.full_body_q_measured is not None:
            self.three_point.reset_with_measured_q(self.feedback_reader.full_body_q_measured)
            print("[PlannerLoop] VR 3PT recalibration scheduled with measured robot pose")
        else:
            print(
                "[PlannerLoop] WARNING: No feedback data for VR 3PT recalibration, "
                "using zero body_q as fallback"
            )
            self.three_point.reset_with_measured_q(np.zeros(29, dtype=np.float64))

    def run_once(self, stream_mode: StreamMode):
        """Execute one iteration of the planner control loop."""
        try:
            # Avoid sending old commands if XRT timestamp hasn't advanced
            xrt_timestamp = xrt.get_time_stamp_ns()
            if xrt_timestamp == self.last_xrt_timestamp:
                return
            self.last_xrt_timestamp = xrt_timestamp

            # ---- Locomotion mode toggles ----
            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()
            ab_now = bool(a_pressed) and bool(b_pressed)
            xy_now = bool(x_pressed) and bool(y_pressed)

            if ab_now and not self.prev_ab:
                # A+B: toggle IDLE <-> SLOW_WALK
                if self.mode == LocomotionMode.SLOW_WALK:
                    self.mode = LocomotionMode.IDLE
                else:
                    self.mode = LocomotionMode.SLOW_WALK
                print(f"[PlannerLoop] Mode -> {self.mode.value}: {self.mode.name}")

            if xy_now and not self.prev_xy:
                # X+Y: toggle IDLE <-> IDLE_SQUAT
                if self.mode == LocomotionMode.IDLE_SQUAT:
                    self.mode = LocomotionMode.IDLE
                    self.height_accumulator = 0.79  # reset to standing height when leaving SQUAT
                else:
                    self.mode = LocomotionMode.IDLE_SQUAT
                    self.height_accumulator = 0.79  # start from standing height
                print(f"[PlannerLoop] Mode -> {self.mode.value}: {self.mode.name}")

            self.prev_ab = ab_now
            self.prev_xy = xy_now

            # ---- Read axes for movement, facing, speed, and height ----
            lx, ly, rx, ry = get_controller_axes()

            # In SQUAT mode, disable rx (facing control) — keep current facing unchanged
            if self.mode == LocomotionMode.IDLE_SQUAT:
                rx = 0.0

            # Facing from RIGHT stick rx: continuous yaw
            facing = self.yaw_accumulator.update(rx, self.dt)

            mode_to_send = LocomotionMode.IDLE
            movement = [0.0, 0.0, 0.0]
            speed = -1.0
            height = -1.0

            # ---- Height control (SQUAT mode only) ----
            if self.mode == LocomotionMode.IDLE_SQUAT:
                self.height_accumulator += ry * 0.02
                self.height_accumulator = np.clip(self.height_accumulator, 0.2, 0.8)
                height = self.height_accumulator

            if self.mode == LocomotionMode.SLOW_WALK:
                # Movement from LEFT stick applies only to SLOW_WALK.
                raw_mag = np.hypot(lx, ly)
                raw_mag = np.clip(raw_mag, 0.0, 1.0)
                if np.abs(raw_mag) >= JOYSTICK_DEADZONE:
                    mag = (raw_mag - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
                    if mag > 1.0:
                        mag = 1.0
                    mode_to_send = LocomotionMode.SLOW_WALK
                    speed = 0.1 + 0.5 * mag  # 0.1 .. 0.6

                    denom = raw_mag if raw_mag > 0.0 else 1.0
                    scale = mag / denom
                    movement_local = np.array([-lx, ly]) * scale
                    perp_x, perp_y = -facing[1], facing[0]
                    rotation_facing = np.array([[perp_x, perp_y], [facing[0], facing[1]]])
                    movement_global = rotation_facing @ movement_local
                    movement = [movement_global[0], movement_global[1], 0.0]
            elif self.mode == LocomotionMode.IDLE_SQUAT:
                # Squat is a static height-controlled mode; ignore LEFT stick.
                mode_to_send = LocomotionMode.IDLE_SQUAT

            # ---- Hand joints and VR 3-point data ----
            left_hand_position = None
            right_hand_position = None
            vr_3pt_position = None
            vr_3pt_orientation = None

            if stream_mode == StreamMode.PLANNER_VR_3PT:
                sample = self.reader.get_latest()
                if sample is not None:
                    print("[PlannerLoop] Sending VR 3-point pose as target")
                    vr_3pt_pose = self.three_point.process_smpl_pose(sample["body_poses_np"])
                    vr_3pt_position = (vr_3pt_pose[:, :3].flatten()).tolist()
                    vr_3pt_orientation = vr_3pt_pose[:, 3:].flatten().tolist()

                # Compute hand joints from trigger/grip inputs
                (
                    left_menu_button,
                    left_trigger,
                    right_trigger,
                    left_grip,
                    right_grip,
                ) = get_controller_inputs()
                lh_joints, rh_joints = compute_hand_joints_from_inputs(
                    self.left_hand_ik_solver,
                    self.right_hand_ik_solver,
                    left_trigger,
                    left_grip,
                    right_trigger,
                    right_grip,
                )
                left_hand_position = lh_joints.reshape(-1).astype(np.float32).tolist()
                right_hand_position = rh_joints.reshape(-1).astype(np.float32).tolist()

            msg = build_planner_message(
                mode_to_send.value,
                movement,
                facing,
                speed=speed,
                height=height,
                left_hand_position=left_hand_position,
                right_hand_position=right_hand_position,
                vr_3pt_position=vr_3pt_position,
                vr_3pt_orientation=vr_3pt_orientation,
            )
            self.socket.send(msg)
        except Exception as e:
            import traceback

            print(f"[PlannerLoop] error: {e}")
            traceback.print_exc()
            raise

        # pacing
        now = time.time()
        sleep_t = self.dt - (now - self.last_send)
        if sleep_t > 0:
            time.sleep(sleep_t)
        self.last_send = time.time()



def run_pico_manager_simple(
    port: int = 5556,
    buffer_size: int = 15,
    zmq_feedback_host: str = "localhost",
    zmq_feedback_port: int = 5557,
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
):
    """
    Simplified PICO manager: PLANNER + VR_3PT only.

    State machine:
        OFF --(A+B+X+Y)--> PLANNER --(A+X)--> PLANNER_VR_3PT --(A+X)--> PLANNER
        Any active mode --(A+B+X+Y)--> OFF

    Button mappings:
        A+B+X+Y : start (OFF->PLANNER) / stop (->OFF)
        A+X     : toggle PLANNER <-> VR_3PT
        A+B     : toggle locomotion IDLE <-> SLOW_WALK
        X+Y     : toggle locomotion IDLE <-> IDLE_SQUAT
        Left Grip + A : toggle data collection
        Left Grip + B : discard episode
        Left stick     : movement direction + magnitude
        Right stick rx : facing (yaw)
        Right stick ry : SQUAT height adjustment (cumulative)
        Left/Right Trigger : hand open/close
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run the manager."
        )
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"[Manager] ZMQ socket bound to port {port}")

    # Print available locomotion modes
    print("[Manager] Available modes:")
    for mode in LocomotionMode:
        print(f"  {mode.value}: {mode.name}")

    # Create shared reader and 3-point pose processor
    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()

    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix="PoseLoop",
    )

    planner_streamer = PlannerStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        poll_hz=10,
        zmq_feedback_host=zmq_feedback_host,
        zmq_feedback_port=zmq_feedback_port,
    )

    # Simplified state machine: OFF, PLANNER, PLANNER_VR_3PT
    print("Manager controls: A+X=toggle PLANNER/VR_3PT, A+B+X+Y=start/stop")
    current_mode = StreamMode.OFF
    prev_toggle_dc = False
    prev_toggle_da = False

    try:
        prev_start_combo = False
        prev_ax_pressed = False

        while True:
            # Poll Pico controller for buttons/axes
            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()
            _, _, _, left_grip_mgr, _ = get_controller_inputs()

            ax_pressed = (a_pressed) and (x_pressed)
            start_combo = (a_pressed) and (b_pressed) and (x_pressed) and (y_pressed)

            new_mode = current_mode

            if current_mode == StreamMode.OFF:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.PLANNER
                    # Calibrate VR 3pt tracking: operator should be in zero-ref pose
                    sample = reader.get_latest()
                    if sample is not None:
                        three_point.calibrate_now(sample["body_poses_np"])
                    else:
                        print("[Manager] WARNING: No SMPL data available for calibration")

            elif current_mode == StreamMode.PLANNER:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.PLANNER_VR_3PT

            elif current_mode == StreamMode.PLANNER_VR_3PT:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.PLANNER

            # Handle mode transitions
            if new_mode != current_mode:
                if new_mode == StreamMode.PLANNER:
                    planner_streamer.reset_yaw()
                    if current_mode == StreamMode.OFF:
                        planner_streamer.seed_button_state()
                elif new_mode == StreamMode.PLANNER_VR_3PT:
                    planner_streamer.recalibrate_for_vr3pt()

                # Send command messages
                if new_mode == StreamMode.OFF:
                    socket.send(build_command_message(start=False, stop=True, planner=True))
                    break  # exit main loop → shutdown
                elif new_mode in (StreamMode.PLANNER, StreamMode.PLANNER_VR_3PT):
                    socket.send(build_command_message(start=True, stop=False, planner=True))

                print(f"[Manager] StreamMode: {current_mode.name} -> {new_mode.name}")
                current_mode = new_mode

            # Run planner loop for active modes
            if current_mode in (StreamMode.PLANNER, StreamMode.PLANNER_VR_3PT):
                planner_streamer.run_once(current_mode)

            # Send manager_state for data exporter (every loop, mode-independent)
            toggle_dc_tmp = bool(a_pressed) and left_grip_mgr > 0.5
            toggle_da_tmp = bool(b_pressed) and left_grip_mgr > 0.5
            toggle_dc = toggle_dc_tmp and not prev_toggle_dc
            toggle_da = toggle_da_tmp and not prev_toggle_da
            prev_toggle_dc = toggle_dc_tmp
            prev_toggle_da = toggle_da_tmp
            socket.send(
                pack_pose_message(
                    {
                        "stream_mode": np.array([current_mode.value], dtype=np.int32),
                        "toggle_data_collection": np.array([toggle_dc], dtype=bool),
                        "toggle_data_abort": np.array([toggle_da], dtype=bool),
                    },
                    topic="manager_state",
                )
            )

            prev_start_combo = start_combo
            prev_ax_pressed = ax_pressed

    except KeyboardInterrupt:
        print("\nStopping manager...")
    finally:
        # Cleanup resources
        reader.stop()
        three_point.close()
        socket.close()
        context.term()
        print("[Manager] Shutdown complete")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Simplified PICO Manager — PLANNER + VR_3PT only")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ PUB port (default: 5556)")
    parser.add_argument("--buffer_size", type=int, default=15, help="SMPL data buffer size")
    parser.add_argument(
        "--zmq_feedback_host", type=str, default="localhost", help="ZMQ feedback host"
    )
    parser.add_argument(
        "--zmq_feedback_port", type=int, default=5557, help="ZMQ feedback port (default: 5557)"
    )
    parser.add_argument(
        "--vis_vr3pt", action="store_true", help="Enable VR 3-point pose visualization"
    )
    parser.add_argument(
        "--no_g1",
        action="store_true",
        help="Disable G1 robot visualization (G1 is shown by default)",
    )
    parser.add_argument(
        "--waist_tracking",
        action="store_true",
        help="Enable G1 robot waist to follow VR head orientation",
    )
    parser.add_argument(
        "--vis_smpl",
        action="store_true",
        help="Enable SMPL body joint visualization in the VR3pt viewer",
    )
    args = parser.parse_args()

    with_g1_robot = not args.no_g1

    run_pico_manager_simple(
        port=args.port,
        buffer_size=args.buffer_size,
        zmq_feedback_host=args.zmq_feedback_host,
        zmq_feedback_port=args.zmq_feedback_port,
        enable_vis_vr3pt=args.vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=args.waist_tracking,
        enable_smpl_vis=args.vis_smpl,
    )
