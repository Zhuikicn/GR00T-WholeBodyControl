"""MY_G1 and MY_G1_BINARY VLA client for SONIC DDS deployment.

The policy server supplies the checkpoint's modality configuration and action
horizon. Controls and action playback follow run_vla_inference; hand observations
use measured Dex3 joints under the selected hand contract. The initial motion
token comes from the SONIC v1.1 initial-pose configuration.
"""

from dataclasses import dataclass
import math
import queue
import threading
import time
from typing import Literal

import numpy as np
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.scripts.run_vla_inference import (
    InferenceConfig,
    _inference_worker_loop,
    _sleep_remaining,
    get_action_field,
    pack_latent_action_message,
    print_green,
    run_policy_inference_and_process,
)
from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.dex3_binary import (
    hand_command_from_qpos,
    hand_qpos_from_command,
)
from gear_sonic.utils.inference.inference_recorder import InferenceRecorder
from gear_sonic.utils.inference.initial_poses_v1_1 import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.rtc import RTCPlayback, RTCSettings, rtc_worker_loop
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    prepare_observation_for_eval,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message


@dataclass
class MyG1InferenceConfig(InferenceConfig):
    """Select the hand contract and independent second-view camera endpoints."""

    embodiment_tag: Literal["my_g1_binary", "my_g1", "MY_G1_BINARY", "MY_G1"] = "my_g1_binary"
    action_horizon: int = 0
    """Read the server horizon with 0; an explicit horizon must match the server."""

    second_view_host: str = "localhost"
    """Second-view camera server host."""

    second_view_port: int = 5560
    """Second-view camera server port, used when the policy requires second_view."""

    rtc: bool = False
    """Enable RTC prefix conditioning; requires the updated Isaac-GR00T server."""

    rtc_execution_steps: int = 0
    """Request next chunk at this index; 0 uses half the server action horizon."""

    rtc_delay_ms: float = 100.0
    """Inference delay budget used to freeze the RTC prefix (milliseconds)."""

    rtc_ramp_rate: float = 5.0
    """Exponential ramp rate for updates to the unfrozen overlap region."""


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    modality_configs,
    log_errors: bool = False,
    second_view_subscriber=None,
    binary_hands: bool = False,
):
    """Assemble the server's state and language fields at the current frame.

    RobotModel maps the measured body and Dex3 motor arrays to joint-group order.
    MY_G1 hand state follows index, middle, thumb. MY_G1_BINARY hand state is
    a signed command classified from measured thumb-first motor angles.
    Video has shape ``[1, 1, H, W, 3]`` and each state has shape ``[1, 1, D]``.
    Each stream keeps its latest sample independently. Returns None while a
    required image or robot-state stream is being acquired.
    """
    camera_msg = camera_subscriber.read()
    state_msg = state_subscriber.get_msg(clear=False)
    second_view_msg = (
        second_view_subscriber.read() if second_view_subscriber is not None else None
    )
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    if second_view_subscriber is not None and second_view_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for second_view msg..", flush=True)
        return None

    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )
    observation = {
        "video": {"ego_view": camera_msg["images"]["ego_view"][np.newaxis, np.newaxis]},
        "state": {},
        "language": {
            key: [[language_prompt]] for key in modality_configs["language"].modality_keys
        },
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }
    if second_view_subscriber is not None:
        observation["video"]["second_view"] = second_view_msg["images"]["second_view"][
            np.newaxis, np.newaxis
        ]
    observation = prepare_observation_for_eval(robot_model, observation)
    observation["state"]["projected_gravity"] = compute_projected_gravity(
        state_msg["base_quat"]
    )[np.newaxis, np.newaxis]
    if binary_hands:
        for side in ("left", "right"):
            observation["state"][f"{side}_hand"] = np.asarray(
                hand_command_from_qpos(state_msg[f"{side}_hand_q"], side),
                dtype=np.float32,
            ).reshape(1, 1, 1)
    observation["state"] = {
        key: observation["state"][key] for key in modality_configs["state"].modality_keys
    }
    return observation


def main(config: MyG1InferenceConfig):
    pause_loop = True

    if config.rtc:
        for name in ("rtc_delay_ms", "rtc_ramp_rate", "action_publish_rate"):
            if not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if config.rtc_execution_steps < 0:
            raise ValueError("rtc_execution_steps must be nonnegative")

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    # Isaac-GR00T PolicyClient
    from gr00t.policy.server_client import PolicyClient

    n1_policy = PolicyClient(host=config.host, port=config.port)

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        print("WARNING: PolicyServer not reachable. Inference will fail until server is up.")

    modality_configs = n1_policy.get_modality_config()
    binary_hands = config.embodiment_tag.lower() == "my_g1_binary"
    hand_action_suffix = "close" if binary_hands else "joints"
    expected_action_keys = {
        "motion_token", f"left_hand_{hand_action_suffix}", f"right_hand_{hand_action_suffix}",
    }
    action_keys = set(modality_configs["action"].modality_keys)
    if action_keys != expected_action_keys:
        raise ValueError(
            f"{config.embodiment_tag} requires action keys {sorted(expected_action_keys)}; "
            f"PolicyServer provides {sorted(action_keys)}"
        )
    server_horizon = len(modality_configs["action"].delta_indices)
    if config.action_horizon and config.action_horizon != server_horizon:
        raise ValueError(f"action_horizon={config.action_horizon} differs from server horizon={server_horizon}")
    config.action_horizon = server_horizon
    rtc_settings = (
        RTCSettings(
            server_horizon,
            config.action_publish_rate,
            config.rtc_execution_steps,
            config.rtc_delay_ms,
            config.rtc_ramp_rate,
        )
        if config.rtc
        else None
    )
    if rtc_settings is not None:
        print_green(
            f"RTC enabled: request at index {rtc_settings.execution_threshold}, "
            f"freeze {rtc_settings.frozen_steps} frames; --rate is ignored"
        )
    print_green(f"{config.embodiment_tag.upper()} state keys: {modality_configs['state'].modality_keys}")
    print_green(f"{config.embodiment_tag.upper()} action horizon: {server_horizon}")

    state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host,
        port=config.state_zmq_port,
    )

    camera_subscriber = ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )
    second_view_subscriber = None
    if "second_view" in modality_configs["video"].modality_keys:
        second_view_subscriber = ComposedCameraClientSensor(
            server_ip=config.second_view_host, port=config.second_view_port
        )

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(
        f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    )
    print_green(f"Using embodiment tag: {config.embodiment_tag}")

    keyboard_listener = ZMQKeyboardSubscriber(
        port=config.keyboard_zmq_port, host=config.keyboard_zmq_host
    )

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    def publish_initial_pose():
        """Publish initial pose command to move robot to starting position."""
        print("Moving to initial pose")
        left_hand = hand_qpos_from_command(1 if initial_pose_left_hand_closed else -1, "left")
        right_hand = hand_qpos_from_command(1 if initial_pose_right_hand_closed else -1, "right")
        zmq_message = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)
        print("Initial pose done.")

    def blend_to_initial_pose(duration_s: float) -> bool:
        """Smoothly interpolate from the last sent motion token to the initial pose.

        Linearly blends over ``duration_s`` seconds at the action publish rate,
        sending intermediate tokens each loop iteration. Returns True if blend
        was performed, False if skipped (no previous token available).
        """
        nonlocal last_sent_motion_token
        if last_sent_motion_token is None:
            print("No previous motion token — snapping to initial pose instead.")
            publish_initial_pose()
            return False

        start_token = last_sent_motion_token.copy()
        target_token = LATENT_INITIAL_MOTION_TOKEN.copy()
        num_steps = max(1, round(config.action_publish_rate * duration_s))
        step_period = 1.0 / config.action_publish_rate

        left_hand = hand_qpos_from_command(1 if initial_pose_left_hand_closed else -1, "left")
        right_hand = hand_qpos_from_command(1 if initial_pose_right_hand_closed else -1, "right")

        print(
            f"Blending to initial pose over {duration_s:.2f}s "
            f"({num_steps} steps at {config.action_publish_rate} Hz)"
        )

        for step in range(num_steps):
            t_step_start = time.monotonic()
            alpha = (step + 1) / num_steps
            blended_token = ((1.0 - alpha) * start_token + alpha * target_token).astype(
                np.float32
            )
            zmq_message = pack_latent_action_message(
                motion_token=blended_token,
                frame_index=np.array([0], dtype=np.int64),
                left_hand_joints=left_hand,
                right_hand_joints=right_hand,
            )
            zmq_socket.send(zmq_message)
            last_sent_motion_token = blended_token.copy()

            elapsed = time.monotonic() - t_step_start
            remaining = step_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print_green("Initial pose blend complete.")
        return True

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 0.0 if config.rtc else 1.0 / config.rate

    zmq_frame_counter = 0
    last_sent_motion_token: np.ndarray | None = None

    PROMPT_MSG_PREFIX = "prompt:"

    def reset_rtc(reason):
        nonlocal cached_action_chunk, action_chunk_index
        if rtc_playback is not None:
            rtc_playback.invalidate(reason)
            cached_action_chunk = None
            action_chunk_index = 0

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter, last_sent_motion_token

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                reset_rtc("prompt_changed")
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            reset_rtc("initial_pose")
            if cpp_loop_running and cpp_mode == "PLANNER":
                if send_cpp_control_command(start=True, planner=False):
                    print("Switched to POSE mode (from PLANNER mode)")
                else:
                    print("Warning: Failed to switch to POSE mode")
            elif not cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")

            pause_loop = True
            if config.initial_pose_blend_duration > 0 and last_sent_motion_token is not None:
                blend_to_initial_pose(config.initial_pose_blend_duration)
            else:
                publish_initial_pose()

            zmq_frame_counter = 0
            cached_action_chunk = None
            action_chunk_index = 0
            print("Cleared cached action chunk, reset frame counter")
        elif key == "p":
            pause_loop = not pause_loop
            reset_rtc("paused" if pause_loop else "resumed")
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if pause_loop:
                print("Policy loop paused (C++ loop still running - press 'k' to stop)")
            else:
                print("Policy loop resumed")
        elif key == "k":
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    if config.rtc:
                        pause_loop = True
                        reset_rtc("cpp_stopped")
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(
                f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}"
            )
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(
                f"Initial pose right hand: "
                f"{'closed' if initial_pose_right_hand_closed else 'open'}"
            )

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    recorder = InferenceRecorder(config.inference_log_path) if config.inference_log_path else None
    rtc_playback = RTCPlayback(rtc_settings, recorder) if rtc_settings is not None else None

    def prepare_obs(prompt):
        return prepare_observation_from_sensors(
            camera_subscriber=camera_subscriber,
            state_subscriber=state_subscriber,
            robot_model=robot_model,
            language_prompt=prompt,
            modality_configs=modality_configs,
            log_errors=True,
            second_view_subscriber=second_view_subscriber,
            binary_hands=binary_hands,
        )

    if rtc_playback is not None:
        inference_worker_thread = threading.Thread(
            target=rtc_worker_loop,
            args=(
                inference_queue,
                result_queue,
                inference_stop_event,
                rtc_playback,
                prepare_obs,
                lambda obs, options, metadata: run_policy_inference_and_process(
                    policy=n1_policy,
                    observation=obs,
                    robot_model=robot_model,
                    recorder=recorder,
                    options=options,
                    request_metadata=metadata,
                ),
            ),
            daemon=True,
        )
    else:
        inference_worker_thread = threading.Thread(
            target=_inference_worker_loop,
            args=(
                inference_queue,
                result_queue,
                inference_stop_event,
                inference_busy_event,
                lambda: prepare_obs(language_prompt_ref[0]),
                lambda obs: run_policy_inference_and_process(
                    policy=n1_policy,
                    observation=obs,
                    robot_model=robot_model,
                    recorder=recorder,
                ),
            ),
            daemon=True,
        )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                result = result_queue.get_nowait()
                if rtc_playback is not None:
                    accepted = rtc_playback.accept(result)
                    if accepted is None:
                        # No usable replacement; keep playing the previous chunk.
                        raise queue.Empty
                    processed_action, action_chunk_index = accepted
                    inference_delay = time.monotonic() - result["metadata"]["started"]
                else:
                    processed_action, inference_start_time = result
                    inference_delay = time.monotonic() - inference_start_time
                    action_chunk_index = calculate_latency_compensated_index(
                        inference_delay, config.action_publish_rate, config.action_horizon
                    )
                if binary_hands and processed_action is not None:
                    for side in ("left", "right"):
                        prediction = np.asarray(
                            get_action_field(processed_action, f"{side}_hand_close"),
                            dtype=np.float32,
                        ).squeeze(axis=-1)
                        processed_action[f"{side}_hand_joints"] = hand_qpos_from_command(
                            np.where(prediction >= 0.0, 1, -1), side
                        ).astype(np.float32)
                cached_action_chunk = processed_action
                last_inference_time = time.monotonic()
                print_green(
                    f'New action chunk (prompt: "{language_prompt_ref[0]}", latency: {inference_delay:.3f}s)'
                )
            except queue.Empty:
                pass

            if rtc_playback is not None:
                request = rtc_playback.reserve(language_prompt_ref[0], paused=pause_loop)
                if request is not None:
                    inference_queue.put_nowait(request)
            else:
                worker_is_busy = inference_busy_event.is_set()
                should_start = should_trigger_new_inference(
                    cached_chunk_exists=(cached_action_chunk is not None),
                    inference_thread_running=worker_is_busy,
                    time_since_last_inference=(time.monotonic() - last_inference_time),
                    inference_interval=inference_interval,
                )

                if should_start:
                    try:
                        inference_queue.put_nowait(None)
                    except queue.Full:
                        pass

            if pause_loop:
                print("Pausing...", end="", flush=True)
                time.sleep(0.2)
                print(".", end="", flush=True)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                else:
                    motion_token = np.asarray(
                        get_action_field(processed_action, "motion_token"),
                        dtype=np.float32,
                    )
                    left_hand_joints = np.asarray(
                        get_action_field(processed_action, "left_hand_joints"),
                        dtype=np.float32,
                    )
                    right_hand_joints = np.asarray(
                        get_action_field(processed_action, "right_hand_joints"),
                        dtype=np.float32,
                    )

                    # Action arrays arrive as (B, T, D) from the model.
                    # Squeeze batch dim to get (T, D), then index by time step.
                    if motion_token.ndim == 3:
                        motion_token = motion_token[0]
                    if left_hand_joints.ndim == 3:
                        left_hand_joints = left_hand_joints[0]
                    if right_hand_joints.ndim == 3:
                        right_hand_joints = right_hand_joints[0]

                    horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                    current_idx = (
                        rtc_playback.playback_index()
                        if rtc_playback is not None
                        else min(action_chunk_index, horizon - 1)
                    )

                    if motion_token.ndim == 2:
                        motion_token = motion_token[current_idx]
                    if left_hand_joints.ndim == 2:
                        left_hand_joints = left_hand_joints[current_idx]
                    if right_hand_joints.ndim == 2:
                        right_hand_joints = right_hand_joints[current_idx]

                    frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                    zmq_frame_counter += 1

                    zmq_message = pack_latent_action_message(
                        motion_token,
                        frame_index,
                        left_hand_joints=left_hand_joints,
                        right_hand_joints=right_hand_joints,
                    )
                    zmq_socket.send(zmq_message)
                    if rtc_playback is not None:
                        rtc_playback.published()
                    last_sent_motion_token = motion_token.copy()
                    if zmq_frame_counter % 50 == 0:
                        print_green(
                            f"ZMQ: Sent latent action - frame: {frame_index[0]}, token shape: {motion_token.shape}"
                        )

                action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif not config.rtc and (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(context="VLA Inference Loop Missed", threshold=0.001)

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=n1_policy.timeout_ms / 1000 + 1 if recorder or config.rtc else 1.0)
        if recorder is not None:
            recorder.close()
        if config.rtc and not inference_worker_thread.is_alive():
            n1_policy.close()
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        camera_subscriber.close()
        if second_view_subscriber is not None:
            second_view_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    config = tyro.cli(
        MyG1InferenceConfig,
        description="MY_G1_BINARY / MY_G1 ego/second-view client. "
        "--action-horizon 0 reads the server horizon; "
        "an explicit horizon must match the server.",
    )
    main(config)
