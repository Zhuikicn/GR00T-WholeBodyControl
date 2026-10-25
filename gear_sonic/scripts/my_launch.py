"""Launch the MY_G1_BINARY / MY_G1 inference stack in tmux (camera runs externally)."""

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", required=True, type=Path, help="Local checkpoint directory for the selected tag"
    )
    parser.add_argument(
        "--embodiment-tag", choices=("MY_G1_BINARY", "MY_G1"), default="MY_G1_BINARY",
        help="Hand contract used by the checkpoint and client (default: MY_G1_BINARY)",
    )
    parser.add_argument("--gr00t-root", type=Path, default=REPO_ROOT.parent / "Isaac-GR00T")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="demo")
    parser.add_argument(
        "--inference-log-path",
        type=Path,
        help="Append raw action chunks and inference timings to this JSONL file",
    )
    parser.add_argument("--camera-host", default="192.168.123.164")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument(
        "--camera-viewer", action="store_true", help="Show live RGB in a desktop OpenCV window"
    )
    parser.add_argument("--policy-port", type=int, default=5550)
    parser.add_argument("--interface", default="real", help="Real robot network interface or IP")
    parser.add_argument(
        "--deploy-checkpoint",
        type=Path,
        default=Path("policy/sonic_v1_1/model"),
        help="SONIC model prefix, relative to gear_sonic_deploy or absolute",
    )
    parser.add_argument(
        "--deploy-obs-config",
        type=Path,
        default=Path("policy/sonic_v1_1/observation_config.yaml"),
        help="Observation config, relative to gear_sonic_deploy or absolute",
    )
    parser.add_argument(
        "--data-exporter", action="store_true", help="Start optional recording window"
    )
    parser.add_argument("--dataset-name")
    parser.add_argument("--session", default="my_g1_inference")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without checking dependencies or launching",
    )
    return parser


def build_commands(config, repo_root=REPO_ROOT, script_path=SCRIPT_PATH):
    """Return bash commands with all user arguments quoted as literal values."""
    gr00t_root = config.gr00t_root.expanduser().resolve()
    python = gr00t_root / ".venv/bin/python"
    deploy_root = repo_root / "gear_sonic_deploy"
    model_path = config.model_path.expanduser().resolve()
    checkpoint = (deploy_root / config.deploy_checkpoint.expanduser()).resolve()
    obs_config = (deploy_root / config.deploy_obs_config.expanduser()).resolve()

    def command(cwd, argv, *, wbc_path=False):
        env = (
            f"export PYTHONPATH={shlex.quote(str(repo_root))}${{PYTHONPATH:+:$PYTHONPATH}} && "
            if wbc_path
            else ""
        )
        return f"cd {shlex.quote(str(cwd))} && {env}exec {shlex.join([str(arg) for arg in argv])}"

    commands = {
        "server": command(
            gr00t_root,
            [
                python,
                "-u",
                "-m",
                "gr00t.eval.run_gr00t_server",
                "--model-path",
                model_path,
                "--embodiment-tag",
                config.embodiment_tag,
                "--device",
                config.device,
                "--host",
                "127.0.0.1",
                "--port",
                config.policy_port,
            ],
        ),
        "deploy": f"cd {shlex.quote(str(deploy_root))} && source scripts/setup_env.sh && exec "
        + shlex.join(
            [
                "./deploy.sh",
                "--cp",
                str(checkpoint),
                "--obs-config",
                str(obs_config),
                "--input-type",
                "zmq_manager",
                "--output-type",
                "zmq",
                "--zmq-host",
                "127.0.0.1",
                config.interface,
            ]
        ),
        "client": command(
            repo_root,
            [
                python,
                "-u",
                script_path,
                "--_client",
                config.policy_port,
                "--host",
                "127.0.0.1",
                "--port",
                config.policy_port,
                "--embodiment-tag",
                config.embodiment_tag.lower(),
                "--action-horizon",
                "0",
                "--state-zmq-host",
                "127.0.0.1",
                "--action-zmq-host",
                "127.0.0.1",
                "--keyboard-zmq-host",
                "127.0.0.1",
                "--camera-host",
                config.camera_host,
                "--camera-port",
                config.camera_port,
                *(
                    ["--inference-log-path", config.inference_log_path.expanduser().resolve()]
                    if config.inference_log_path is not None else []
                ),
                "--prompt",
                config.prompt,
            ],
            wbc_path=True,
        ),
        "keyboard": command(repo_root, [python, "-u", script_path, "--_keyboard"]),
    }
    if config.data_exporter:
        argv = [
            repo_root / ".venv_data_collection/bin/python",
            "-u",
            "-m",
            "gear_sonic.scripts.run_data_exporter",
            "--task-prompt",
            config.prompt,
            "--camera-host",
            config.camera_host,
            "--camera-port",
            config.camera_port,
            "--state-zmq-host",
            "127.0.0.1",
            "--sonic-zmq-host",
            "127.0.0.1",
        ]
        if config.dataset_name is not None:
            argv += ["--dataset-name", config.dataset_name]
        commands["recorder"] = command(repo_root, argv, wbc_path=True)
    if config.camera_viewer:
        # The Isaac environment uses headless OpenCV; use the GUI-enabled viewer environment.
        gui_env = [
            f"{key}={os.environ[key]}"
            for key in ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR")
            if key in os.environ
        ]
        commands["viewer"] = command(
            repo_root,
            [
                "env",
                *gui_env,
                repo_root / ".venv_data_collection/bin/python",
                "-u",
                "-m",
                "gear_sonic.scripts.run_camera_viewer",
                "--camera-host",
                config.camera_host,
                "--camera-port",
                config.camera_port,
            ],
            wbc_path=True,
        )
    return commands


def check_camera_viewer(repo_root=REPO_ROOT):
    python = repo_root / ".venv_data_collection/bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(
            f"Camera viewer interpreter not found: {python}. "
            "Run bash install_scripts/install_data_collection.sh"
        )
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import cv2, re, sys; "
            "sys.exit('Camera viewer requires GUI-enabled opencv-python, not headless OpenCV') "
            "if re.search(r'GUI:\\s+NONE', cv2.getBuildInformation()) else None",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError(f"Camera viewer dependency check failed: {result.stderr.strip()}")


def check_prerequisites(config, repo_root=REPO_ROOT):
    errors = []
    if not shutil.which("tmux"):
        errors.append("tmux not found; install it with sudo apt install tmux")
    python = config.gr00t_root.expanduser().resolve() / ".venv/bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        errors.append(f"Python interpreter missing or not executable: {python}")
    model_path = config.model_path.expanduser().resolve()
    if not model_path.is_dir():
        errors.append(f"Checkpoint directory not found: {model_path}")
    deploy_root = repo_root / "gear_sonic_deploy"
    checkpoint = (deploy_root / config.deploy_checkpoint.expanduser()).resolve()
    required_files = [
        deploy_root / "deploy.sh",
        deploy_root / "scripts/setup_env.sh",
        Path(str(checkpoint) + "_encoder.onnx"),
        Path(str(checkpoint) + "_decoder.onnx"),
        (deploy_root / config.deploy_obs_config.expanduser()).resolve(),
    ]
    for path in required_files:
        if not path.is_file():
            errors.append(f"Deployment file not found: {path}")
    if config.data_exporter:
        recorder_python = repo_root / ".venv_data_collection/bin/python"
        if not recorder_python.is_file() or not os.access(recorder_python, os.X_OK):
            errors.append(f"Recording interpreter not found: {recorder_python}")
    if errors:
        raise RuntimeError("Prerequisites not met:\n  - " + "\n  - ".join(errors))
    if config.camera_viewer:
        check_camera_viewer(repo_root)


def wait_for_server(port, timeout=300):
    """Ping with the PolicyServer's scalar msgpack protocol, without loading ML code."""
    import msgpack
    import zmq

    deadline = time.monotonic() + timeout
    print(f"Waiting for PolicyServer at 127.0.0.1:{port} (up to {timeout}s)...", flush=True)
    with zmq.Context() as context:
        while time.monotonic() < deadline:
            # Recreate the REQ socket after each attempt and close it without linger.
            with context.socket(zmq.REQ) as socket:
                socket.setsockopt(zmq.LINGER, 0)
                attempt_ms = max(1, min(1000, int((deadline - time.monotonic()) * 1000)))
                socket.setsockopt(zmq.RCVTIMEO, attempt_ms)
                socket.setsockopt(zmq.SNDTIMEO, attempt_ms)
                socket.connect(f"tcp://127.0.0.1:{port}")
                try:
                    socket.send(msgpack.packb({"endpoint": "ping"}))
                    response = msgpack.unpackb(socket.recv(), raw=False)
                    if isinstance(response, dict) and response.get("status") == "ok":
                        print("PolicyServer ready; starting SONIC VLA client.", flush=True)
                        return
                except zmq.Again:
                    continue
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    raise TimeoutError(
        f"PolicyServer did not become ready within {timeout}s; check the server pane."
    )


def run_keyboard(port=5580):
    import zmq

    with zmq.Context() as context, context.socket(zmq.PUB) as publisher:
        publisher.setsockopt(zmq.LINGER, 0)
        publisher.bind(f"tcp://127.0.0.1:{port}")
        time.sleep(0.5)
        print("Wait for deployment and client readiness, then enter k -> i -> p.")
        print("Recording (if enabled): c toggles recording, x discards the episode.")
        try:
            while True:
                key = input("k/i/p/[/]/c/x/s/f or t <prompt>: ")
                publisher.send_string("prompt:" + key[2:] if key.startswith("t ") else key)
        except (EOFError, KeyboardInterrupt):
            print("\nKeyboard publisher stopped.")


def tmux(*args):
    return subprocess.run(
        ["tmux", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def session_exists(session):
    return (
        subprocess.run(["tmux", "has-session", "-t", f"={session}"], capture_output=True).returncode
        == 0
    )


def create_session(session, commands):
    """Create idle panes first so remain-on-exit is set before any child can exit."""
    first = tmux("new-session", "-d", "-s", session, "-n", "inference", "-P", "-F", "#{pane_id}")
    window = tmux("display-message", "-p", "-t", first, "#{window_id}")
    session_id = tmux("display-message", "-p", "-t", first, "#{session_id}")
    tmux("set-option", "-t", session_id, "mouse", "on")
    tmux("set-window-option", "-t", window, "remain-on-exit", "on")
    right = tmux("split-window", "-h", "-t", first, "-P", "-F", "#{pane_id}")
    bottom_left = tmux("split-window", "-v", "-t", first, "-P", "-F", "#{pane_id}")
    bottom_right = tmux("split-window", "-v", "-t", right, "-P", "-F", "#{pane_id}")
    panes = dict(
        zip(("server", "deploy", "client", "keyboard"), (first, right, bottom_left, bottom_right))
    )
    if "recorder" in commands:
        recorder = tmux(
            "new-window", "-d", "-t", session_id, "-n", "recording", "-P", "-F", "#{pane_id}"
        )
        tmux("set-window-option", "-t", recorder, "remain-on-exit", "on")
        panes["recorder"] = recorder
    for role, pane in panes.items():
        tmux("select-pane", "-t", pane, "-T", role)
        # Pass bash and its arguments separately; tmux must not reinterpret quotes.
        tmux("respawn-pane", "-k", "-t", pane, "bash", "-c", commands[role])
    tmux("select-window", "-t", window)
    tmux("select-pane", "-t", panes["deploy"])
    if "viewer" in commands:
        ensure_camera_viewer(session, commands["viewer"])


def ensure_camera_viewer(session, command):
    """Add or reopen the viewer without touching running inference/control processes."""
    windows = tmux("list-windows", "-t", f"={session}", "-F", "#{window_id}|#{window_name}")
    camera_window = next(
        (row.split("|")[0] for row in windows.splitlines() if row.split("|")[1] == "camera"),
        None,
    )
    if camera_window:
        pane, dead = (
            tmux("list-panes", "-t", camera_window, "-F", "#{pane_id}|#{pane_dead}")
            .splitlines()[0]
            .split("|")
        )
        if dead != "1":
            print("Camera viewer is already running.")
            return
    else:
        pane = tmux(
            "new-window", "-d", "-t", f"={session}", "-n", "camera", "-P", "-F", "#{pane_id}"
        )
    tmux("set-window-option", "-t", pane, "remain-on-exit", "on")
    tmux("select-pane", "-t", pane, "-T", "viewer")
    tmux("respawn-pane", "-k", "-t", pane, "bash", "-c", command)
    print(
        "Camera viewer started; RGB appears in 'SONIC Camera Viewer'. Focus it and press q to close."
    )


def attach(session):
    action = "switch-client" if os.environ.get("TMUX") else "attach-session"
    try:
        subprocess.run(["tmux", action, "-t", f"={session}"], check=True)
    except KeyboardInterrupt:
        pass
    print(f"Reattach: {shlex.join(['tmux', 'attach', '-t', '=' + session])}")
    print(f"Close:    {shlex.join(['tmux', 'kill-session', '-t', '=' + session])}")


def main(argv=None):
    parser = make_parser()
    config = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config.session):
        parser.error("--session must contain only letters, digits, underscores or hyphens")
    for name in ("camera_port", "policy_port"):
        if not 1 <= getattr(config, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be between 1 and 65535")
    if config.policy_port in (5556, 5557, 5580):
        parser.error(
            "--policy-port conflicts with fixed action/state/keyboard ports 5556/5557/5580"
        )
    if config.interface in ("sim", "lo", "lo0", "127.0.0.1"):
        parser.error("This launcher supports real robots only; no simulator is started")
    commands = build_commands(config)
    if config.dry_run:
        for role, command in commands.items():
            print(f"[{role}]\n{command}\n")
        return
    try:
        if shutil.which("tmux") and session_exists(config.session):
            print(
                f"Session {config.session!r} already exists; attaching without restarting processes."
            )
            if config.camera_viewer:
                check_camera_viewer()
                ensure_camera_viewer(config.session, commands["viewer"])
        else:
            check_prerequisites(config)
            create_session(config.session, commands)
            print(f"{config.embodiment_tag} stack launched. Top: server | C++; bottom: client | keyboard.")
            print("In the C++ pane, confirm deploy.sh with Enter and wait for initialization.")
            print(
                "When the client is ready, enter k, then i in the keyboard pane; once stable, enter p."
            )
            print("Ctrl+b then arrows: switch panes; Ctrl+b then d: detach and keep running.")
        attach(config.session)
    except (RuntimeError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = (
            exc.stderr.strip()
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr
            else str(exc)
        )
        parser.exit(1, f"ERROR: {detail}\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--_keyboard"]:
        run_keyboard()
    elif sys.argv[1:2] == ["--_client"]:
        try:
            wait_for_server(int(sys.argv[2]))
        except TimeoutError as exc:
            sys.exit(f"ERROR: {exc}")
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-u",
                "-m",
                "gear_sonic.scripts.run_my_g1_vla_inference",
                *sys.argv[3:],
            ],
        )
    else:
        main()
