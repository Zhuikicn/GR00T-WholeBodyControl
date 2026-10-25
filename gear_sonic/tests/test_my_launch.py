"""Launcher checks use fake executables and a separate tmux server, never robot control."""

import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import uuid

from gear_sonic.scripts import my_launch as launcher


class LaunchTests(unittest.TestCase):
    def config(self, *args):
        return launcher.make_parser().parse_args(["--model-path", "/missing/checkpoint", *args])

    def test_dry_run_and_help_need_no_dependencies(self):
        for args in (["--help"], ["--model-path", "/missing/checkpoint", "--dry-run"]):
            result = subprocess.run(
                [sys.executable, str(launcher.SCRIPT_PATH), *args], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        with mock.patch.object(launcher, "create_session") as create, contextlib.redirect_stdout(
            io.StringIO()
        ) as output:
            launcher.main(["--model-path", "/missing/checkpoint", "--dry-run"])
            create.assert_not_called()
            self.assertIn("[keyboard]", output.getvalue())

    def test_commands_preserve_arguments_and_environments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wbc = root / "WBC 中文 ' $(touch injected) `touch injected2`"
            gr00t = root / "Isaac GR00T ' 中文"
            model = root / "model checkpoint ' 中文"
            model.mkdir()
            deploy = wbc / "gear_sonic_deploy"
            (deploy / "scripts").mkdir(parents=True)
            (deploy / "scripts/setup_env.sh").write_text("export SETUP_DONE=yes\n")
            stub = (
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['FAKE_LOG']).write_text(json.dumps({"
                "'argv': sys.argv[1:], 'cwd': os.getcwd(), 'path': os.getenv('PYTHONPATH'), "
                "'setup': os.getenv('SETUP_DONE')}))\n"
            )
            for executable in (
                gr00t / ".venv/bin/python",
                wbc / ".venv_data_collection/bin/python",
                deploy / "deploy.sh",
            ):
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.write_text(stub)
                executable.chmod(0o755)
            prompt = "倒水 ' \" ; $(touch injected3) `touch injected4`\n下一行"
            config = self.config(
                "--gr00t-root",
                str(gr00t),
                "--model-path",
                str(model),
                "--prompt",
                prompt,
                "--data-exporter",
                "--camera-viewer",
                "--dataset-name",
                "dataset ' 中文",
                "--inference-log-path",
                str(root / "log ' 中文 $(touch injected5).jsonl"),
            )
            commands = launcher.build_commands(config, wbc, root / "my launch.py")
            records = {}
            for role, command in commands.items():
                log = root / f"{role}.json"
                result = subprocess.run(
                    ["bash", "-c", command],
                    cwd=root,
                    env={**os.environ, "FAKE_LOG": str(log), "PYTHONPATH": "/existing path"},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                records[role] = json.loads(log.read_text())
            self.assertEqual(
                records["server"]["argv"][records["server"]["argv"].index("--model-path") + 1],
                str(model),
            )
            self.assertEqual(records["client"]["argv"][-1], prompt)
            self.assertEqual(
                records["client"]["argv"][records["client"]["argv"].index("--inference-log-path") + 1],
                str(config.inference_log_path),
            )
            self.assertEqual(records["client"]["path"], f"{wbc}:/existing path")
            self.assertEqual(
                records["client"]["argv"][records["client"]["argv"].index("--action-horizon") + 1],
                "0",
            )
            self.assertEqual(records["deploy"]["setup"], "yes")
            self.assertEqual(records["deploy"]["cwd"], str(deploy))
            self.assertEqual(records["recorder"]["argv"][-1], "dataset ' 中文")
            self.assertIn(prompt, records["recorder"]["argv"])
            self.assertIn("gear_sonic.scripts.run_camera_viewer", records["viewer"]["argv"])
            self.assertEqual(
                records["viewer"]["argv"][-4:],
                ["--camera-host", config.camera_host, "--camera-port", str(config.camera_port)],
            )
            self.assertEqual(records["viewer"]["path"], f"{wbc}:/existing path")
            self.assertFalse(list(root.rglob("injected*")))

    def test_missing_dependencies_report_together(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            launcher.shutil, "which", return_value=None
        ):
            config = self.config("--gr00t-root", directory, "--data-exporter")
            with self.assertRaises(RuntimeError) as error:
                launcher.check_prerequisites(config, Path(directory))
            message = str(error.exception)
            for expected in (
                "tmux not found",
                "Python interpreter missing",
                "Checkpoint directory not found",
                "Deployment file not found",
                "Recording interpreter not found",
            ):
                self.assertIn(expected, message)

    def test_policy_wait_retries_and_times_out(self):
        import msgpack
        import zmq

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        errors = []

        def fake_server():
            try:
                time.sleep(1.1)  # Force at least one failed ping before readiness.
                with zmq.Context() as context, context.socket(zmq.REP) as server:
                    server.setsockopt(zmq.LINGER, 0)
                    server.setsockopt(zmq.RCVTIMEO, 3000)
                    server.bind(f"tcp://127.0.0.1:{port}")
                    self.assertEqual(
                        msgpack.unpackb(server.recv(), raw=False), {"endpoint": "ping"}
                    )
                    server.send(msgpack.packb({"status": "ok", "message": "Server is running"}))
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=fake_server)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                fake_client = Path(directory) / "gear_sonic/scripts/run_my_g1_vla_inference.py"
                fake_client.parent.mkdir(parents=True)
                # A namespace-only directory loses to the installed gear_sonic package.
                # Use regular packages so this subprocess can only run the fake client.
                (fake_client.parent / "__init__.py").touch()
                (fake_client.parent.parent / "__init__.py").touch()
                fake_client.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(launcher.SCRIPT_PATH),
                        "--_client",
                        str(port),
                        "--prompt",
                        "倒水 ' 中文",
                    ],
                    cwd=directory,
                    env={**os.environ, "PYTHONPATH": directory},
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("PolicyServer ready; starting SONIC VLA client.", result.stdout)
                self.assertEqual(
                    json.loads(result.stdout.splitlines()[-1]), ["--prompt", "倒水 ' 中文"]
                )
        finally:
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        start = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            TimeoutError, "check the server pane"
        ):
            launcher.wait_for_server(port, timeout=0.15)
        self.assertLess(time.monotonic() - start, 1)

    def test_keyboard_messages_over_isolated_zmq_port(self):
        import zmq

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with zmq.Context() as context, context.socket(zmq.SUB) as subscriber:
            subscriber.setsockopt(zmq.LINGER, 0)
            subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
            subscriber.setsockopt(zmq.RCVTIMEO, 2000)
            subscriber.connect(f"tcp://127.0.0.1:{port}")
            prompts = iter(("k", "t 倒水 ' 中文", "x"))
            received = []

            def keyboard_input(_):
                if received:
                    received.append(subscriber.recv_string())
                else:
                    received.append(None)
                try:
                    return next(prompts)
                except StopIteration:
                    raise EOFError

            with mock.patch(
                "builtins.input", side_effect=keyboard_input
            ), contextlib.redirect_stdout(io.StringIO()):
                launcher.run_keyboard(port)
            self.assertEqual(received[1:], ["k", "prompt:倒水 ' 中文", "x"])

    @unittest.skipUnless(shutil.which("tmux"), "tmux not installed")
    def test_isolated_tmux_layout_exit_logs_and_existing_session(self):
        socket_name = f"my_launch_test_{uuid.uuid4().hex}"
        real_run = subprocess.run

        def isolated_run(args, **kwargs):
            if args[0] == "tmux":
                args = ["tmux", "-L", socket_name, "-f", "/dev/null", *args[1:]]
            return real_run(args, **kwargs)

        try:
            with mock.patch.object(launcher.subprocess, "run", side_effect=isolated_run):
                commands = {
                    role: f"printf '%s\\n' {shlex.quote(role + ' fake output')}; exit 7"
                    for role in ("server", "deploy", "client", "keyboard", "recorder", "viewer")
                }
                launcher.create_session("test_my_launch", commands)
                self.assertTrue(launcher.session_exists("test_my_launch"))
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    panes = launcher.tmux(
                        "list-panes",
                        "-s",
                        "-t",
                        "=test_my_launch",
                        "-F",
                        "#{pane_id}|#{pane_title}|#{pane_dead}|#{pane_dead_status}|#{window_name}",
                    )
                    if all(row.split("|")[2] == "1" for row in panes.splitlines()):
                        break
                    time.sleep(0.05)
                rows = [row.split("|") for row in panes.splitlines()]
                self.assertEqual(len(rows), 6)
                self.assertEqual({row[1] for row in rows}, set(commands))
                for pane, role, dead, status, window in rows:
                    self.assertEqual(dead, "1")
                    # tmux 3.2a may omit the exit status for a detached background window.
                    self.assertIn(status, ("", "7") if role in ("recorder", "viewer") else ("7",))
                    self.assertEqual(
                        window, {"recorder": "recording", "viewer": "camera"}.get(role, "inference")
                    )
                    output = launcher.tmux("capture-pane", "-p", "-S", "-", "-t", pane)
                    self.assertIn(f"{role} fake output", output)
                # Reopening a closed viewer keeps all control panes and the same camera window.
                core_logs = {
                    pane: launcher.tmux("capture-pane", "-p", "-S", "-", "-t", pane)
                    for pane, role, *_ in rows
                    if role in ("server", "deploy", "client", "keyboard")
                }
                launcher.ensure_camera_viewer(
                    "test_my_launch", "printf 'viewer reopened\\n'; sleep 30"
                )
                launcher.ensure_camera_viewer(
                    "test_my_launch", "printf 'must not replace running viewer\\n'"
                )
                windows = launcher.tmux(
                    "list-windows", "-t", "=test_my_launch", "-F", "#{window_name}"
                )
                self.assertEqual(windows.splitlines().count("camera"), 1)
                for pane, output in core_logs.items():
                    self.assertEqual(
                        launcher.tmux("capture-pane", "-p", "-S", "-", "-t", pane), output
                    )
                with mock.patch.object(launcher, "create_session") as create, mock.patch.object(
                    launcher, "check_prerequisites"
                ) as check, mock.patch.object(
                    launcher, "attach"
                ) as attach, contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    launcher.main(["--model-path", "/missing/model", "--session", "test_my_launch"])
                    create.assert_not_called()
                    check.assert_not_called()
                    attach.assert_called_once_with("test_my_launch")
                with mock.patch.object(launcher, "create_session") as create, mock.patch.object(
                    launcher, "check_prerequisites"
                ) as check, mock.patch.object(
                    launcher, "check_camera_viewer"
                ) as check_viewer, mock.patch.object(
                    launcher, "ensure_camera_viewer"
                ) as ensure_viewer, mock.patch.object(
                    launcher, "attach"
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    launcher.main(
                        [
                            "--model-path",
                            "/missing/model",
                            "--session",
                            "test_my_launch",
                            "--camera-viewer",
                        ]
                    )
                    create.assert_not_called()
                    check.assert_not_called()
                    check_viewer.assert_called_once()
                    ensure_viewer.assert_called_once()
        finally:
            real_run(["tmux", "-L", socket_name, "kill-server"], capture_output=True)


if __name__ == "__main__":
    unittest.main()
