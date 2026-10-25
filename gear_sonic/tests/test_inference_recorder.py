"""Inference recording checks with fake policies; no robot or GPU is used."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from gear_sonic.scripts.run_vla_inference import run_policy_inference_and_process
from gear_sonic.utils.inference.inference_recorder import InferenceRecorder


class InferenceRecorderTests(unittest.TestCase):
    def setUp(self):
        self.action = {
            "action.motion_token": np.zeros((1, 4, 64), dtype=np.float32),
            "action.left_hand_joints": np.ones((1, 4, 7), dtype=np.float32),
            "action.task_progress": np.array([0.5], dtype=np.float32),
        }
        self.obs = {"language": {"task": [["倒水"]]}}

    def test_success_preserves_full_response_and_appends_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "inference.jsonl"
            sessions = []
            for _ in range(2):
                policy = Mock()
                policy.get_action.return_value = (self.action, {"server_policy_ms": np.float64(12)})
                recorder = InferenceRecorder(path)
                sessions.append(recorder.session_id)
                try:
                    processed = run_policy_inference_and_process(policy, self.obs, None, recorder)
                    self.assertIn("motion_token", processed)
                    self.assertNotIn("task_progress", processed)
                    self.assertIn("action.task_progress", self.action)
                    policy.get_action.assert_called_once_with(self.obs, options={"record_timing": True})
                finally:
                    recorder.close()
                    recorder.close()
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["session_id"] for row in records], sessions)
            for row in records:
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["language"]["task"], [["倒水"]])
                self.assertEqual(row["action_shapes"]["action.motion_token"], [1, 4, 64])
                np.testing.assert_array_equal(row["action"]["action.motion_token"], self.action["action.motion_token"])
                self.assertEqual(row["action"]["action.task_progress"], [0.5])
                self.assertEqual(row["info"]["server_policy_ms"], 12)
                self.assertGreaterEqual(row["total_ms"], row["client_round_trip_ms"])
                self.assertGreaterEqual(row["processing_ms"], 0)

    def test_rejected_chunk_is_still_recorded(self):
        self.action["action.motion_token"].fill(2)
        policy = Mock()
        policy.get_action.return_value = (self.action, {})
        recorder = Mock()
        self.assertIsNone(run_policy_inference_and_process(policy, self.obs, None, recorder))
        row = recorder.record.call_args.kwargs
        self.assertEqual(row["status"], "rejected")
        self.assertEqual(row["reason"], "motion_token_out_of_bounds")
        self.assertIs(row["action"], self.action)

    def test_failed_request_is_recorded(self):
        policy = Mock()
        policy.get_action.side_effect = TimeoutError("server timeout")
        recorder = Mock()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(run_policy_inference_and_process(policy, self.obs, None, recorder))
        row = recorder.record.call_args.kwargs
        self.assertEqual(row["status"], "error")
        self.assertIn("TimeoutError", row["error"])
        self.assertGreaterEqual(row["total_ms"], 0)

    def test_disabled_recording_preserves_policy_call(self):
        policy = Mock()
        policy.get_action.return_value = (self.action, {})
        run_policy_inference_and_process(policy, self.obs, None)
        policy.get_action.assert_called_once_with(self.obs)

    def test_records_snapshot_arrays_before_caller_changes_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.jsonl"
            recorder = InferenceRecorder(path)
            values = np.array([1], dtype=np.float32)
            recorder.record(action=values)
            values[0] = 9
            recorder.close()
            self.assertEqual(json.loads(path.read_text())["action"], [1])


if __name__ == "__main__":
    unittest.main()
