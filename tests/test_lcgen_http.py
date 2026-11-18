"""Exercise the RoboDojo HTTP policy and regular control path with mock hardware."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time
import unittest

import numpy as np

from piper_control.config import load_config
from piper_control.control.arbiter import Controller
from piper_control.hardware.robot import MockRobot
from piper_control.policies.interfaces import PolicyError
from piper_control.policies.lcgen_http import parse_actions, PiperKinematics
from piper_control.workflows.session import Session

ROOT = Path(__file__).resolve().parents[1]
CAMERA_KEYS = {"cam_high", "cam_left_wrist", "cam_right_wrist"}


def action(step):
    result = {}
    for side, sign in (("left", 1), ("right", -1)):
        result[side + "_arm_joint_state"] = [
            sign * (step + joint / 100) for joint in range(1, 7)
        ]
    result["left_ee_joint_state"] = [0.6]
    result["right_ee_joint_state"] = [1.0]
    return result


def write_test_urdf(path):
    joints = []
    for index in range(1, 7):
        parent = "base_link" if index == 1 else "link%d" % (index - 1)
        joints.append(
            '<joint name="joint%d" type="revolute"><parent link="%s"/>'
            '<child link="link%d"/><origin xyz="0.1 0 0"/><axis xyz="0 0 1"/>'
            '<limit lower="-1" upper="1"/></joint>' % (index, parent, index)
        )
    path.write_text('<robot name="mock-piper">' + "".join(joints) + "</robot>")
    return path


class RecordingRobot(MockRobot):
    def __init__(self, config):
        super().__init__(config)
        self.sent = []

    def command(self, target):
        sent = super().command(target)
        self.sent.append((time.monotonic(), sent))
        return sent


class TestLCGenHTTP(unittest.TestCase):
    def test_parse_joint_arrays_and_convert_only_grippers(self):
        sample = action(0)
        next_sample = action(0.01)
        next_sample["left_ee_joint_state"] = [0.0]
        actual = parse_actions({"actions": [sample, next_sample]})
        np.testing.assert_array_equal(actual[0, :6], [i / 100 for i in range(1, 7)])
        np.testing.assert_array_equal(actual[0, 7:13], [-i / 100 for i in range(1, 7)])
        np.testing.assert_array_equal(actual[:, [6, 13]], [[0.06, 0.1], [0.0, 0.1]])

    def test_parse_clips_normalized_gripper_widths_before_converting_to_metres(self):
        sample = action(0)
        sample["left_ee_joint_state"] = [-0.01]
        sample["right_ee_joint_state"] = [1.01]
        actual = parse_actions({"actions": [sample]})
        np.testing.assert_array_equal(actual[0, [6, 13]], [0.0, 0.1])

    def test_parse_rejects_invalid_gripper_values(self):
        sample = action(0)
        next_sample = action(0.01)
        for invalid in ([float("nan")], [True], [0.0, 0.5]):
            with self.subTest(invalid=invalid):
                next_sample["right_ee_joint_state"] = invalid
                with self.assertRaises(PolicyError):
                    parse_actions({"actions": [sample, next_sample]})

    def test_parse_rejects_invalid_joint_arrays(self):
        valid = action(0)["right_arm_joint_state"]
        for invalid in (
            {"joint%d" % index: value for index, value in enumerate(valid, 1)},
            valid[:-1],
            [*valid[:-1], {"joint6": valid[-1]}],
            [*valid[:-1], float("inf")],
        ):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(PolicyError, "right_arm_joint_state"):
                row = action(0)
                row["right_arm_joint_state"] = invalid
                parse_actions({"actions": [row]})

    def test_urdf_joint_rotation_and_position(self):
        with TemporaryDirectory() as directory:
            fk = PiperKinematics(write_test_urdf(Path(directory) / "piper.urdf"))
            pose = fk.pose([np.pi / 2, 0, 0, 0, 0, 0])
            np.testing.assert_allclose(
                pose, [0.1, 0.5, 0.0, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
                atol=1e-10,
            )

    def run_mock(self, reply_mode, *, resume=False, request_mode="v0"):
        import subprocess
        if subprocess.run([sys.executable, "-s", "-c", "import cv2"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            self.skipTest("RoboDojo adapter requires OpenCV in the isolated worker Python")
        requests = []
        sessions = []
        resets = []
        metadata = {
            "robot": "piper", "execution_action_type": "joint",
            "executed_raw_action_dim": 14, "runtime_fps": 30.0,
            "ik": {"base_link": "base_link", "tip_link": "link6", "joint_units": "rad",
                   "joint_order": ["joint%d" % i for i in range(1, 7)],
                   "max_initial_joint_step_rad": [0.13] * 6,
                   "max_joint_step_rad": [0.07] * 6},
        }

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, code, body):
                blob = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                try:
                    self.wfile.write(blob)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path == "/health":
                    self.respond(200, {"status": "ready", "metadata": metadata})
                else:
                    self.respond(404, {"error": "not found"})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/reset":
                    resets.append(payload["session_id"])
                    self.respond(200, {})
                    return
                if self.path != "/predict":
                    self.respond(404, {"error": "not found"})
                    return
                observations = payload["observations"]
                if payload.get("request_mode") == "v1":
                    sessions.append(payload["session_id"])
                requests.append(observations)
                state = observations[-1]["state"]
                for side in ("left", "right"):
                    joints = state[side + "_arm_joint_state"]
                    if not isinstance(joints, list) or len(joints) != 6:
                        self.respond(400, {"error": "expected six joint values"})
                        return
                mode = reply_mode if len(requests) == 1 else "valid"
                if reply_mode == "http_error" or (reply_mode == "ik_recovery" and len(requests) <= 3):
                    self.respond(422, {"request_id": "test-422-%d" % len(requests), "error": "IK failed"})
                    return
                if mode == "timeout":
                    time.sleep(0.5)
                count = 25 if sessions and mode != "short" else 8
                rows = [action(index * 0.003) for index in range(count)] if sessions else [
                    action(index * 0.005) for index in range(count)
                ]
                if mode == "invalid":
                    rows[-1]["right_arm_joint_state"][5] = 0.5
                reply = {"request_id": "test-action", "actions": rows}
                if sessions:
                    reply["session_id"] = sessions[-1]
                self.respond(200, reply)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with TemporaryDirectory() as directory:
                cfg = load_config(ROOT / "configs" / "robodojo_piper_api.yaml", {
                    "policy": {"python": sys.executable, "inference_timeout_sec": 0.8,
                               "lcgen_http": {"url": "http://127.0.0.1:%d" % server.server_port,
                                               "connect_timeout_sec": 0.15, "request_timeout_sec": 0.25,
                                               "request_mode": request_mode,
                                               "max_observation_skew_sec": 0.15,
                                               "urdf_path": str(write_test_urdf(Path(directory) / "piper.urdf"))}},
                    "observation": {"width": 160, "height": 120},
                })
                robot = RecordingRobot(cfg["robot"])
                session = Session(cfg, mock=True, robot=robot, command_schedule=[(0, "p")])
                events = []
                original = session.event

                def event(value):
                    events.append(value)
                    original(value)
                    if value["type"] == "policy_error":
                        session.commands.put("p" if resume else "q")

                session.event = event
                session.controller.event_sink = event
                steps = 26 if request_mode == "v1" else 9
                session.run(max_steps=1 if resume else steps, max_seconds=2.5, interactive=False)
                if request_mode == "v1":
                    return requests, robot.sent, events, sessions, resets
                return requests, robot.sent, events
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_v1_sends_only_current_frame_and_releases_session(self):
        requests, sent, events, sessions, resets = self.run_mock("valid", request_mode="v1")
        self.assertEqual([[obs["wam_frame_index"] for obs in batch] for batch in requests],
                         [[0], [25]])
        self.assertEqual(len(set(sessions)), 1)
        self.assertEqual(resets, [sessions[0]])
        self.assertFalse([event for event in events if event["type"] == "policy_error"])
        self.assertEqual(requests[1][0]["state"]["left_arm_joint_state"],
                         action(24 * 0.003)["left_arm_joint_state"])

    def test_v1_rejects_short_chunks_before_dispatch(self):
        requests, sent, events, sessions, resets = self.run_mock("short", request_mode="v1")
        self.assertEqual(len(requests), 1)
        self.assertTrue(all(np.all(target == 0.0) for _, target in sent))
        self.assertIn("must return 25 actions",
                      next(event["reason"] for event in events if event["type"] == "policy_error"))
        self.assertEqual(resets, [sessions[0]])

    def test_mock_history_and_target_timing(self):
        requests, sent, events = self.run_mock("valid")
        self.assertEqual([len(history) for history in requests], [1, 9])
        self.assertFalse([event for event in events if event["type"] == "policy_error"])
        for history in requests:
            for index, obs in enumerate(history):
                self.assertEqual(obs["wam_frame_index"], index)
                self.assertEqual(obs["wam_fps"], 30.0)
                self.assertEqual(obs["env_idx"], 0)
                self.assertEqual(set(obs["images"]), CAMERA_KEYS)
                self.assertTrue(all(base64.b64decode(img).startswith(b"\xff\xd8")
                                    for img in obs["images"].values()))
                self.assertEqual("state" in obs, index == len(history) - 1)
                self.assertEqual("instruction" in obs, index == len(history) - 1)
        first = requests[0][-1]["state"]
        np.testing.assert_allclose(first["left_ee_pose"], [0.6, 0, 0, 1, 0, 0, 0])
        last = requests[1][-1]["state"]
        for side in ("left", "right"):
            field = side + "_arm_joint_state"
            self.assertEqual(first[field], [0.0] * 6)
            self.assertEqual(last[field], action(7 * 0.005)[field])
        self.assertEqual(last["left_ee_joint_state"], [0.6])
        self.assertEqual(last["right_ee_joint_state"], [1.0])
        model = [(stamp, target) for stamp, target in sent if target[6] == 0.06]
        self.assertGreaterEqual(len(model), 9)
        np.testing.assert_allclose([target[0] for _, target in model[:8]], np.arange(8) * 0.005 + 0.01)
        interval = np.diff([stamp for stamp, _ in model[:8]])
        self.assertTrue(np.all((interval > 0.025) & (interval < 0.070)), interval)

    def test_mock_rejects_whole_chunk_and_starts_fresh_episode(self):
        requests, sent, events = self.run_mock("invalid", resume=True)
        self.assertEqual([len(history) for history in requests], [1, 1])
        error = next(event for event in events if event["type"] == "policy_error")
        self.assertIn("test-action", error["reason"])
        self.assertIn("adjacent joint step", error["reason"])
        model = [(stamp, target) for stamp, target in sent if target[6] == 0.06]
        self.assertTrue(model)
        self.assertTrue(all(target[0] == 0.01 for _, target in model))

    def test_mock_ik_retries_reuse_frame_and_continue(self):
        for request_mode, next_frame in (("v0", 8), ("v1", 25)):
            with self.subTest(request_mode=request_mode):
                requests, _, events, *session_info = self.run_mock("ik_recovery", request_mode=request_mode)
                self.assertEqual(len(requests), 5)
                self.assertEqual(requests[:4], [requests[0]] * 4)
                self.assertEqual(requests[4][-1]["wam_frame_index"], next_frame)
                if session_info:
                    sessions, resets = session_info
                    self.assertEqual(sessions, [sessions[0]] * 5)
                    self.assertEqual(resets, [sessions[0]])
                self.assertFalse([event for event in events if event["type"] == "policy_error"])

    def test_mock_http_failures_hold_without_model_targets(self):
        for mode, status, attempts in (("http_error", "request_id=test-422-4", 4),
                                       ("timeout", "elapsed=", 1)):
            with self.subTest(mode=mode):
                requests, sent, events = self.run_mock(mode)
                self.assertEqual(len(requests), attempts)
                self.assertEqual(requests, [requests[0]] * attempts)
                self.assertTrue(all(np.all(target == 0.0) for _, target in sent))
                error = next(event for event in events if event["type"] == "policy_error")
                self.assertIn(status, error["reason"])
                if mode == "http_error":
                    self.assertIn("request_ids=['test-422-1', 'test-422-2', 'test-422-3', 'test-422-4']",
                                  error["reason"])
                self.assertTrue(any(event.get("control_state") == Controller.HOLD
                                    and event["type"] == "mode" for event in events))


if __name__ == "__main__":
    unittest.main()
