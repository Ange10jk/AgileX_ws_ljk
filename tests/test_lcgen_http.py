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
        for invalid in ([-0.01], [1.01], [float("nan")], [True], [0.0, 0.5]):
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

    def run_mock(self, reply_mode, *, resume=False):
        import subprocess
        if subprocess.run([sys.executable, "-s", "-c", "import cv2"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            self.skipTest("RoboDojo adapter requires OpenCV in the isolated worker Python")
        requests = []
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
                if self.path != "/predict":
                    self.respond(404, {"error": "not found"})
                    return
                observations = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["observations"]
                requests.append(observations)
                state = observations[-1]["state"]
                for side in ("left", "right"):
                    joints = state[side + "_arm_joint_state"]
                    if not isinstance(joints, list) or len(joints) != 6:
                        self.respond(400, {"error": "expected six joint values"})
                        return
                mode = reply_mode if len(requests) == 1 else "valid"
                if mode == "http_error":
                    self.respond(422, {"request_id": "test-422", "error": "rejected"})
                    return
                if mode == "timeout":
                    time.sleep(0.5)
                rows = [action(index * 0.005) for index in range(8)]
                if mode == "invalid":
                    rows[-1]["right_arm_joint_state"][5] = 0.5
                self.respond(200, {"request_id": "test-action", "actions": rows})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with TemporaryDirectory() as directory:
                cfg = load_config(ROOT / "configs" / "robodojo_piper_api.yaml", {
                    "policy": {"python": sys.executable, "inference_timeout_sec": 0.8,
                               "lcgen_http": {"url": "http://127.0.0.1:%d" % server.server_port,
                                               "connect_timeout_sec": 0.15, "request_timeout_sec": 0.25,
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
                session.run(max_steps=1 if resume else 9, max_seconds=2.5, interactive=False)
                return requests, robot.sent, events
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

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

    def test_mock_http_failures_hold_without_model_targets(self):
        for mode, status in (("http_error", "status=422"), ("timeout", "elapsed=")):
            with self.subTest(mode=mode):
                _, sent, events = self.run_mock(mode)
                self.assertTrue(all(np.all(target == 0.0) for _, target in sent))
                error = next(event for event in events if event["type"] == "policy_error")
                self.assertIn(status, error["reason"])
                self.assertTrue(any(event.get("control_state") == Controller.HOLD
                                    and event["type"] == "mode" for event in events))


if __name__ == "__main__":
    unittest.main()
