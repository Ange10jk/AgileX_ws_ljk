"""Exercise local RoboDojo replay with production startup and fake hardware."""

from contextlib import redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from piper_control.control.arbiter import Controller
from piper_control.hardware.cameras import MockCameras
from piper_control.hardware.robot import RobotError
from piper_control.policies.lcgen_http import validate_trajectory
from piper_control.recording.session import RecordingSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import test_robodojo_piper_replay as replay
except ModuleNotFoundError as exc:
    if exc.name != "pyarrow":
        raise
    replay = None
finally:
    sys.path.pop(0)

LIMITS = np.asarray([[-1, 1]] * 6, dtype=float)
CAPS = {"joint_limits": LIMITS, "initial_step_rad": [0.12] * 6, "step_rad": [0.065] * 6}
DATASET_TEST_PATH = os.environ.get("PIPER_REPLAY_TEST_DATASET")
INITIAL = np.asarray([-1.25, 0.5, -0.6, 0, 0, 0, 0.029,
                      0.04, 0.3, -0.35, 0, 0, 0, 0.073])


class Recorder(RecordingSession):
    def __init__(self, config):
        super().__init__(config)
        self.targets, self.times, self.events = [], [], []

    def publish_action(self, action, source, metadata=None):
        if metadata["predicted_action"] is not None:
            self.targets.append(action.copy())
            self.times.append(time.monotonic())

    def write_event(self, event):
        self.events.append(event)


class FakeCameras(MockCameras):
    def __init__(self, config, events):
        super().__init__(config)
        self.events = events

    def start(self):
        self.events.append(("cameras", "started"))
        return super().start()

    def close(self):
        self.events.append(("cameras", "closed"))
        super().close()


class FakeFollower:
    def __init__(self, channel, pose, events, refusal=None):
        self.channel, self.pose, self.events = channel, pose, events
        self.refusal = refusal
        self.enabled = False
        self.stale = False
        self.arm_status = 0
        self.arm_status_age = 0.0
        self.arm_status_on_enable = None
        self.arm_status_on_target = None
        self.force_enabled_feedback = False

    def connect(self):
        self.events.append((self.channel, "connect"))

    def disconnect(self):
        self.events.append((self.channel, "disconnect"))

    def set_follower_mode(self):
        self.events.append((self.channel, "mode"))

    def enable(self):
        self.events.append((self.channel, "enable"))
        self.enabled = True
        if self.arm_status_on_enable is not None:
            self.arm_status = self.arm_status_on_enable

    def set_speed_percent(self, speed):
        self.events.append((self.channel, "speed", speed))

    def get_driver_states(self, index):
        return SimpleNamespace(timestamp=time.time(), msg=SimpleNamespace(
            foc_status=SimpleNamespace(driver_enable_status=self.enabled or self.force_enabled_feedback)))

    def get_motor_states(self, index):
        return None

    def get_arm_status(self):
        if self.arm_status is not None:
            return SimpleNamespace(timestamp=time.time() - self.arm_status_age,
                                   msg=SimpleNamespace(arm_status=self.arm_status))
        return None

    def get_joint_angles(self):
        if not self.enabled:
            raise AssertionError("Follower position was read before production enable")
        if (self.channel, "feedback") not in self.events:
            self.events.append((self.channel, "feedback"))
        stamp = time.time() - (1 if self.stale else 0)
        return SimpleNamespace(timestamp=stamp, msg=self.pose[:6])

    def get_gripper_status(self):
        return SimpleNamespace(timestamp=time.time(), msg=SimpleNamespace(value=self.pose[6]))

    def move_js(self, joints):
        self.events.append((self.channel, "joint", joints))
        if self.arm_status_on_target == joints[0]:
            self.arm_status = replay.PIPER_EMERGENCY_STOP
        if self.refusal == joints[0]:
            return False

    def move_gripper_m(self, *, value, force):
        self.events.append((self.channel, "gripper", value))

    def electronic_emergency_stop(self):
        self.events.append((self.channel, "stop"))


def mock_config():
    config = replay.load_config(replay.CONFIG, {"observation": {"width": 160, "height": 120}})
    config["robot"]["move_speed_percent"] = 10
    for side, channel in zip(config["robot"]["arms"], ("fake0", "fake1")):
        config["robot"]["pairs"][side]["follower"] = {"channel": channel}
    return config


def mock_hardware(config, robot_type, *, initial=INITIAL, refusal=None):
    events = []
    sdk = {channel: FakeFollower(channel, pose, events, refusal=refusal)
           for channel, pose in zip(("fake0", "fake1"), initial.reshape(2, 7))}

    def factory(spec, _):
        device = sdk[spec["channel"]]
        return device, device

    robot = robot_type(factory)
    cameras = FakeCameras(config["observation"], events)
    return robot, sdk, cameras, events


@unittest.skipIf(replay is None, "PyArrow is required for the replay entry point")
class TestRoboDojoPiperReplay(unittest.TestCase):
    def run_session(self, actions, *, checks=False, refusal=None, schedule=None, stale_after_start=False,
                    configure_devices=None, expected_error=None, arrival=None):
        config = mock_config()
        if configure_devices:
            config["robot"]["startup_enable_timeout_sec"] = 0.1
        robot, devices, cameras, events = mock_hardware(
            config, lambda factory: replay.ReplayRobot(config, sdk_factory=factory, arrival=arrival),
            refusal=refusal,
        )
        if configure_devices:
            configure_devices(devices)
        recorder = Recorder(config)
        session = replay.ReplaySession(
            config, robot, CAPS, 8, actions, recording=recorder, cameras=cameras,
            command_schedule=[(0.12, "p")] if schedule is None else schedule,
        )
        replies = []
        poll = session.policy.poll

        def observe_reply():
            reply = poll()
            if reply is not None:
                replies.append((reply["metadata"]["request_id"], len(reply["actions"])))
            return reply

        original_start = session.start

        def start():
            original_start()
            if stale_after_start:
                devices["fake1"].stale = True

        outcome = None
        with (patch.object(session, "start", start),
              patch.object(robot, "joint_limits", return_value=LIMITS),
              patch.object(session.policy, "poll", observe_reply),
              patch.object(replay, "CHECK_REPLAY_LIMITS", checks)):
            if refusal is not None:
                expected_error = "Piper SDK refused move_js"
            elif stale_after_start:
                expected_error = "Stale or unavailable"
            if expected_error:
                with self.assertRaisesRegex(RobotError, expected_error):
                    session.run(max_steps=len(actions), max_seconds=2, interactive=False)
            else:
                outcome = session.run(max_steps=len(actions), max_seconds=50, interactive=False)
        return outcome, session, recorder, events, replies

    def assert_startup(self, events):
        self.assertEqual(events[0], ("cameras", "started"))
        for channel in ("fake0", "fake1"):
            steps = [event[1:] for event in events if event[0] == channel]
            self.assertEqual(steps[:5], [("connect",), ("mode",), ("enable",), ("speed", 10), ("feedback",)])
            self.assertEqual(steps[-2:], [("stop",), ("disconnect",)])
        self.assertEqual(events[-1], ("cameras", "closed"))

    def assert_targets(self, actions, events, recorder):
        sent = [event[1:] for event in events if event[1] in ("joint", "gripper")]
        np.testing.assert_array_equal([*sent[0][1], sent[1][1], *sent[2][1], sent[3][1]], INITIAL)
        dispatch = sent[8:8 + 4 * len(actions)]
        for index, part in enumerate(actions.reshape(-1, 7)):
            joints, gripper = dispatch[2 * index:2 * index + 2]
            self.assertEqual((joints[0], gripper[0]), ("joint", "gripper"))
            np.testing.assert_array_equal([*joints[1], gripper[1]], part)
        np.testing.assert_array_equal(recorder.targets, actions)

    def test_camera_enable_hold_then_p_dispatches_unmodified_8_8_tail(self):
        actions = np.tile(np.asarray([1.25, 0.4, -0.6, 0, 0, 0, -0.001,
                                     0, 0.3, -0.5, 0, 0, 0, 0.102]), (19, 1))
        actions[3, 1] += 0.4
        outcome, session, recorder, events, replies = self.run_session(actions)
        self.assert_startup(events)
        self.assertEqual(outcome["steps"], len(actions))
        self.assertIsNone(session.rejection)
        self.assert_targets(actions, events, recorder)
        self.assertEqual(replies, [("replay-episode-8-frame-0", 8),
                                   ("replay-episode-8-frame-8", 8),
                                   ("replay-episode-8-frame-16", 3)])
        inside = np.diff(recorder.times[:8])
        self.assertAlmostEqual(float(np.median(inside)), 1 / 30, delta=0.010)
        self.assertGreater(recorder.times[8] - recorder.times[7], np.median(inside) * 1.5)
        self.assertIs(replay.session_module.validate_trajectory, validate_trajectory)

    def test_arrival_uses_follower_feedback_and_excludes_hold_sends(self):
        actions = np.tile(INITIAL, (9, 1))
        actions[2, 0] += 0.03
        arrival = replay.ReplayArrival(30, 8, 0.002, 0.001)
        self.run_session(actions, arrival=arrival)
        report = arrival.report()
        self.assertEqual(report["attempted_frames"], 9)
        self.assertTrue(report["frames"][7]["interchunk"])
        failed = report["frames"][2]
        self.assertEqual(failed["status"]["before_replacement_or_stop"], "not_observed")
        self.assertAlmostEqual(
            failed["sides"]["left"]["before_replacement_or_stop"]["last_feedback"]["joint_error_rad"][0],
            -0.03,
        )
        self.assertEqual(report["frames"][1]["status"]["before_replacement_or_stop"], "reached")
        final = report["final_attempted_frame"]
        self.assertEqual(final["frame_index"], 8)
        self.assertEqual(final["sides"]["left"]["boundary"], "non_replay_send")

    def test_arrival_distinguishes_late_interchunk_feedback_and_stale_samples(self):
        arrival = replay.ReplayArrival(30, 8, 0.002, 0.001)
        arrival.channels = {"left": "fake0", "right": "fake1"}
        for frame, start in ((7, 1.0), (8, 1.065), (None, 1.10)):
            arrival.frame = frame
            for channel in arrival.channels.values():
                for kind, value in (("joint", [0.0] * 6), ("gripper", 0.0)):
                    with patch("time.monotonic", return_value=start):
                        call = arrival.begin_call(channel, kind, value)
                    call["return_mono_sec"] = start + 0.001

        for observed, stamp, joint in ((1.01, 1.0005, 0.0), (1.025, 1.02, 0.01),
                                       (1.027, 1.02, 0.01), (1.051, 1.05, 0.0)):
            pair = [joint, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            arrival.observe(dict(sampled_mono_sec=observed, sampled_wall_sec=observed,
                                 sdk_wall_sec=[stamp] * 4, actual=pair * 2))
        report = arrival.report()
        frame = report["frames"][0]
        self.assertEqual(frame["status"], dict(within_nominal_period="not_observed",
                                                before_replacement_or_stop="reached"))
        self.assertEqual(frame["sides"]["left"]["before_replacement_or_stop"]["distinct_samples"], 2)
        self.assertEqual(report["interchunk_frames_reached_after_nominal_period"], 1)
        self.assertEqual(report["final_attempted_frame"]["status"]["before_replacement_or_stop"], "unknown")

    def test_arrival_sdk_refusal_stays_unknown(self):
        actions = np.tile(INITIAL, (8, 1))
        actions[0, 0] = 0.96
        arrival = replay.ReplayArrival(30, 8, 0.002, 0.001)
        self.run_session(actions, arrival=arrival, refusal=0.96)
        final = arrival.report()["final_attempted_frame"]
        self.assertIsNone(final["sides"]["left"]["dispatch"]["joint"]["return_mono_sec"])
        self.assertEqual(final["status"]["before_replacement_or_stop"], "unknown")

    def test_software_limit_checks_can_be_restored(self):
        actions = np.tile(INITIAL, (8, 1))
        actions[:, 0] = 1.25
        outcome, session, recorder, events, _ = self.run_session(actions, checks=True)
        self.assert_startup(events)
        self.assertEqual(outcome["steps"], 0)
        self.assertIn("joint position limits", session.rejection)
        self.assertEqual(recorder.targets, [])
        self.assertIs(replay.session_module.validate_trajectory, validate_trajectory)

    def test_nonfinite_chunk_is_rejected_before_any_recorded_target(self):
        actions = np.tile(INITIAL, (8, 1))
        actions[3, 0] = np.nan
        outcome, session, recorder, events, _ = self.run_session(actions)
        self.assert_startup(events)
        self.assertEqual(outcome["steps"], 0)
        self.assertIn("Expected a finite", session.rejection)
        self.assertEqual(recorder.targets, [])

    def test_sdk_refusal_stops_both_followers(self):
        actions = np.tile(INITIAL, (8, 1))
        actions[:, 0] = 0.96
        _, _, recorder, events, _ = self.run_session(actions, refusal=0.96)
        self.assert_startup(events)
        self.assertEqual(recorder.targets, [])
        refused = events.index(("fake0", "joint", actions[0, :6].tolist()))
        self.assertEqual(events[refused + 1], ("fake0", "stop"))

    def test_fresh_enabled_driver_bits_do_not_override_startup_estop(self):
        def estop(devices):
            for device in devices.values():
                device.force_enabled_feedback = True
                device.arm_status = replay.PIPER_EMERGENCY_STOP

        actions = np.tile(INITIAL, (8, 1))
        _, _, recorder, events, replies = self.run_session(
            actions, configure_devices=estop,
            expected_error="EMERGENCY_STOP in arm_status for left/follower, right/follower",
        )
        self.assertEqual(recorder.targets, [])
        self.assertEqual(replies, [])
        for channel in ("fake0", "fake1"):
            self.assertEqual([event[1] for event in events if event[0] == channel],
                             ["connect", "disconnect"])

    def test_estop_after_enable_prevents_startup_hold(self):
        def estop_after_enable(devices):
            devices["fake1"].arm_status_on_enable = replay.PIPER_EMERGENCY_STOP

        actions = np.tile(INITIAL, (8, 1))
        _, _, recorder, events, replies = self.run_session(
            actions, configure_devices=estop_after_enable,
            expected_error="EMERGENCY_STOP in arm_status for right/follower",
        )
        self.assertEqual(recorder.targets, [])
        self.assertEqual(replies, [])
        for channel in ("fake0", "fake1"):
            self.assertEqual([event[1] for event in events if event[0] == channel],
                             ["connect", "mode", "enable", "speed", "stop", "disconnect"])

    def test_missing_and_stale_startup_arm_status_prevent_enable(self):
        def unavailable(devices):
            devices["fake0"].arm_status = None
            devices["fake1"].arm_status_age = 1.0

        actions = np.tile(INITIAL, (8, 1))
        _, _, recorder, events, replies = self.run_session(
            actions, configure_devices=unavailable,
            expected_error="before enable: left/follower missing arm_status feedback, right/follower stale arm_status feedback",
        )
        self.assertEqual(recorder.targets, [])
        self.assertEqual(replies, [])
        self.assertNotIn("enable", [event[1] for event in events])

    def test_estop_during_replay_discards_queued_targets(self):
        def estop_on_first_motion(devices):
            devices["fake0"].arm_status_on_target = 0.95

        actions = np.tile(INITIAL, (8, 1))
        actions[:, 0] = 0.95
        _, _, recorder, events, replies = self.run_session(
            actions, configure_devices=estop_on_first_motion,
            expected_error="EMERGENCY_STOP in arm_status for left/follower",
        )
        self.assert_startup(events)
        np.testing.assert_array_equal(recorder.targets, actions[:1])
        self.assertEqual(replies, [("replay-episode-8-frame-0", 8)])

    def test_fresh_feedback_and_operator_p_are_required(self):
        actions = np.tile(INITIAL, (8, 1))
        _, _, recorder, events, replies = self.run_session(actions, stale_after_start=True)
        self.assert_startup(events)
        self.assertEqual(recorder.targets, [])
        self.assertEqual(replies, [])
        outcome, _, recorder, events, replies = self.run_session(actions, schedule=[(0.12, "q")])
        self.assert_startup(events)
        self.assertEqual(outcome["steps"], 0)
        self.assertEqual(replies, [])
        self.assertEqual([event[1] for event in events].count("joint"), 2)

    def test_default_preview_is_read_only_with_raw_targets(self):
        actions = np.tile(INITIAL, (8, 1))
        with (patch.object(replay, "load_episode", return_value=(8, actions, actions.copy())),
              patch.object(replay, "load_limits", return_value=(CAPS, LIMITS)),
              patch.object(replay, "ReplayRobot") as robot,
              patch.object(replay, "ReplaySession") as session,
              redirect_stdout(io.StringIO()) as output):
            self.assertEqual(replay.main([]), 0)
        robot.assert_not_called()
        session.assert_not_called()
        self.assertIn('"replay_physical_admission_checks": false', output.getvalue())

    @unittest.skipUnless(DATASET_TEST_PATH, "Set PIPER_REPLAY_TEST_DATASET to run the full episode")
    def test_stand_up_bottles_raw_dispatches_all_651_frames(self):
        config = mock_config()
        episode, actions, _ = replay.load_episode(
            Path(DATASET_TEST_PATH), replay.component_names(config["robot"]["arms"]),
            config["workflow"]["policy_hz"],
        )
        self.assertEqual((episode, len(actions)), (8, 651))
        outcome, session, recorder, events, replies = self.run_session(actions)
        self.assert_startup(events)
        self.assertEqual(outcome["steps"], 651)
        self.assertIsNone(session.rejection)
        self.assertEqual(replies[-1], ("replay-episode-8-frame-648", 3))
        self.assert_targets(actions, events, recorder)


if __name__ == "__main__":
    unittest.main()
