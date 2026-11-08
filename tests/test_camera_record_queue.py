"""Recording delivery under a slow camera sink."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
import io
import sys
import threading
import types
import unittest
from unittest import mock

from piper_control.hardware.cameras import CameraRig


class CameraRecordingQueueTests(unittest.TestCase):
    def test_preserves_frames_and_drains_on_close(self):
        received = []
        rig = CameraRig({})
        rig.record_to(received.append, capacity=4)
        for stamp in range(4):
            rig._store_frame('cam', {'stamp': stamp})
        self.assertEqual(rig._frames['cam']['stamp'], 3)
        rig.start()
        rig.close()
        self.assertEqual([item['cam']['stamp'] for item in received], list(range(4)))

    def test_latest_replaces_pending_groups_without_changing_camera_order(self):
        rig = CameraRig({
            "camera_map": {"left": "left", "right": "right"},
            "cameras": {"left": {}, "right": {}},
        })
        in_flight = threading.Event()
        release = threading.Event()
        received = []
        first = {"stamp": 10.0}
        newest_left = {"stamp": 12.0}
        newest_right = {
            "stamp": 22.0, "depth_stamp": 21.99, "aligned_depth_stamp": 21.98,
            "rgb": object(), "depth": object(), "aligned_depth": object(),
            "camera_info": object(), "depth_camera_info": object(),
            "aligned_depth_camera_info": object(),
        }

        def publish(frames):
            if frames.get("left") is first:
                in_flight.set()
                if not release.wait(5):
                    raise AssertionError("Timed out waiting to release camera publisher")
            received.append(frames)

        rig.record_to(publish, capacity=1, queue_policy="latest")
        self.addCleanup(rig.close)
        self.addCleanup(release.set)
        rig._store_frame("left", first)
        rig._store_frame("right", {"stamp": 20.0})
        with mock.patch.object(rig, "_capture"), mock.patch.dict(
            sys.modules, {"pyrealsense2": types.ModuleType("pyrealsense2")}
        ):
            rig.start()
        self.assertTrue(in_flight.wait(2))

        rig._store_frame("left", {"stamp": 11.0})
        rig._store_frame("right", {"stamp": 21.0})
        rig._store_frame("left", newest_left)
        rig._store_frame("right", newest_right)
        with ThreadPoolExecutor(max_workers=1) as executor:
            closing = executor.submit(rig.close)
            try:
                self.assertTrue(rig._publisher_stop.wait(2))
                self.assertFalse(closing.done())
            finally:
                release.set()
            closing.result(timeout=2)

        self.assertEqual(received, [{"left": first}, {"right": newest_right}, {"left": newest_left}])
        self.assertIs(received[1]["right"], newest_right)

        output = io.StringIO()
        with redirect_stderr(output):
            rig.report_recording()
            self.assertEqual(output.getvalue(), "")
            rig.report_recording(final=True)
        self.assertIn("queue=0/2 peak=2", output.getvalue())
        self.assertRegex(output.getvalue(), r"left captured=3 published=2 .* overwritten=1 unpublished=0")
        self.assertRegex(output.getvalue(), r"right captured=3 published=1 .* overwritten=2 unpublished=0")

    def test_overflow_is_explicit(self):
        rig = CameraRig({"camera_map": {"rgb": "cam"}, "cameras": {"cam": {}}})
        rig.record_to(lambda frames: None, capacity=1, diagnostics=True)
        rig._store_frame('cam', {'stamp': 1})
        with self.assertRaisesRegex(RuntimeError, 'queue full'):
            rig._store_frame('cam', {'stamp': 2})
        output = io.StringIO()
        with redirect_stderr(output):
            rig.report_recording(final=True)
        self.assertIn("queue=1/1 peak=1", output.getvalue())
        self.assertIn("cam captured=2 published=0 overflow=1", output.getvalue())

    def test_publisher_failure_propagates(self):
        def fail(frames):
            raise RuntimeError('sink failed')
        rig = CameraRig({})
        rig.record_to(fail)
        rig._store_frame('cam', {'stamp': 1})
        rig.start()
        rig._publisher.join(timeout=1)
        with self.assertRaisesRegex(RuntimeError, 'sink failed'):
            rig.close()
