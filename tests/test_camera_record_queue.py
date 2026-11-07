"""Recording must preserve captures even when snapshots skip ahead."""
import unittest

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

    def test_overflow_is_explicit(self):
        rig = CameraRig({})
        rig.record_to(lambda frames: None, capacity=1)
        rig._store_frame('cam', {'stamp': 1})
        with self.assertRaisesRegex(RuntimeError, 'queue full'):
            rig._store_frame('cam', {'stamp': 2})

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
