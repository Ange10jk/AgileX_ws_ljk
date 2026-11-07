from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from piper_control.config import load_config
from piper_control.hardware.cameras import CameraRig
from piper_control.ipc import _decode, _owned, read_message, write_message
from piper_control.recording.client import CollectionError, CollectorClient
from piper_control.recording.session import RecordingSession
from piper_control.recording.ros_worker import serve
from piper_control.workflows.session import Session


ROOT = Path(__file__).resolve().parents[1]
FAKE_WORKER = '''
import sys, json, signal, time
from pathlib import Path
sys.path.insert(0, ROOT_PATH)
from piper_control.ipc import _encode
from piper_control.recording.ros_worker import serve
class Bridge:
    def __init__(self, cfg):
        self.options=cfg['record']
        self.path=None
    def start(self):
        if self.options['backend']=='ros_publish':
            self.path=Path(self.options['output_dir'])
            self.log('ready', {})
    def start_bag(self, directory):
        self.path=Path(directory)
        self.log('ready', {})
    def log(self, kind, payload):
        with (self.path/'collector.audit.jsonl').open('a') as f:
            f.write(json.dumps({'kind':kind, 'payload':payload}, default=_encode)+'\\n')
    def publish_state(self, data): self.log('state', data)
    def publish_images(self, data): self.log('images', data)
    def publish_action(self, action, source, metadata): self.log('action', {'action':action,'source':source})
    def publish_event(self, data):
        if data.get('type')=='fail': raise OSError('collector disk failed')
        if data.get('type')=='block': time.sleep(10)
        self.log('event',data)
    def check_health(self): pass
    def close(self):
        manifest=json.loads((self.path/'manifest.json').read_text()) if (self.path/'manifest.json').exists() else {}
        self.log('closed', {'manifest_status':manifest.get('status'),
            'hardware_imports':[m for m in sys.modules if m.startswith('piper_control.hardware')]})
def stop(*args): raise SystemExit(143)
signal.signal(signal.SIGTERM,stop)
serve(sys.stdin.buffer,sys.stdout.buffer,bridge_factory=Bridge)
'''


class CollectionProcessTests(unittest.TestCase):
    def config(self, root, backend="ros"):
        return {"record": {"backend": backend, "output_dir": root, "min_free_gb": 0,
                            "python": sys.executable, "ros_setup": None, "queue_size": 16,
                            "startup_timeout_sec": 3., "command_timeout_sec": 2., "shutdown_timeout_sec": 2.},
                "observation": {}}

    def factory(self, root):
        script = Path(root) / "fake_collector.py"
        script.write_text(FAKE_WORKER.replace("ROOT_PATH", repr(str(ROOT))))
        def factory(cfg, directory):
            client = CollectorClient(cfg, directory)
            client._argv = lambda: [sys.executable, str(script)]
            return client
        return factory

    def test_process_owns_snapshots_deduplicates_images_and_closes_before_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            session = RecordingSession(self.config(root), collector_factory=self.factory(root)).start()
            state = np.arange(7, dtype=np.float32)
            frame = {"rgb": np.full((2, 3, 3), 17, dtype=np.uint8),
                     "stamp": time.time(), "monotonic": time.monotonic()}
            session.publish_state({"state": state})
            session.publish_images({"top": frame})
            session.publish_images({"top": frame})
            session.publish_action(state, "HUMAN", {"timestamp": frame["stamp"]})
            session.write_event({"type": "episode_end", "final_observation": {
                "state": state, "stamp": frame["stamp"], "frames": {"top": frame}}})
            state[:] = 99
            frame["rgb"][:] = 0
            session.close()
            audit = [json.loads(line, object_hook=_decode) for line in
                     (session.directory / "collector.audit.jsonl").read_text().splitlines()]
            self.assertEqual([row["kind"] for row in audit], ["ready", "state", "images", "action", "event", "closed"])
            np.testing.assert_array_equal(audit[1]["payload"]["state"], np.arange(7))
            self.assertTrue(np.all(audit[2]["payload"]["top"]["rgb"] == 17))
            self.assertNotIn("frames", audit[4]["payload"]["final_observation"])
            self.assertEqual(audit[-1]["payload"]["manifest_status"], "recording")
            self.assertEqual(audit[-1]["payload"]["hardware_imports"], [])
            self.assertEqual(json.loads((session.directory / "manifest.json").read_text())["status"], "complete")
            self.assertEqual(session.collector.process.returncode, 0)

    def test_sidecar_error_fails_session_instead_of_marking_complete(self):
        with tempfile.TemporaryDirectory() as root:
            session = RecordingSession(self.config(root), collector_factory=self.factory(root)).start()
            session.write_event({"type": "fail"})
            session.collector._writer_thread.join(timeout=3.)
            with self.assertRaises(CollectionError):
                session.check_health()
            with self.assertRaises(CollectionError):
                session.close()
            manifest = json.loads((session.directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("failed", manifest["error"])

    def test_aligned_depth_timestamp_change_reaches_collector(self):
        with tempfile.TemporaryDirectory() as root:
            client = CollectorClient(self.config(root), root)
            client._enqueue = mock.Mock()
            frame = {"stamp": 100.0, "depth_stamp": 99.9, "aligned_depth_stamp": 99.8}
            client.publish_images({"top": frame})
            client._enqueue.assert_called_once_with("images", {"top": frame})
            client._enqueue.reset_mock()
            client.publish_images({"top": frame})
            client._enqueue.assert_not_called()
            frame = dict(frame, aligned_depth_stamp=99.81)
            client.publish_images({"top": frame})
            client._enqueue.assert_called_once_with("images", {"top": frame})

    def test_bounded_queue_refuses_to_drop_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.config(root)
            cfg["record"]["queue_size"] = 1
            client = CollectorClient(cfg, root)
            client.process = types.SimpleNamespace(poll=lambda: None)
            client.publish_event({"value": 1})
            with self.assertRaisesRegex(CollectionError, "queue is full"):
                client.publish_event({"value": 2})
            self.assertEqual(client.queue.qsize(), 1)

    def test_image_copy_does_not_block_control_and_sequences_follow_fifo(self):
        with tempfile.TemporaryDirectory() as root:
            client = CollectorClient(self.config(root), root)
            client.process = types.SimpleNamespace(poll=lambda: None)
            copying, release = threading.Event(), threading.Event()
            pixels = np.full((2, 3, 3), 17, dtype=np.uint8)
            frames = {"top": {"rgb": pixels, "stamp": 1.0}}

            def copy_payload(payload):
                if "top" in payload:
                    copying.set()
                    if not release.wait(5):
                        raise AssertionError("Control messages did not finish during image copy")
                return _owned(payload)

            with mock.patch("piper_control.recording.client._owned", side_effect=copy_payload):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    image = executor.submit(client.publish_images, frames)
                    try:
                        self.assertTrue(copying.wait(5))
                        client.publish_state({"state": np.zeros(7)})
                        client.publish_action(np.zeros(7), "HUMAN")
                        client.publish_event({"type": "test"})
                        self.assertFalse(image.done())
                    finally:
                        release.set()
                    image.result()

            messages = [client.queue.get_nowait() for _ in range(4)]
            self.assertEqual([m["kind"] for m in messages], ["state", "action", "event", "images"])
            self.assertEqual([m["seq"] for m in messages], [1, 2, 3, 4])

    def test_publish_session_drains_captured_frames_before_collector_closes(self):
        class TestRig(CameraRig):
            def _capture(self, name, config, rs, np):
                for stamp in (1.0, 2.0):
                    self._store_frame(name, {"stamp": stamp, "rgb": np.zeros((2, 3, 3), dtype=np.uint8)})
                self._stop.wait()
                # A frame already being captured can finish after stop is signalled.
                self._store_frame(name, {"stamp": 3.0, "rgb": np.zeros((2, 3, 3), dtype=np.uint8)})
                self.capture_finished.set()

        with tempfile.TemporaryDirectory() as root:
            cfg = load_config(ROOT / "configs" / "collect_dual_RoboDojo.yaml", {
                "observation": {"camera_map": {"observation.images.cam_h": "cam_h"}},
                "record": self.config(root, "ros_publish")["record"],
            })
            rig = TestRig(cfg["observation"])
            rig.capture_finished = threading.Event()
            recording = RecordingSession(cfg, collector_factory=self.factory(root))
            publish_images = recording.publish_images

            def publish_after_capture(frames):
                if not rig.capture_finished.wait(5):
                    raise AssertionError("Capture did not stop before publisher drain")
                publish_images(frames)

            recording.publish_images = publish_after_capture
            session = Session(cfg, mock=True, cameras=rig, recording=recording)
            with mock.patch.dict(sys.modules, {"pyrealsense2": types.ModuleType("pyrealsense2")}):
                result = session.run(max_steps=1, max_seconds=1.0, interactive=False)
            self.assertEqual(result["steps"], 1)
            self.assertFalse(rig._publisher.is_alive())
            self.assertEqual(recording.collector.process.returncode, 0)
            audit = [json.loads(line) for line in
                     (Path(root) / "collector.audit.jsonl").read_text().splitlines()]
            stamps = [row["payload"]["cam_h"]["stamp"] for row in audit if row["kind"] == "images"]
            self.assertEqual(stamps, [1.0, 2.0, 3.0])
            self.assertEqual(audit[-1]["kind"], "closed")

    def test_worker_rejects_motion_messages_and_closes_collector(self):
        incoming, outgoing = io.BytesIO(), io.BytesIO()
        bridge = mock.Mock()
        write_message(incoming, {"kind": "start", "seq": 0, "config": {}, "directory": "/tmp"})
        write_message(incoming, {"kind": "move_joints", "seq": 1, "payload": [0.] * 7,
                                 "captured_epoch": 10., "captured_monotonic": 1.})
        incoming.seek(0)
        with mock.patch("piper_control.recording.ros_worker.traceback.print_exc"):
            serve(incoming, outgoing, bridge_factory=lambda cfg: bridge)
        outgoing.seek(0)
        self.assertEqual(read_message(outgoing)["kind"], "ready")
        self.assertIn("Unsupported collection input", read_message(outgoing)["error"])
        bridge.close.assert_called_once()
        bridge.publish_action.assert_not_called()

    def test_ros_setup_paths_are_positional_arguments(self):
        with tempfile.TemporaryDirectory() as root:
            setup = Path(root) / "ros $(not-a-command).bash"
            setup.write_text(":\n")
            cfg = self.config(root)
            cfg["record"]["ros_setup"] = str(setup)
            cfg["record"]["python"] = "/path with spaces/python"
            argv = CollectorClient(cfg, root)._argv()
            self.assertNotIn(str(setup), argv[4])
            self.assertEqual(argv[-2:], [str(setup), "/path with spaces/python"])
            self.assertEqual(argv[4], 'source "$1" && exec "$2" -m piper_control.recording.ros_worker')

    def test_none_and_files_never_start_collector(self):
        with tempfile.TemporaryDirectory() as root:
            factory = mock.Mock(side_effect=AssertionError("ROS process must stay absent"))
            none = RecordingSession(self.config(root, "none"), collector_factory=factory).start()
            none.publish_state({})
            none.write_event({"type": "test"})
            none.close()
            self.assertIsNone(none.directory)
            files = RecordingSession(self.config(root, "files"), collector_factory=factory).start()
            files.write_event({"type": "test"})
            files.close()
            factory.assert_not_called()
            self.assertEqual(json.loads((files.directory / "manifest.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
