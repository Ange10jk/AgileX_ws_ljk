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

from piper_control.config import ConfigError, load_config
from piper_control.hardware.cameras import CameraRig
from piper_control.ipc import _decode, _owned, read_message, write_message
from piper_control.recording.client import CollectionError, CollectorClient
from piper_control.recording.session import RecordingSession
from piper_control.recording.ros_worker import serve
from piper_control.workflows.session import Session


ROOT = Path(__file__).resolve().parents[1]
FAKE_WORKER = '''
import sys, json, time, types
from pathlib import Path
sys.path.insert(0, ROOT_PATH)
from piper_control.ipc import _encode
from piper_control.recording import ros_worker
serve = ros_worker.serve
class Bridge:
    def __init__(self, cfg):
        self.options=cfg['record']
        self.path=None
        self._publish_bridge = None
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
    def publish_images(self, data):
        if self.options.get('test_fail_publish'):
            raise OSError('image publish failed')
        if any(frame["stamp"] == 20.0 for frame in data.values()):
            from piper_control.recording.ros_bridge import ROSBridge
            self._publish_bridge = ROSBridge({"record": self.options})
            self._publish_bridge._started = True
            self._publish_bridge.messages = {"image": object}

            def blocked_publish(message):
                (self.path / "publish_blocked").touch()
                while not (self.path / "release_publish").exists():
                    time.sleep(0.005)

            def create_publisher(*args, queue_size):
                assert queue_size is None
                return types.SimpleNamespace(
                    publish=blocked_publish, unregister=lambda: self.log('unregistered', {}),
                )
            self._publish_bridge.rospy = types.SimpleNamespace(
                Publisher=create_publisher, is_shutdown=lambda: False,
            )
            self._publish_bridge._emit("cameras/cam_r/color/image_raw", "image", object(), None)
        self.log('images', data)
    def publish_action(self, action, source, metadata): self.log('action', {'action':action,'source':source})
    def publish_event(self, data):
        if data.get('type')=='fail': raise OSError('collector disk failed')
        if data.get('type')=='block': time.sleep(10)
        self.log('event',data)
    def check_health(self): pass
    def close(self):
        if self._publish_bridge is not None:
            self._publish_bridge.close()
        manifest=json.loads((self.path/'manifest.json').read_text()) if (self.path/'manifest.json').exists() else {}
        self.log('closed', {'manifest_status':manifest.get('status'),
            'hardware_imports':[m for m in sys.modules if m.startswith('piper_control.hardware')]})
def cameras(observation):
    from piper_control.hardware.cameras import CameraRig
    sys.modules['pyrealsense2'] = types.ModuleType('pyrealsense2')
    root = Path(__file__).parent
    class TestRig(CameraRig):
        def close(self, *, publish_timeout):
            if observation.get('test_short_publish_timeout'):
                publish_timeout = 0.05
            super().close(publish_timeout=publish_timeout)

        def _capture(self, name, config, rs, np):
            if name == 'cam_r' and observation.get('test_delay_right'):
                while not self._stop.wait(0.005) and not (root/'release_right').exists():
                    pass
                if self._stop.is_set():
                    return
            pixels = np.zeros((2, 3, 3), dtype=np.uint8)
            for stamp in (10.0, 11.0):
                self._store_frame(name, {'stamp': stamp, 'rgb': pixels})
            while not self._stop.wait(0.005):
                if name == 'cam_r' and ((root/'overflow').exists() or (root/'block_publish').exists()):
                    self._store_frame(name, {'stamp': 20.0, 'rgb': pixels})
                    if (root/'block_publish').exists():
                        self._stop.wait()
                        return
                    while not self._stop.wait(0.005) and not (root/'publish_blocked').exists():
                        pass
                    try:
                        for stamp in range(21, 41):
                            self._store_frame(name, {'stamp': float(stamp), 'rgb': pixels})
                    except RuntimeError as exc:
                        with self._lock:
                            self._errors[name] = exc
                        return
                if (root/'disconnect').exists() and name == 'cam_l':
                    with self._lock:
                        self._errors[name] = RuntimeError('test camera disconnected')
                    return
            if not (root/'overflow').exists():
                self._store_frame(name, {'stamp': 12.0, 'rgb': pixels})
    return TestRig(observation)

ros_worker.serve = lambda incoming, outgoing, poll_input: serve(
    incoming, outgoing, bridge_factory=Bridge, camera_factory=cameras, poll_input=poll_input,
)
ros_worker.main()
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

    def worker_camera_config(self, root, **options):
        record = dict(self.config(root, "ros_publish")["record"], camera_in_ros_worker=True, **options)
        return load_config(ROOT / "configs" / "RoboDojo" / "collect_stand_up_bottles.yaml", {
            "observation": {"startup_timeout_sec": 1.5},
            "record": record,
        })

    def audit(self, root):
        return [json.loads(line) for line in
                (Path(root) / "collector.audit.jsonl").read_text().splitlines()]

    def test_worker_camera_option_is_opt_in_and_rejects_pixel_workflows(self):
        path = ROOT / "configs" / "collect_dual_RoboDojo.yaml"
        self.assertFalse(load_config(path)["record"]["camera_in_ros_worker"])
        with self.assertRaisesRegex(ConfigError, "requires collect/teleop with ros_publish"):
            load_config(path, {
                "record": {"camera_in_ros_worker": True},
                "workflow": {"mode": "deploy", "max_episode_sec": 120},
            })
        with self.assertRaisesRegex(ConfigError, "requires collect/teleop with ros_publish"):
            load_config(path, {
                "record": {"camera_in_ros_worker": True, "backend": "ros"},
                "workflow": {"max_episode_sec": 120},
            })

    def test_latest_camera_queue_policy_requires_worker_cameras(self):
        path = ROOT / "configs" / "collect_dual_RoboDojo.yaml"
        self.assertEqual(load_config(path)["record"]["camera_queue_policy"], "strict")
        with self.assertRaisesRegex(ConfigError, "camera_queue_policy must be strict or latest"):
            load_config(path, {"record": {"camera_queue_policy": "fifo"}})
        with self.assertRaisesRegex(ConfigError, "latest requires record.camera_in_ros_worker=true"):
            load_config(path, {"record": {"camera_queue_policy": "latest"}})

    def test_worker_waits_for_all_cameras_and_reports_idle_disconnect(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.worker_camera_config(root)
            cfg["observation"]["test_delay_right"] = True
            client = self.factory(root)(cfg, None)
            with ThreadPoolExecutor(max_workers=1) as executor:
                started = executor.submit(client.start)
                try:
                    deadline = time.monotonic() + 2.
                    path = Path(root) / "collector.audit.jsonl"
                    while time.monotonic() < deadline:
                        if path.exists() and '"cam_l"' in path.read_text():
                            break
                        time.sleep(0.01)
                    self.assertIn('"cam_l"', path.read_text())
                    self.assertFalse(started.done())
                finally:
                    (Path(root) / "release_right").touch()
                    started.result(timeout=4.)

            try:
                for name in ("cam_h", "cam_l", "cam_r"):
                    self.assertTrue(any(name in row["payload"] for row in self.audit(root)
                                        if row["kind"] == "images"))
                (Path(root) / "disconnect").touch()
                client._reader_thread.join(timeout=2.)
                with self.assertRaisesRegex(CollectionError, "cam_l.*test camera disconnected"):
                    client.check_health()
            finally:
                with self.assertRaisesRegex(CollectionError, "cam_l.*test camera disconnected"):
                    client.close()
            client._writer_thread.join(timeout=1.)
            self.assertFalse(client._writer_thread.is_alive())
            self.assertEqual(self.audit(root)[-1]["kind"], "closed")

    def test_worker_reports_overflow_before_a_blocked_publish_finishes(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryFile(mode="w+") as stderr:
            cfg = self.worker_camera_config(root, camera_queue_size=8, queue_diagnostics=False)
            client = self.factory(root)(cfg, None)
            popen = subprocess.Popen
            with mock.patch("piper_control.recording.client.subprocess.Popen",
                            side_effect=lambda *args, **kwargs: popen(*args, stderr=stderr, **kwargs)):
                client.start()

            try:
                (Path(root) / "overflow").touch()
                client._reader_thread.join(timeout=2.)
                with self.assertRaisesRegex(CollectionError, "cam_r.*queue full.*capacity 8"):
                    client.check_health()
                self.assertNotIn("unregistered", [row["kind"] for row in self.audit(root)])
            finally:
                (Path(root) / "release_publish").touch()
                with self.assertRaisesRegex(CollectionError, "cam_r.*queue full"):
                    client.close()
            audit = self.audit(root)
            self.assertEqual([row["kind"] for row in audit[-2:]], ["unregistered", "closed"])
            self.assertIn(28.0, [row["payload"]["cam_r"]["stamp"] for row in audit
                                 if row["kind"] == "images" and "cam_r" in row["payload"]])
            self.assertFalse((Path(root) / "diagnostics").exists())
            stderr.seek(0)
            self.assertEqual(stderr.read(), "")

    def test_worker_exits_without_unregistering_a_blocked_publisher(self):
        for trigger, error in (
            ("block_publish", "Camera recording shutdown timed out"),
            ("overflow", "cam_r.*queue full"),
        ):
            with self.subTest(trigger=trigger), tempfile.TemporaryDirectory() as root:
                cfg = self.worker_camera_config(root, camera_queue_size=8)
                cfg["observation"]["test_short_publish_timeout"] = True
                client = self.factory(root)(cfg, None).start()
                (Path(root) / trigger).touch()
                try:
                    deadline = time.monotonic() + 2.
                    blocked = Path(root) / "publish_blocked"
                    while not blocked.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(blocked.exists())
                    if trigger == "overflow":
                        client._reader_thread.join(timeout=2.)
                finally:
                    with self.assertRaisesRegex(CollectionError, error):
                        client.close()
                self.assertEqual(client.process.returncode, 1)
                self.assertFalse({"unregistered", "closed"} & {row["kind"] for row in self.audit(root)})

    def test_latest_worker_reports_overwrites_on_close_without_diagnostics(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryFile(mode="w+") as stderr:
            cfg = self.worker_camera_config(root, camera_queue_policy="latest")
            client = self.factory(root)(cfg, None)
            popen = subprocess.Popen
            with mock.patch("piper_control.recording.client.subprocess.Popen",
                            side_effect=lambda *args, **kwargs: popen(*args, stderr=stderr, **kwargs)):
                client.start()
            client.close()

            audit = self.audit(root)
            self.assertEqual(audit[-1]["kind"], "closed")
            self.assertEqual(client.process.returncode, 0)
            stderr.seek(0)
            report = stderr.read()
            self.assertEqual(report.count("[ROS cameras final]"), 1)
            for name in ("cam_h", "cam_l", "cam_r"):
                stamps = [row["payload"][name]["stamp"] for row in audit
                          if row["kind"] == "images" and name in row["payload"]]
                self.assertEqual(stamps, sorted(set(stamps)))
                self.assertEqual(stamps[-1], 12.0)
                self.assertRegex(report, rf"{name} captured=3 .* overwritten=\d+ unpublished=0")

    def test_worker_camera_publish_error_blocks_readiness(self):
        for policy in ("strict", "latest"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as root:
                cfg = self.worker_camera_config(root, camera_queue_policy=policy, test_fail_publish=True)
                client = self.factory(root)(cfg, None)
                with self.assertRaisesRegex(CollectionError, "Publishing camera .* image publish failed"):
                    client.start()
                self.assertIsNotNone(client.process.returncode)
                self.assertEqual(self.audit(root)[-1]["kind"], "closed")

    def test_human_worker_cameras_do_not_construct_parent_rig_or_enqueue_images(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.worker_camera_config(root)
            factory = self.factory(root)

            def tracked_client(config, directory):
                client = factory(config, directory)
                client._enqueue = mock.Mock(wraps=client._enqueue)
                return client

            recording = RecordingSession(cfg, collector_factory=tracked_client)
            with mock.patch("piper_control.workflows.session.MockCameras") as parent_rig:
                session = Session(cfg, mock=True, recording=recording)
                result = session.run(max_steps=2, max_seconds=2., interactive=False)
            parent_rig.assert_not_called()
            self.assertIsNone(session.cameras)
            self.assertFalse(session._camera_recording)
            self.assertEqual(result["steps"], 2)
            self.assertEqual(
                {call.args[0] for call in recording.collector._enqueue.call_args_list},
                {"state", "action", "event"},
            )
            audit = self.audit(root)
            for name in ("cam_h", "cam_l", "cam_r"):
                stamps = [row["payload"][name]["stamp"] for row in audit
                          if row["kind"] == "images" and name in row["payload"]]
                self.assertEqual(stamps, [10.0, 11.0, 12.0])
            events = [row["payload"] for row in audit if row["kind"] == "event"]
            final = next(event["final_observation"] for event in events
                         if event["type"] == "episode_end")
            self.assertEqual(final["camera_stamps"], dict.fromkeys(("cam_h", "cam_l", "cam_r"), 11.0))
            self.assertNotIn("frames", final)
            self.assertEqual(audit[-1]["kind"], "closed")
            self.assertFalse((Path(root) / "diagnostics").exists())

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
