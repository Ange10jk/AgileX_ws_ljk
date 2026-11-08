"""RealSense capture without ROS dependencies or hardware access on import.

The rig retains the latest immutable frame per camera. ``snapshot`` returns that
cache with the actual capture timestamps. SDK hardware-clock timestamps are not
relabelled with the computer's receive time.
"""

import math
import queue
import sys
import threading
import time


def _selected(config):
    names = list(dict.fromkeys(config.get("camera_map", {}).values()))
    cameras = config.get("cameras", {})
    return {name: cameras[name] for name in names}


def _validate_depth_options(config, selected):
    for name, camera in selected.items():
        depth = camera.get("depth", config.get("depth", False))
        align_depth = camera.get("align_depth", config.get("align_depth", False))
        if align_depth and not depth:
            raise ValueError("Camera {}: align_depth requires depth=true".format(name))


def _capture_stamp(frame, rs):
    if frame.get_frame_timestamp_domain() == rs.timestamp_domain.hardware_clock:
        raise ValueError("Camera timestamp has not converged to the SDK global clock")
    stamp = float(frame.get_timestamp()) * 0.001
    if not math.isfinite(stamp) or stamp <= 0:
        raise ValueError("Camera capture timestamp is invalid")
    return stamp


def _intrinsics(frame, serial):
    intrinsics = frame.get_profile().as_video_stream_profile().get_intrinsics()
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "coeffs": list(intrinsics.coeffs),
        "distortion_model": str(intrinsics.model),
        "serial": serial,
    }


class CameraPublishTimeout(RuntimeError):
    """The recording sink may still be running after camera shutdown."""


class CameraRig:
    def __init__(self, observation_cfg):
        self.config = dict(observation_cfg)
        self.selected = _selected(self.config)
        _validate_depth_options(self.config, self.selected)
        self.startup_timeout = float(self.config.get("startup_timeout_sec", 15.0))
        if not math.isfinite(self.startup_timeout) or self.startup_timeout <= 0:
            raise ValueError("Camera startup_timeout_sec must be finite and positive")
        self._frames = {}
        self._errors = {}
        self._threads = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._started = False
        self._record_queue = None
        self._record_sink = None
        self._publisher = None
        self._publisher_stop = threading.Event()
        self._published_cameras = set()
        self._record_stats = None
        self._report_since = 0.0
        self._queue_peak = 0

    def record_to(self, sink, capacity=32, *, diagnostics=False):
        """Enable loss-intolerant capture delivery before starting cameras."""
        if self._started:
            raise RuntimeError("Configure camera recording before start")
        if capacity < 1:
            raise ValueError("Camera recording queue capacity must be positive")
        self._record_sink = sink
        self._record_queue = queue.Queue(maxsize=capacity)
        if diagnostics:
            self._record_stats = {
                name: dict(captured=0, published=0, overflow=0, publish_max_ms=0.0)
                for name in self.selected
            }
            self._report_since = time.monotonic()

    def report_recording(self, *, final=False):
        if self._record_stats is None:
            return
        now = time.monotonic()
        if not final and now - self._report_since < 1.0:
            return
        self._report_since = now
        with self._lock:
            stats = {name: dict(counts) for name, counts in self._record_stats.items()}
            size, peak = self._record_queue.qsize(), self._queue_peak
        print(
            "[ROS cameras{}] queue={}/{} peak={}".format(
                " final" if final else "", size, self._record_queue.maxsize, peak),
            file=sys.stderr, flush=True,
        )
        for name, counts in stats.items():
            line = (
                "captured={captured} published={published} overflow={overflow} "
                "publish_max_ms={publish_max_ms:.1f}".format(**counts)
            )
            if final:
                line += " unpublished={}".format(counts["captured"] - counts["published"])
            print("  {} {}".format(name, line), file=sys.stderr, flush=True)

    def _store_frame(self, name, frame):
        with self._lock:
            stats = self._record_stats[name] if self._record_stats is not None else None
            if stats is not None:
                stats["captured"] += 1
            if self._record_queue is not None and not self._publisher_stop.is_set():
                try:
                    self._record_queue.put_nowait((name, frame))
                except queue.Full as exc:
                    if stats is not None:
                        stats["overflow"] += 1
                        self._queue_peak = self._record_queue.maxsize
                    error = RuntimeError(
                        "Camera recording queue full (capacity {} frame groups); cannot drop frames".format(
                            self._record_queue.maxsize
                        )
                    )
                    self._errors.setdefault(name, error)
                    raise error from exc
                if stats is not None:
                    self._queue_peak = max(self._queue_peak, self._record_queue.qsize())
            self._frames[name] = frame

    def _publish_frames(self):
        while not self._publisher_stop.is_set() or not self._record_queue.empty():
            try:
                name, frame = self._record_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if self._record_stats is not None:
                started = time.monotonic()
            try:
                self._record_sink({name: frame})
            except BaseException as exc:
                error = RuntimeError("Publishing camera {} failed: {}".format(name, exc))
                error.__cause__ = exc
                with self._lock:
                    self._errors["recording"] = error
                return
            with self._lock:
                self._published_cameras.add(name)
                if self._record_stats is not None:
                    stats = self._record_stats[name]
                    stats["published"] += 1
                    stats["publish_max_ms"] = max(
                        stats["publish_max_ms"], (time.monotonic() - started) * 1000
                    )

    def start(self, *, wait_for_publish=False):
        if self._started:
            raise RuntimeError("Camera rig has already started")
        if wait_for_publish and self._record_sink is None:
            raise RuntimeError("Camera startup requires a recording sink to wait for publishing")
        self._started = True
        if self._record_sink is not None:
            self._publisher = threading.Thread(
                target=self._publish_frames, name="camera-recording", daemon=True)
            self._publisher.start()
        if not self.selected:
            return self
        try:
            import numpy as np
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "RealSense capture requires numpy and pyrealsense2"
            ) from exc
        for name, config in self.selected.items():
            thread = threading.Thread(
                target=self._capture,
                args=(name, config, rs, np),
                name="camera-" + name,
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()
        deadline = time.monotonic() + self.startup_timeout
        try:
            while time.monotonic() < deadline:
                self.check_health()
                with self._lock:
                    ready = len(self._frames) == len(self.selected)
                    if wait_for_publish:
                        ready = ready and self._published_cameras.issuperset(self.selected)
                if ready:
                    self.snapshot()
                    return self
                self._stop.wait(0.01)
            raise RuntimeError(
                "Timed out waiting for {}frames from every configured camera".format(
                    "published " if wait_for_publish else "")
            )
        except BaseException:
            self._stop.set()
            # The ROS worker reports startup failure before draining its queue.
            if not wait_for_publish:
                self.close()
            raise

    def _capture(self, name, config, rs, np):
        running = False
        try:
            pipeline = rs.pipeline()
            sdk_config = rs.config()
            serial = str(config["serial"])
            sdk_config.enable_device(serial)
            width = int(config.get("width", self.config.get("width", 640)))
            height = int(config.get("height", self.config.get("height", 480)))
            fps = int(config.get("fps", self.config.get("fps", 15)))
            depth = bool(config.get("depth", self.config.get("depth", False)))
            align_depth = bool(
                config.get("align_depth", self.config.get("align_depth", False))
            )
            aligner = rs.align(rs.stream.color) if align_depth else None
            sdk_config.enable_stream(
                rs.stream.color, width, height, rs.format.rgb8, fps
            )
            if depth:
                sdk_config.enable_stream(
                    rs.stream.depth, width, height, rs.format.z16, fps
                )
            device = sdk_config.resolve(rs.pipeline_wrapper(pipeline)).get_device()
            if device.get_info(rs.camera_info.serial_number) != serial:
                raise RuntimeError(
                    "Requested camera serial does not match resolved device"
                )
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            profile = pipeline.start(sdk_config)
            running = True
            scale = (
                float(profile.get_device().first_depth_sensor().get_depth_scale())
                if depth
                else None
            )
            if depth and (not math.isfinite(scale) or scale <= 0):
                raise RuntimeError("Invalid RealSense depth scale")
            last_valid = time.monotonic()
            timeout = self.startup_timeout
            while not self._stop.is_set():
                if time.monotonic() - last_valid > timeout:
                    raise RuntimeError("Camera frame timeout waiting for a valid frame group")
                received, frames = pipeline.try_wait_for_frames(500)
                if not received:
                    continue
                color = frames.get_color_frame()
                depth_frame = frames.get_depth_frame() if depth else None
                if not color or (depth and not depth_frame):
                    continue
                aligned_depth_frame = (
                    aligner.process(frames).get_depth_frame() if aligner else None
                )
                if aligner and not aligned_depth_frame:
                    continue
                try:
                    stamp = _capture_stamp(color, rs)
                    depth_stamp = _capture_stamp(depth_frame, rs) if depth else None
                    aligned_depth_stamp = (
                        _capture_stamp(aligned_depth_frame, rs) if aligner else None
                    )
                except ValueError:
                    if time.monotonic() - last_valid > timeout:
                        raise RuntimeError(
                            "Camera has no valid global-clock timestamps"
                        )
                    continue
                now_epoch, now_mono = time.time(), time.monotonic()
                rgb = np.asanyarray(color.get_data()).copy()
                rgb.flags.writeable = False
                frame = {
                    "rgb": rgb,
                    "stamp": stamp,
                    "monotonic": now_mono - (now_epoch - stamp),
                    "camera_info": _intrinsics(color, serial),
                    "timestamp_source": "sdk_global_or_system_clock",
                    "frame_number": int(color.get_frame_number()),
                }
                if depth:
                    depth_data = np.asanyarray(depth_frame.get_data()).copy()
                    depth_data.flags.writeable = False
                    frame.update(
                        depth=depth_data,
                        depth_stamp=depth_stamp,
                        depth_monotonic=now_mono - (now_epoch - depth_stamp),
                        depth_scale_m=scale,
                        depth_camera_info=_intrinsics(depth_frame, serial),
                    )
                if aligner:
                    aligned_depth = np.asanyarray(aligned_depth_frame.get_data()).copy()
                    aligned_depth.flags.writeable = False
                    frame.update(
                        aligned_depth=aligned_depth,
                        aligned_depth_stamp=aligned_depth_stamp,
                        aligned_depth_monotonic=now_mono
                        - (now_epoch - aligned_depth_stamp),
                        aligned_depth_camera_info=_intrinsics(color, serial),
                    )
                self._store_frame(name, frame)
                last_valid = time.monotonic()
        except BaseException as exc:
            with self._lock:
                self._errors.setdefault(name, exc)
        finally:
            if running:
                try:
                    pipeline.stop()
                except Exception as exc:
                    with self._lock:
                        self._errors.setdefault(name, exc)

    def check_health(self):
        with self._lock:
            errors = dict(self._errors)
        if errors:
            name, error = next(iter(errors.items()))
            raise RuntimeError("Camera {} failed: {}".format(name, error)) from error

    def snapshot(self):
        if not self._started or self._stop.is_set():
            raise RuntimeError("Camera rig is not running")
        self.check_health()
        with self._lock:
            result = {name: dict(frame) for name, frame in self._frames.items()}
        for name in self.selected:
            if name not in result:
                raise RuntimeError("Camera {} has no captured frame".format(name))
        return result

    def rgb_stamps(self):
        with self._lock:
            return {name: frame["stamp"] for name, frame in self._frames.items()}

    def close(self, *, publish_timeout=20.0):
        self._stop.set()
        deadline = time.monotonic() + 3.0
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        still_running = [thread.name for thread in self._threads if thread.is_alive()]
        with self._lock:
            self._publisher_stop.set()
        if self._publisher is not None:
            self._publisher.join(timeout=publish_timeout)
            if self._publisher.is_alive():
                raise CameraPublishTimeout("Camera recording shutdown timed out")
        if still_running:
            raise RuntimeError("Camera shutdown timed out: " + ", ".join(still_running))
        # A capture thread can fail while stopping its pipeline after the last
        # successful health check. Propagate that error before the recording
        # session marks its manifest complete.
        self.check_health()


class MockCameras:
    """Deterministic synthetic images, explicitly marked as simulated captures."""

    def __init__(self, observation_cfg):
        self.config = dict(observation_cfg)
        self.selected = _selected(self.config)
        _validate_depth_options(self.config, self.selected)
        self._started = False
        self._frames = {}
        self._next_capture = 0.0

    def start(self):
        self._started = True
        self._next_capture = 0.0
        self.snapshot()
        return self

    def snapshot(self):
        if not self._started:
            raise RuntimeError("Mock cameras are not running")
        now = time.monotonic()
        if now < self._next_capture:
            return {name: dict(frame) for name, frame in self._frames.items()}
        if self.selected:
            import numpy as np

            for index, (name, camera) in enumerate(self.selected.items()):
                width = int(camera.get("width", self.config.get("width", 640)))
                height = int(camera.get("height", self.config.get("height", 480)))
                rgb = np.zeros((height, width, 3), dtype=np.uint8)
                rgb[:, :, index % 3] = 60 + index * 50 % 195
                rgb.flags.writeable = False
                info = {
                    "width": width,
                    "height": height,
                    "fx": float(width),
                    "fy": float(height),
                    "ppx": width / 2.0,
                    "ppy": height / 2.0,
                    "coeffs": [0.0] * 5,
                    "distortion_model": "brown_conrady",
                    "serial": "MOCK-" + name,
                }
                frame = {
                    "rgb": rgb,
                    "stamp": time.time(),
                    "monotonic": now,
                    "camera_info": info,
                    "timestamp_source": "mock_capture",
                }
                if camera.get("depth", self.config.get("depth", False)):
                    depth = np.full((height, width), 1000, dtype=np.uint16)
                    depth.flags.writeable = False
                    frame.update(
                        depth=depth,
                        depth_stamp=frame["stamp"],
                        depth_monotonic=now,
                        depth_scale_m=0.001,
                        depth_camera_info=dict(info),
                    )
                    if camera.get("align_depth", self.config.get("align_depth", False)):
                        aligned_depth = depth.copy()
                        aligned_depth.flags.writeable = False
                        frame.update(
                            aligned_depth=aligned_depth,
                            aligned_depth_stamp=frame["stamp"],
                            aligned_depth_monotonic=now,
                            aligned_depth_camera_info=dict(info),
                        )
                self._frames[name] = frame
        self._next_capture = now + 1.0 / float(self.config.get("fps", 15))
        return {name: dict(frame) for name, frame in self._frames.items()}

    def check_health(self):
        if not self._started:
            raise RuntimeError("Mock cameras are not running")

    def close(self):
        self._started = False
