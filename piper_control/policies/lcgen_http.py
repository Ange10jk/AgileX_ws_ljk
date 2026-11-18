"""RoboDojo's Piper HTTP policy and physical action contract."""

import base64
import http.client
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import numpy as np

from ..config import component_names
from .interfaces import PolicyError

ARMS = ("left", "right")
JOINT_NAMES = tuple("joint%d" % index for index in range(1, 7))
CAMERAS = {
    "observation.images.cam_h": "cam_high",
    "observation.images.cam_l": "cam_left_wrist",
    "observation.images.cam_r": "cam_right_wrist",
}
GRIPPER_RANGE_M = 0.1
MAX_BODY_BYTES = 16 * 1024 * 1024


def _number(value, label):
    if type(value) not in (int, float):
        raise PolicyError("%s must be a finite number" % label)
    try:
        number = float(value)
    except OverflowError as exc:
        raise PolicyError("%s must be a finite number" % label) from exc
    if not math.isfinite(number):
        raise PolicyError("%s must be a finite number" % label)
    return number


def parse_actions(reply):
    """Return absolute SDK targets in left-first order, radians and metres."""
    actions = reply.get("actions") if isinstance(reply, dict) else None
    if not isinstance(actions, list) or not 0 < len(actions) <= 4096:
        raise PolicyError("Expected a nonempty actions list of at most 4096 steps")
    values = np.empty((len(actions), 14), dtype=float)
    for row, action in enumerate(actions):
        if not isinstance(action, dict):
            raise PolicyError("Action %d must be an object" % row)
        for offset, side in zip((0, 7), ARMS):
            joints_key = side + "_arm_joint_state"
            joints = action.get(joints_key)
            if not isinstance(joints, list) or len(joints) != len(JOINT_NAMES):
                raise PolicyError("Action %d needs %s as six values in joint1 through joint6 order"
                                  % (row, joints_key))
            values[row, offset:offset + 6] = [
                _number(value, "%s.%s" % (joints_key, name))
                for name, value in zip(JOINT_NAMES, joints)
            ]
            gripper_key = side + "_ee_joint_state"
            gripper = action.get(gripper_key)
            if not isinstance(gripper, list) or len(gripper) != 1:
                raise PolicyError("Action %d needs a single value in %s" % (row, gripper_key))
            width = _number(gripper[0], gripper_key)
            if not 0.0 <= width <= 1.0:
                raise PolicyError("%s is outside [0, 1]" % gripper_key)
            values[row, offset + 6] = width
    values[:, 6] *= GRIPPER_RANGE_M
    values[:, 13] *= GRIPPER_RANGE_M
    return values


def _rotation(axis, angle):
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def _axis_quaternion(axis, angle):
    half = angle / 2
    return np.array([math.cos(half), *(math.sin(half) * axis)])


def _quaternion_product(left, right):
    lw, rw = left[0], right[0]
    lv, rv = left[1:], right[1:]
    return np.array([lw * rw - np.dot(lv, rv), *(lw * rv + rw * lv + np.cross(lv, rv))])


def _xml_vector(element, attribute, default):
    value = element.get(attribute, default) if element is not None else default
    result = np.asarray(value.split(), dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("URDF %s must contain three finite numbers" % attribute)
    return result


class PiperKinematics:
    """Load the API's base_link to link6 revolute chain from its URDF."""

    def __init__(self, path):
        try:
            root = ET.fromstring(Path(path).read_bytes())
            joints = {joint.find("child").attrib["link"]: joint for joint in root.findall("joint")}
            chain, link = [], "link6"
            for name in reversed(JOINT_NAMES):
                joint = joints[link]
                if joint.get("name") != name or joint.get("type") != "revolute":
                    raise ValueError("Expected the six Piper revolute joints in order")
                link = joint.find("parent").attrib["link"]
                origin = joint.find("origin")
                xyz = _xml_vector(origin, "xyz", "0 0 0")
                angles = _xml_vector(origin, "rpy", "0 0 0")
                axis = _xml_vector(joint.find("axis"), "xyz", "1 0 0")
                norm = np.linalg.norm(axis)
                if not np.isfinite(norm) or norm == 0.0:
                    raise ValueError("Invalid axis for %s" % name)
                limits = joint.find("limit").attrib
                lower, upper = float(limits["lower"]), float(limits["upper"])
                if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
                    raise ValueError("Invalid limits for %s" % name)
                rotation, quaternion = np.eye(3), np.array([1.0, 0.0, 0.0, 0.0])
                for origin_axis, angle in reversed(list(zip(np.eye(3), angles))):
                    rotation = rotation @ _rotation(origin_axis, angle)
                    quaternion = _quaternion_product(quaternion, _axis_quaternion(origin_axis, angle))
                chain.append((xyz, rotation, quaternion, axis / norm, (lower, upper)))
            if link != "base_link":
                raise ValueError("Expected base_link at the root of the Piper joint chain")
        except (OSError, ET.ParseError, ValueError, KeyError, AttributeError) as exc:
            raise PolicyError("Cannot load the RoboDojo API Piper URDF at %s: %s" % (path, exc)) from exc
        self.chain = chain[::-1]
        self.limits = [item[4] for item in self.chain]

    def pose(self, joints):
        position, rotation = np.zeros(3), np.eye(3)
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        for angle, (xyz, origin_rotation, origin_quaternion, axis, _) in zip(joints, self.chain):
            position += rotation @ xyz
            rotation = rotation @ origin_rotation @ _rotation(axis, angle)
            quaternion = _quaternion_product(quaternion, origin_quaternion)
            quaternion = _quaternion_product(quaternion, _axis_quaternion(axis, angle))
        quaternion /= np.linalg.norm(quaternion)
        if quaternion[0] < 0:
            quaternion *= -1
        return [*position.tolist(), *quaternion.tolist()]


def _step_limits(ik, key):
    values = ik.get(key)
    if not isinstance(values, list) or len(values) != len(JOINT_NAMES):
        raise PolicyError("RoboDojo /health requires six values for ik.%s" % key)
    limits = [_number(value, key) for value in values]
    if any(value <= 0 for value in limits):
        raise PolicyError("RoboDojo joint step limits must be positive")
    return limits


def validate_trajectory(actions, measured, capabilities, sdk_limits=None):
    """Check every row before enqueuing any action, using fresh follower feedback."""
    values = np.asarray(actions, dtype=float)
    measured = np.asarray(measured, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 14 or not len(values)
            or not np.isfinite(values).all()):
        raise PolicyError("Expected a finite left-first 14-dimensional RoboDojo trajectory")
    if measured.shape != (14,) or not np.isfinite(measured).all():
        raise PolicyError("Expected fresh finite 14-dimensional follower joint feedback")
    targets = values.reshape(-1, 2, 7)
    joints = targets[:, :, :6]
    limits = np.asarray(capabilities["joint_limits"], dtype=float)
    lower, upper = limits[:, 0], limits[:, 1]
    if sdk_limits is not None:
        lower = np.maximum(lower, sdk_limits[..., 0])
        upper = np.minimum(upper, sdk_limits[..., 1])
    if np.any((joints < lower) | (joints > upper)):
        raise PolicyError("RoboDojo action chunk exceeds Piper joint position limits")
    width = targets[:, :, 6]
    if np.any((width < 0.0) | (width > GRIPPER_RANGE_M)):
        raise PolicyError("RoboDojo action chunk exceeds Piper gripper width limits")
    if np.any(np.abs(joints[0] - measured.reshape(2, 7)[:, :6]) > capabilities["initial_step_rad"]):
        raise PolicyError("RoboDojo first action exceeds the joint step limit from measured feedback")
    if np.any(np.abs(np.diff(joints, axis=0)) > capabilities["step_rad"]):
        raise PolicyError("RoboDojo action chunk exceeds an adjacent joint step limit")


class LCGenHTTP:
    """Own the RoboDojo episode history inside the policy subprocess."""

    mode = "chunk"
    rlt = None

    def __init__(self, cfg):
        options = cfg["policy"]["lcgen_http"]
        self.endpoint = urlsplit(options["url"])
        self.connect_timeout = options["connect_timeout_sec"]
        self.request_timeout = options["request_timeout_sec"]
        self.urdf_path = options["urdf_path"]
        self.names = component_names(cfg["robot"]["arms"])
        self.invalidate()

    def _request(self, method, path, body=None):
        # HTTPConnection connects to the configured origin directly, ignoring proxy variables.
        connection = http.client.HTTPConnection(self.endpoint.hostname, self.endpoint.port, timeout=self.connect_timeout)
        started = time.monotonic()
        status, request_id = None, None
        try:
            connection.connect()
            connection.sock.settimeout(self.request_timeout)
            headers = {"Accept": "application/json"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body, headers)
            response = connection.getresponse()
            status = response.status
            request_id = response.getheader("X-Request-ID")
            data = response.read(MAX_BODY_BYTES + 1)
            if len(data) > MAX_BODY_BYTES:
                raise PolicyError("Response exceeds 16 MiB")
            result = json.loads(data)
            if not isinstance(result, dict):
                raise PolicyError("Response must be a JSON object")
            request_id = result.get("request_id") or request_id
            if status != 200:
                message = result.get("error") or result.get("detail") or response.reason
                raise PolicyError(str(message)[:500])
            return result, dict(http_status=status, request_id=request_id,
                                elapsed_sec=time.monotonic() - started)
        except (OSError, http.client.HTTPException, ValueError, PolicyError) as exc:
            raise PolicyError("RoboDojo %s %s status=%s elapsed=%.3fs request_id=%s: %s"
                              % (method, path, status, time.monotonic() - started,
                                 request_id, exc)) from exc
        finally:
            connection.close()

    def load(self):
        try:
            import cv2
        except ImportError as exc:
            raise PolicyError("Install opencv-python-headless in the RoboDojo policy interpreter") from exc
        self.cv2 = cv2
        health, _ = self._request("GET", "/health")
        metadata = health.get("metadata")
        required = {"robot": "piper", "execution_action_type": "joint",
                    "executed_raw_action_dim": 14, "runtime_fps": 30.0}
        if health.get("status") != "ready" or not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in required.items()
        ):
            raise PolicyError("RoboDojo /health must be ready with Piper 14D joint actions at 30 Hz")
        ik = metadata.get("ik")
        if not isinstance(ik, dict) or any(ik.get(key) != value for key, value in {
            "base_link": "base_link", "tip_link": "link6", "joint_order": list(JOINT_NAMES),
            "joint_units": "rad",
        }.items()):
            raise PolicyError("RoboDojo /health has an incompatible Piper IK contract")
        self.kinematics = PiperKinematics(self.urdf_path)
        self.capabilities = {
            "joint_limits": self.kinematics.limits,
            "initial_step_rad": _step_limits(ik, "max_initial_joint_step_rad"),
            "step_rad": _step_limits(ik, "max_joint_step_rad"),
        }
        return self

    def invalidate(self):
        self.history = []
        self.history_bytes = 0

    reset = invalidate

    def close(self):
        self.invalidate()

    def observe(self, observation):
        index = len(self.history)
        if observation["wam_frame_index"] != index:
            raise PolicyError("Expected RoboDojo episode observation %d" % index)
        images = {}
        for feature, name in CAMERAS.items():
            rgb = observation[feature]
            bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
            ok, jpeg = self.cv2.imencode(".jpg", bgr, [self.cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise PolicyError("Could not encode RoboDojo camera %s" % name)
            images[name] = base64.b64encode(jpeg).decode("ascii")
        entry = {"wam_frame_index": index, "env_idx": 0, "wam_fps": 30.0, "images": images}
        # Bound memory while accumulating and check the exact serialized size before POST.
        self.history_bytes += sum(len(img) for img in images.values())
        if self.history_bytes > MAX_BODY_BYTES:
            raise PolicyError("RoboDojo observation history exceeds the 16 MiB request limit")
        self.history.append(entry)
        return entry

    def infer(self, observation):
        latest = dict(self.observe(observation), instruction=observation["task"], state={})
        for side, part in zip(ARMS, observation["observation.state"].reshape(2, 7)):
            joints = part[:6].tolist()
            latest["state"][side + "_arm_joint_state"] = joints
            latest["state"][side + "_ee_pose"] = self.kinematics.pose(joints)
            latest["state"][side + "_ee_joint_state"] = [float(part[6]) / GRIPPER_RANGE_M]
        body = json.dumps({"observations": [*self.history[:-1], latest]},
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(body) > MAX_BODY_BYTES:
            raise PolicyError("RoboDojo observation history exceeds the 16 MiB request limit")
        result, metadata = self._request("POST", "/predict", body)
        try:
            actions = parse_actions(result)
        except PolicyError as exc:
            raise PolicyError("RoboDojo POST /predict status=%s elapsed=%.3fs request_id=%s: %s"
                              % (metadata["http_status"], metadata["elapsed_sec"],
                                 metadata["request_id"], exc)) from exc
        return {"actions": actions, "metadata": metadata}
