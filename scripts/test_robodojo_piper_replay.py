#!/usr/bin/env python3
"""Replay a recorded RoboDojo episode through the live follower Session."""

import argparse
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

from _bootstrap import ROOT
from _piper_replay_arrival import ReplayArrival
from piper_control.config import component_names, load_config
from piper_control.control.arbiter import ControlError, Controller
from piper_control.hardware.robot import RobotError, SDKRobot
from piper_control.policies.interfaces import PolicyError
from piper_control.policies.lcgen_http import GRIPPER_RANGE_M, PiperKinematics, validate_trajectory
from piper_control.workflows import session as session_module
from piper_control.workflows.session import Session

CONFIG = ROOT / "configs/robodojo_piper_api.yaml"
DATASET = Path.home() / "luojk/lerobot_datasets/stand_up_bottles"

# Temporary replay trial only. Set this back to True to restore the production
# joint/gripper position and initial/adjacent step admission checks for replay.
CHECK_REPLAY_LIMITS = False
# Piper arm-status feedback distinguishes e-stop from the driver's enable bits.
PIPER_EMERGENCY_STOP = 0x01
SIDES = ("left", "right")
CHUNK_SIZE = 8
# Documented 30 Hz service limits, used as offline fixtures.
INITIAL_STEP_RAD = [0.13, 0.18, 0.16, 0.24, 0.23, 0.25]
STEP_RAD = [0.07, 0.10, 0.08, 0.12, 0.12, 0.13]


def load_episode(root, names, fps, *, episode_index=None):
    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != fps:
        raise ValueError("Dataset FPS must match the tested RoboDojo policy rate")
    for field in ("action", "observation.state"):
        feature = info["features"][field]
        if feature["names"] != names or feature["shape"] != [14]:
            raise ValueError("Unexpected dataset component order or shape for " + field)
    episodes = pq.read_table(root / "meta/episodes",
                             columns=["episode_index", "length"]).to_pylist()
    if not episodes:
        raise ValueError("Dataset has no episodes")
    if episode_index is None:
        selected = min(episodes, key=lambda row: (row["length"], row["episode_index"]))
    else:
        by_index = {row["episode_index"]: row for row in episodes}
        if episode_index not in by_index:
            available = ", ".join(str(index) for index in sorted(by_index))
            raise ValueError(
                f"Episode index {episode_index} not found. Available indices: {available}"
            )
        selected = by_index[episode_index]
    index = selected["episode_index"]
    rows = pq.read_table(root / "data", columns=["frame_index", "action", "observation.state"],
                         filters=[("episode_index", "=", index)]).to_pylist()
    rows.sort(key=lambda row: row["frame_index"])
    if selected["length"] <= 0 or [row["frame_index"] for row in rows] != list(range(selected["length"])):
        raise ValueError("Selected episode is missing contiguous frames")
    values = [np.asarray([row[field] for row in rows], dtype=float)
              for field in ("action", "observation.state")]
    if any(array.shape != (len(rows), 14) or not np.isfinite(array).all() for array in values):
        raise ValueError("Expected finite 14-dimensional recorded actions and states")
    return index, *values


def load_limits(config):
    capabilities = {
        "joint_limits": PiperKinematics(config["policy"]["lcgen_http"]["urdf_path"]).limits,
        "initial_step_rad": INITIAL_STEP_RAD,
        "step_rad": STEP_RAD,
    }
    sdk_limits = SDKRobot(config["robot"]).joint_limits()  # No CAN connection.
    return capabilities, sdk_limits


def condition_test_actions(actions, measured, capabilities, sdk_limits):
    """Project a test copy onto the local numeric bounds for offline replay."""
    adjusted = actions.copy().reshape(-1, 2, 7)
    joints = adjusted[:, :, :6]
    limits = np.asarray(capabilities["joint_limits"])
    lower = np.broadcast_to(np.maximum(limits[:, 0], sdk_limits[:, 0]), joints.shape).copy()
    upper = np.broadcast_to(np.minimum(limits[:, 1], sdk_limits[:, 1]), joints.shape).copy()

    # Leave a roundoff margin because the production guard uses a strict bound.
    initial = np.asarray(capabilities["initial_step_rad"]) - 1e-12
    adjacent = np.asarray(capabilities["step_rad"]) - 1e-12
    chunk_reference = measured[::CHUNK_SIZE].reshape(-1, 2, 7)[:, :, :6]
    lower[::CHUNK_SIZE] = np.maximum(lower[::CHUNK_SIZE], chunk_reference - initial)
    upper[::CHUNK_SIZE] = np.minimum(upper[::CHUNK_SIZE], chunk_reference + initial)

    # Keep every frame reachable from its predecessor, including chunk seams.
    # The backward pass prevents an early adjustment from blocking a later
    # frame with a narrower interval from recorded follower feedback.
    for index in range(len(joints) - 2, -1, -1):
        lower[index] = np.maximum(lower[index], lower[index + 1] - adjacent)
        upper[index] = np.minimum(upper[index], upper[index + 1] + adjacent)
    if np.any(lower > upper):
        raise ValueError("Recorded feedback and joint constraints have no feasible test replay")

    joints[0] = np.clip(joints[0], lower[0], upper[0])
    for index in range(1, len(joints)):
        low = np.maximum(lower[index], joints[index - 1] - adjacent)
        high = np.minimum(upper[index], joints[index - 1] + adjacent)
        joints[index] = np.clip(joints[index], low, high)
    widths = adjusted[:, :, 6]
    np.clip(widths, 0.0, GRIPPER_RANGE_M, out=widths)
    return adjusted.reshape(-1, 14)


def fresh_feedback(snapshot, max_age):
    state = np.asarray(snapshot["state"], dtype=float)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise RobotError("Expected finite 14D follower feedback")
    sampled = time.time()
    monotonic = time.monotonic()
    timestamps = []
    for side in SIDES:
        sample = snapshot["arms"][side]["follower"]
        stamps = [sample.get(key) for key in ("joint_timestamp", "gripper_timestamp")]
        if (not sample["valid"] or sample["source"] != "feedback"
                or any(stamp is None or not math.isfinite(stamp)
                       or abs(sampled - stamp) > max_age for stamp in stamps)):
            raise RobotError("Stale or unavailable follower feedback: " + side)
        timestamps.extend(stamps)
    if max(timestamps) - min(timestamps) > max_age:
        raise RobotError("Dual follower feedback skew exceeds %.3f s" % max_age)
    return state, {
        "sampled_wall_sec": sampled,
        "sampled_mono_sec": monotonic,
        "sdk_wall_sec": timestamps,
        "actual": state.tolist(),
    }


class LocalChunks:
    """Supply one ready synthetic reply per real Session request."""

    def __init__(self, caps, *, actions=None, request_prefix="local-frame"):
        self.capabilities = caps
        self.actions = actions
        self.request_prefix = request_prefix
        self.reply = None

    def load(self):
        return self

    def submit(self, observation, generation, seq):
        frame = observation["wam_frame_index"]
        if self.actions is None or not 0 <= frame < len(self.actions) or frame % CHUNK_SIZE:
            raise PolicyError("Unexpected local chunk request for frame %s" % frame)
        self.reply = {
            "kind": "infer",
            "generation": generation,
            "seq": seq,
            "actions": self.actions[frame:frame + CHUNK_SIZE].copy(),
            "metadata": {"request_id": "%s-%d" % (self.request_prefix, frame), "source": "BASE"},
        }
        self.reply["metadata"]["local_reply_ready_mono_sec"] = time.monotonic()

    def observe(self, observation, generation):
        pass

    def poll(self):
        reply, self.reply = self.reply, None
        return reply

    def invalidate(self, generation):
        self.reply = None

    def close(self):
        self.reply = None


class CheckedDispatch:
    """Turn an explicit SDK refusal into a Session fault."""

    def __init__(self, device, method, channel, arrival=None):
        self.device, self.method, self.channel = device, method, channel
        self.arrival = arrival

    def __getattr__(self, name):
        method = getattr(self.device, name)
        if name != self.method:
            return method

        def check(*args, **kwargs):
            call = None
            if self.arrival is not None:
                kind, value = ("joint", list(args[0])) if name == "move_js" else (
                    "gripper", kwargs["value"])
                call = self.arrival.begin_call(self.channel, kind, value)
            result = method(*args, **kwargs)
            if result is False:
                raise RobotError("Piper SDK refused %s on %s" % (name, self.channel))
            if call is not None:
                call["return_mono_sec"] = time.monotonic()
            return result

        return check


def validate_replay_chunk(actions, measured, capabilities, sdk_limits):
    """Keep the replay action shape and finiteness checks for every chunk."""
    try:
        values = np.asarray(actions, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise PolicyError("Expected numeric RoboDojo replay actions") from exc
    if (values.ndim != 2 or values.shape[1] != 14
            or not 0 < len(values) <= CHUNK_SIZE or not np.isfinite(values).all()):
        raise PolicyError("Expected a finite RoboDojo chunk with 1–8 14D actions")
    if CHECK_REPLAY_LIMITS:
        validate_trajectory(values, measured, capabilities, sdk_limits)


class ReplayRobot(SDKRobot):
    def __init__(self, config, *, arrival=None, **kwargs):
        self.arrival = arrival
        super().__init__(config["robot"], **kwargs)
        self.max_age = config["policy"]["lcgen_http"]["max_observation_skew_sec"]

    def _make_sdk(self, spec):
        sdk, gripper = super()._make_sdk(spec)
        return (CheckedDispatch(sdk, "move_js", spec["channel"], self.arrival),
                CheckedDispatch(gripper, "move_gripper_m", spec["channel"], self.arrival))

    def connect(self):
        super().connect()
        if self.arrival is not None:
            self.arrival.channels = {
                side: self.devices[(side, "follower")]["sdk"].channel for side in SIDES
            }

    def stop(self):
        if self.arrival is not None:
            self.arrival.stop()
        return super().stop()

    request_stop = stop

    def _arm_status(self):
        return {
            side: self._detail(self.devices[(side, "follower")]["sdk"].get_arm_status())
            for side in self.arms
        }

    def _arm_status_issues(self, statuses, newer_than=None):
        now = time.time()
        pending, stopped = [], []
        for side, status in statuses.items():
            stamp, code = status.get("stamp"), status.get("arm_status")
            name = side + "/follower"
            if not status.get("valid") or stamp is None or stamp <= 0 or not isinstance(code, int):
                pending.append(name + " missing arm_status feedback")
            elif (not math.isfinite(stamp) or abs(now - stamp) > self.max_age
                  or (newer_than is not None and stamp <= newer_than)):
                pending.append(name + " stale arm_status feedback")
            elif code == PIPER_EMERGENCY_STOP:
                stopped.append(name)
        if stopped:
            raise RobotError("EMERGENCY_STOP in arm_status for %s. Inspect the robot and use the "
                             "explicit recovery procedure before retrying." % ", ".join(stopped))
        return pending

    def _await_arm_status(self, stage, newer_than=None):
        deadline = time.monotonic() + self.config["startup_enable_timeout_sec"]
        while self.mode != "STOPPED":
            pending = self._arm_status_issues(self._arm_status(), newer_than)
            if not pending:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RobotError("Cannot replay %s: %s" % (stage, ", ".join(pending)))
            time.sleep(min(0.01, remaining))
        raise RobotError("Stop requested while waiting for arm_status " + stage)

    def enable(self):
        if not self.connected:
            raise RobotError("Connect before enabling")
        self._await_arm_status("before enable")
        enable_started = time.time()
        super().enable()
        self._await_arm_status("before the startup hold", newer_than=enable_started)

    def snapshot(self):
        snapshot = super().snapshot()
        statuses = {side: snapshot["arms"][side]["follower"]["arm_status"] for side in self.arms}
        pending = self._arm_status_issues(statuses)
        if pending:
            raise RobotError(", ".join(pending))
        _, evidence = fresh_feedback(snapshot, self.max_age)
        if self.arrival is not None:
            self.arrival.observe(evidence)
        return snapshot

    def command(self, action):
        # Status can change after the control-loop snapshot, including during
        # reply admission. Read it again immediately before any SDK target.
        pending = self._arm_status_issues(self._arm_status())
        if pending:
            raise RobotError(", ".join(pending))
        return super().command(action)


class ReplaySession(Session):
    def __init__(self, config, robot, caps, episode, actions, **kwargs):
        self.dataset_episode = episode
        self.rejection = None
        policy = LocalChunks(caps, actions=actions, request_prefix="replay-episode-%d-frame" % episode)
        super().__init__(config, robot=robot, policy=policy, **kwargs)

    def _lcgen_tick(self, snapshot, frames):
        if self._lcgen_reply is None:
            return super()._lcgen_tick(snapshot, frames)
        # Scope the temporary replay validator to this reply on this control
        # thread. Keep production feedback checks, queue admission and dispatch.
        with patch.object(session_module, "validate_trajectory", validate_replay_chunk):
            return super()._lcgen_tick(snapshot, frames)

    def _dispatch(self, snapshot, frames, observation):
        arrival = self.robot.arrival
        if arrival is None or self.controller.state != Controller.POLICY or not self.controller.actions:
            return super()._dispatch(snapshot, frames, observation)
        arrival.frame = self.step_id
        try:
            return super()._dispatch(snapshot, frames, observation)
        finally:
            arrival.frame = None

    def _lcgen_fail(self, exc, snapshot, frames):
        self.rejection = "episode=%d frame=%d: %s" % (self.dataset_episode, self.step_id, exc)
        print(self.rejection, file=sys.stderr, flush=True)
        super()._lcgen_fail(exc, snapshot, frames)
        self.quit = True

    def _input(self):
        try:
            super()._input()
        finally:
            self.commands.put("q")

    def command(self, command, snapshot, frames):
        super().command(command, snapshot, frames)
        if command.strip().lower() in ("hold", "pause", "s", "success", "t", "truncate", "f", "failure"):
            self.quit = True


def positive_tolerance(value):
    try:
        tolerance = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive finite tolerance") from None
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise argparse.ArgumentTypeError("expected a positive finite tolerance")
    return tolerance


def main(argv=None, *, config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=int,
                        help="Zero-based LeRobot episode index (default: shortest episode)")
    parser.add_argument("--condition-for-test", action="store_true",
                        help="Condition an in-memory test copy using the documented joint/gripper limits")
    parser.add_argument("--execute", action="store_true",
                        help="Enable and hold the current pose, then wait for interactive p to start")
    parser.add_argument("--arrival-report", type=Path,
                        help="Write a new JSON report of measured target arrival (requires --execute)")
    parser.add_argument("--arrival-joint-tol-rad", type=positive_tolerance, default=0.002,
                        help="Per-joint absolute arrival tolerance in radians (default: 0.002)")
    parser.add_argument("--arrival-gripper-tol-m", type=positive_tolerance, default=0.001,
                        help="Absolute gripper-width arrival tolerance in meters (default: 0.001)")
    args = parser.parse_args(argv)
    if args.arrival_report is not None and not args.execute:
        parser.error("--arrival-report requires --execute")
    if args.execute and not sys.stdin.isatty():
        parser.error("--execute requires an interactive terminal")
    try:
        if config is None:
            config = load_config(CONFIG)
        if config["workflow"]["auto_start"]:
            raise ValueError("Replay requires workflow.auto_start: false")
        episode, actions, measured = load_episode(
            DATASET, component_names(config["robot"]["arms"]), config["workflow"]["policy_hz"],
            episode_index=args.episode_index,
        )
        caps, sdk_limits = load_limits(config)
        print("episode=%d frames=%d chunk_size=%d policy_hz=%d"
              % (episode, len(actions), CHUNK_SIZE, config["workflow"]["policy_hz"]), flush=True)
        if args.condition_for_test:
            conditioned = condition_test_actions(actions, measured, caps, sdk_limits)
            changes = np.abs(conditioned - actions).reshape(-1, 2, 7)
            joints, widths = changes[:, :, :6], changes[:, :, 6]
            print(json.dumps({
                "test_conditioning": {
                    "joint_values_changed": int(np.count_nonzero(joints)),
                    "joint_frames_changed": int(np.count_nonzero(joints.any(axis=(1, 2)))),
                    "max_joint_adjustment_rad": float(joints.max()),
                    "gripper_values_changed": int(np.count_nonzero(widths)),
                    "gripper_frames_changed": int(np.count_nonzero(widths.any(axis=1))),
                    "max_gripper_adjustment_m": float(widths.max()),
                }
            }), flush=True)
            actions = conditioned

        print(json.dumps({"first_target": actions[0].tolist(), "conditioned": args.condition_for_test,
                          "replay_physical_admission_checks": CHECK_REPLAY_LIMITS}), flush=True)
        if not args.execute:
            print("Read-only replay preview complete.")
            return 0

        if not CHECK_REPLAY_LIMITS:
            print("WARNING: Replay joint, gripper and step admission checks are temporarily disabled.",
                  file=sys.stderr, flush=True)
        print("WARNING: Startup holds the measured pose. There is no positioning to frame zero. "
              "After p, the first recorded target can cause abrupt movement. Clear both arms' workspace. "
              "The SDK's configured joint clamps still apply.",
              file=sys.stderr, flush=True)
        arrival = None
        if args.arrival_report is not None:
            arrival = ReplayArrival(config["workflow"]["policy_hz"], CHUNK_SIZE,
                                    args.arrival_joint_tol_rad, args.arrival_gripper_tol_m)
        session = ReplaySession(config, ReplayRobot(config, arrival=arrival), caps, episode, actions)
        run_options = dict(max_steps=len(actions), max_seconds=config["workflow"]["max_episode_sec"])
        if arrival is None:
            result = session.run(**run_options)
        else:
            # Check output accessibility before enabling. Preserve partial evidence on failure.
            with args.arrival_report.open("x") as output:
                try:
                    result = session.run(**run_options)
                finally:
                    report = arrival.report()
                    json.dump(dict(report, dataset_episode_index=episode, expected_frames=len(actions)),
                              output, indent=2, allow_nan=False)
                    output.write("\n")
            print(json.dumps({
                "arrival_report": str(args.arrival_report),
                "attempted_frames": report["attempted_frames"],
                "nonfinal_frames": report["nonfinal_frames"],
                "interchunk_frames_reached_after_nominal_period":
                    report["interchunk_frames_reached_after_nominal_period"],
                "final_attempted_frame_status": (report["final_attempted_frame"] or {}).get("status"),
            }), flush=True)
        status = ("rejected" if session.rejection else
                  "complete" if result["steps"] == len(actions) else "stopped")
        print(json.dumps(dict(result, status=status, dataset_episode_index=episode,
                              failure=session.rejection)))
        return 0 if status == "complete" else 1
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print("Replay refused or stopped: %s" % exc, file=sys.stderr, flush=True)
        return 130 if isinstance(exc, ControlError) and str(exc) == "Stop requested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
