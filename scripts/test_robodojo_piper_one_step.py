#!/usr/bin/env python3
"""Preflight the shortest RoboDojo episode's first target, or submit it once."""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np

from _bootstrap import ROOT
from piper_control.config import component_names, load_config
from piper_control.control.arbiter import ControlError, Controller
from piper_control.hardware.robot import RobotError, SDKRobot
from piper_control.policies.interfaces import PolicyError
from piper_control.policies.lcgen_http import GRIPPER_RANGE_M, parse_actions, validate_trajectory
from test_robodojo_piper_replay import condition_test_actions, load_episode, load_limits

MOVE_SPEED_PERCENT = 10
OBSERVE_SECONDS = 2.0


class CheckedFollowerRobot(SDKRobot):
    """The production SDK transport, with fresh feedback required for this test."""

    def __init__(self, config, max_age):
        super().__init__(config)
        self.max_age = max_age
        self.before_hold = None

    def snapshot(self):
        sample = super().snapshot()
        now = time.time()
        stamps = []
        for side in self.arms:
            follower = sample["arms"][side]["follower"]
            times = [follower.get(key) for key in ("joint_timestamp", "gripper_timestamp")]
            if not follower["valid"] or follower["source"] != "feedback" or not all(
                stamp is not None and math.isfinite(stamp) and abs(now - stamp) <= self.max_age
                for stamp in times
            ):
                raise RobotError("Follower joint/gripper feedback is missing or stale: " + side)
            stamps.extend(times)
        if max(stamps) - min(stamps) > self.max_age:
            raise RobotError("Left/right follower feedback exceeds the allowed time skew")
        if sample["state"].shape != (14,) or not np.isfinite(sample["state"]).all():
            raise RobotError("Expected finite follower feedback in canonical 14D order")
        if self.before_hold is not None:
            self.before_hold(sample["state"])
        return sample

    def check_enabled(self):
        for (side, _), device in self.devices.items():
            sdk = device["sdk"]
            enabled = sdk.get_joints_enable_status_list()
            if enabled is None or len(enabled) != 6 or not all(enabled):
                raise RobotError("Follower motor enable lost: " + side)
            faults = self._driver_faults(sdk)
            if faults:
                raise RobotError("Follower driver fault on %s: %s" % (side, faults))


def initial_feedback(robot):
    deadline = time.monotonic() + 3.0
    while True:
        try:
            return robot.snapshot()
        except RobotError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def api_reply(chunk):
    # Dataset grippers are already in metres. Normalize only to construct a
    # test response, so the real parser restores the original physical values.
    actions = []
    for row in chunk:
        action = {}
        for side, part in zip(("left", "right"), row.reshape(2, 7)):
            action[side + "_arm_joint_state"] = part[:6].tolist()
            action[side + "_ee_joint_state"] = [float(part[6]) / GRIPPER_RANGE_M]
        actions.append(action)
    return {"actions": actions}


def report(stage, names, recorded, current, target):
    print(json.dumps({
        "stage": stage,
        "action_names": names,
        "recorded_start": recorded.tolist(),
        "current_follower": current.tolist(),
        "first_target": target[0].tolist(),
        "current_minus_recorded_start": (current - recorded).tolist(),
        "target_minus_current": (target[0] - current).tolist(),
    }, indent=2), flush=True)


@contextmanager
def robot_session(controller):
    closing = False

    def interrupt(signum, frame):
        # Unwind SDK I/O before issuing the electronic stop in close().
        controller.abort.set()
        if not closing:
            raise KeyboardInterrupt("One-step test interrupted")

    handlers = {signum: signal.signal(signum, interrupt) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        closing = True
        try:
            # SDKRobot.close stops only after the SDK was touched.
            controller.robot.close()
        finally:
            for signum, previous in handlers.items():
                signal.signal(signum, previous)


def run_one_step(config, robot, recorded, target, capabilities, sdk_limits, execute, confirm):
    names = component_names(config["robot"]["arms"])
    controller = Controller(config, robot)

    def valid_hold(state):
        validate_trajectory(state[None, :], state, capabilities, sdk_limits)

    def valid_start(state):
        try:
            delta = np.abs(state - recorded).reshape(2, 7)[:, :6]
            exceeded = delta > capabilities["initial_step_rad"]
            if exceeded.any():
                side, joint = np.argwhere(exceeded)[0]
                raise PolicyError(
                    "Current pose differs from recorded start: %s joint%d %.6f rad exceeds %.6f rad"
                    % (robot.arms[side], joint + 1, delta[side, joint], capabilities["initial_step_rad"][joint])
                )
            valid_hold(state)
            validate_trajectory(target, state, capabilities, sdk_limits)
        except PolicyError:
            report("rejected_feedback", names, recorded, state, target)
            raise

    with robot_session(controller):
        robot.connect()
        start = initial_feedback(robot)["state"]
        report("preflight", names, recorded, start, target)
        valid_start(start)
        print("Preflight passed. Joint angles are rad; gripper widths are m.", flush=True)
        if not execute:
            return
        if not confirm():
            print("Cancelled before enabling the followers.", flush=True)
            return

        valid_start(robot.snapshot()["state"])
        robot.enable()
        robot.check_enabled()
        # The transition holds a freshly read pose. Guard that exact sample.
        robot.before_hold = valid_start
        controller.enter_policy()
        robot.before_hold = None

        robot.check_enabled()
        current = robot.snapshot()
        valid_start(current["state"])
        if not controller.expect(1) or not controller.offer({
            "generation": controller.generation, "seq": 1, "actions": target,
        }):
            raise ControlError("Controller refused the one-step target")
        step = controller.step(current)
        if step is None or step["predicted_action"] is None or controller.actions:
            raise ControlError("Expected exactly one recorded target")
        report("recorded_target_submitted", names, recorded, current["state"], target)

        deadline = time.monotonic() + OBSERVE_SECONDS
        while True:
            robot.snapshot()
            robot.check_enabled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.02, remaining))
        # Hold a fresh measured pose, with no further recorded target.
        robot.before_hold = valid_hold
        controller.pause()
        print(json.dumps({
            "stage": "HOLD",
            "recorded_targets_submitted": 1,
            "measured_hold_target": controller.target.tolist(),
            "remaining_error": (target[0] - controller.target).tolist(),
        }, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path.home() / "luojk/lerobot_datasets/stand_up_bottles")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/robodojo_piper_api.yaml")
    parser.add_argument("--execute", action="store_true", help="Allow one real step after interactive confirmation")
    args = parser.parse_args(argv)
    if args.execute and not sys.stdin.isatty():
        parser.error("--execute requires an interactive terminal")
    try:
        config = load_config(args.config, {"robot": {"move_speed_percent": MOVE_SPEED_PERCENT}})
        if config["policy"]["backend"] != "lcgen_http":
            parser.error("Use the validated RoboDojo LCGEN HTTP deployment config")
        names = component_names(config["robot"]["arms"])
        index, actions, states = load_episode(
            args.dataset.expanduser(), names, config["workflow"]["policy_hz"],
        )
        capabilities, sdk_limits = load_limits(config)
        conditioned = condition_test_actions(actions, states, capabilities, sdk_limits)
        target = parse_actions(api_reply(conditioned[:1]))
        print(json.dumps({
            "episode_index": index, "frame": 0, "execute": args.execute,
            "joint_speed_percent": MOVE_SPEED_PERCENT,
            "first_target_changed_from_dataset": bool(np.any(target[0] != actions[0])),
            "first_step_limit_rad": capabilities["initial_step_rad"],
        }), flush=True)
        robot = CheckedFollowerRobot(config["robot"], config["policy"]["lcgen_http"]["max_observation_skew_sec"])

        def confirm():
            return input("Confirm clear workspace and the displayed target. Type MOVE ONE STEP to continue: ").strip() == "MOVE ONE STEP"

        run_one_step(config, robot, states[0], target, capabilities, sdk_limits, args.execute, confirm)
        return 0
    except KeyboardInterrupt:
        print("One-step test interrupted.", file=sys.stderr)
        return 130
    except (ValueError, OSError, EOFError, ControlError, PolicyError, RobotError) as exc:
        print("One-step test stopped: %s" % exc, file=sys.stderr)
        print("Inspect live feedback and the recorded start before retrying; no automatic alignment is performed.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
