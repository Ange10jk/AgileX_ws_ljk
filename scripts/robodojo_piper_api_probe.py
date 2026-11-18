#!/usr/bin/env python3
"""Request one RoboDojo prediction from real observations without motor commands."""

import argparse
from contextlib import ExitStack
import json
import time

import _bootstrap  # Add the checkout to the import path.
from piper_control.config import ConfigError, load_config
from piper_control.hardware.robot import RobotError
from piper_control.policies.client import PolicyClient
from piper_control.policies.interfaces import PolicyError
from piper_control.policies.lcgen_http import validate_trajectory
from piper_control.workflows.session import Session


def capture(session):
    deadline = time.monotonic() + 3
    while True:
        session.cameras.check_health()
        try:
            snapshot = session.robot.snapshot()
            observation = session._lcgen_observation(snapshot, session.cameras.snapshot())
            return snapshot, observation
        except (RobotError, PolicyError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def probe(config):
    if config["policy"].get("backend") != "lcgen_http":
        raise ConfigError("The RoboDojo probe requires policy.backend=lcgen_http")
    session = Session(config)
    robot, cameras = session.robot, session.cameras
    policy = PolicyClient(config)

    # Manage passive resources directly. Session.start() enables the arms.
    with ExitStack() as cleanup:
        cleanup.callback(policy.close)
        policy.load()
        sdk_limits = robot.joint_limits()
        cleanup.callback(cameras.close)
        cameras.start()
        cleanup.callback(robot.close)
        robot.connect()

        before, observation = capture(session)
        observation["wam_frame_index"] = 0
        policy.submit(observation, generation=0, seq=1)
        deadline = time.monotonic() + config["policy"]["inference_timeout_sec"] + 1
        while True:
            reply = policy.poll()
            if reply is not None and reply.get("kind") == "infer":
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("No prediction returned before the worker deadline")
            time.sleep(0.01)

        actions = reply["actions"]
        print(json.dumps({
            "instruction": observation["task"],
            "camera_timestamps": observation["camera_timestamps"],
            "follower_timestamps": observation["follower_timestamps"],
            "measured_state": before["state"].tolist(),
            "http": reply["metadata"],
            "action_shape": list(actions.shape),
            "actions": actions.tolist(),
        }, ensure_ascii=False, indent=2), flush=True)
        latest, _ = capture(session)
        validate_trajectory(actions, latest["state"], policy.capabilities, sdk_limits)
        print("整段通过本地轨迹校验。未下发目标。", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="RoboDojo HTTP deployment YAML")
    args = parser.parse_args(argv)
    probe(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
