#!/usr/bin/env python3
"""Record the ROS topics published during an interactive RoboDojo replay."""

import argparse
from pathlib import Path
import sys
import time

from _bootstrap import ROOT
from piper_control.config import load_config
from piper_control.recording.manual_bag import record_bag
from piper_control.recording.topics import print_topic_plan, topic_plan
from test_robodojo_piper_replay import main as replay_main


def wait_for_writer(process, bag):
    active = bag.with_suffix(".bag.active")
    deadline = time.monotonic() + 15.0
    while True:
        if process.poll() is not None:
            raise RuntimeError("rosbag exited at startup; inspect rosbag.log")
        if active.is_file():
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("rosbag did not open its output; check the ROS master and rosbag.log")
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Replay options include --episode-index, --condition-for-test and --arrival-report.",
    )
    parser.add_argument("--output", help="Directory for data.bag, manifest.json and rosbag.log")
    parser.add_argument("--topics-config", type=Path, help="Override the recording topic YAML")
    parser.add_argument("--list-topics", action="store_true", help="Print the bag topic selection")
    parser.add_argument("--execute", action="store_true", help="Enable live replay and record it")
    args, replay_args = parser.parse_known_args()
    if not args.list_topics and args.execute and (not args.output or not sys.stdin.isatty()):
        parser.error("--execute requires --output and an interactive terminal")

    try:
        config = load_config(ROOT / "configs/robodojo_piper_api.yaml")
        if args.topics_config:
            config["record"]["topics_config"] = str(args.topics_config.expanduser().resolve())
        if args.list_topics:
            print_topic_plan(topic_plan(config))
            return 0
        if not args.execute:
            return replay_main(replay_args, config=config)

        # The validated deployment remains unchanged. This replay entry point
        # enables the existing Session publisher for the duration of the bag.
        config["record"]["backend"] = "ros_publish"
        replay_status = 130

        def run_replay(process, bag):
            nonlocal replay_status
            wait_for_writer(process, bag)
            replay_status = replay_main([*replay_args, "--execute"], config=config)

        result = record_bag(config, args.output, during_recording=run_replay)
        if result["status"] != "complete":
            print("Recording %s: %s" % (result["status"], result["reason"]), file=sys.stderr)
            return 130 if replay_status == 130 and result["reason"] == "KeyboardInterrupt" else 1
        print("Saved: " + result["output_file"])
        return replay_status
    except (OSError, ValueError, RuntimeError) as exc:
        print("Recording failed: " + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
