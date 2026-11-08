"""Observation-only ROS process. This module never imports a robot or policy."""
import os
import select
import signal
import sys
import traceback

from piper_control.ipc import read_message, write_message


class _UnsafeROSShutdown(RuntimeError):
    """A camera thread may still hold a synchronous ROS publisher lock."""


def _close_resources(bridge, cameras):
    if cameras is not None:
        from ..hardware.cameras import CameraPublishTimeout

        try:
            cameras.close(publish_timeout=5.0)
        except CameraPublishTimeout as exc:
            raise _UnsafeROSShutdown(str(exc)) from exc
        except (KeyboardInterrupt, SystemExit) as exc:
            raise _UnsafeROSShutdown("Camera recording shutdown interrupted") from exc
        except Exception:
            # CameraRig propagates capture errors after joining the publisher.
            bridge.close()
            raise
    bridge.close()


def serve(incoming, outgoing, bridge_factory=None, camera_factory=None, poll_input=None):
    # These imports are confined to the child.
    if bridge_factory is None:
        from .ros_bridge import ROSBridge
        bridge_factory = ROSBridge
    bridge = cameras = None
    expected = 0
    failed = False
    report = False

    def close():
        nonlocal bridge
        if bridge is not None:
            try:
                _close_resources(bridge, cameras)
            finally:
                bridge = None

    def check_health():
        if cameras is not None:
            cameras.check_health()
        bridge.check_health()

    try:
        while True:
            sequence = expected
            try:
                if cameras is not None and poll_input is not None:
                    while not poll_input(0.1):
                        check_health()
                        if report:
                            cameras.report_recording()
                message = read_message(incoming)
                kind, sequence = message.get("kind"), message.get("seq")
                if sequence != expected:
                    raise ValueError("Collection sequence mismatch")
                expected += 1
                if kind == "start" and bridge is None:
                    config = message["config"]
                    options = config.get("record", {})
                    publish_only = options.get("backend", "ros") == "ros_publish"
                    bridge = bridge_factory(config)
                    bridge.start()
                    if not publish_only:
                        bridge.start_bag(message["directory"])
                    reply = {"kind": "ready", "seq": sequence}
                    if options.get("camera_in_ros_worker", False):
                        if not publish_only:
                            raise ValueError("Worker cameras require record.backend=ros_publish")
                        if camera_factory is None:
                            from ..hardware.cameras import CameraRig
                            camera_factory = CameraRig
                        cameras = camera_factory(config["observation"])
                        report = options.get("queue_diagnostics", False)
                        cameras.record_to(
                            bridge.publish_images,
                            capacity=options.get("camera_queue_size", 32),
                            diagnostics=report,
                            queue_policy=options.get("camera_queue_policy", "strict"),
                        )
                        cameras.start(wait_for_publish=True)
                        check_health()
                elif bridge is None or kind == "start":
                    raise ValueError("Collection worker startup protocol violation")
                elif kind == "close":
                    close()
                    write_message(outgoing, {"kind": "closed", "seq": sequence})
                    return
                else:
                    payload = message["payload"]
                    if kind == "state":
                        bridge.publish_state(payload)
                    elif kind == "images":
                        if cameras is not None:
                            raise ValueError("Image forwarding is disabled when the ROS worker owns cameras")
                        bridge.publish_images(payload)
                    elif kind == "action":
                        bridge.publish_action(payload["action"], payload["source"], payload.get("metadata"))
                    elif kind == "event":
                        if cameras is not None and payload.get("type") == "episode_end":
                            final = payload.get("final_observation")
                            if final is not None and not final.get("camera_stamps"):
                                final["camera_stamps"] = cameras.rgb_stamps()
                        bridge.publish_event(payload)
                    else:
                        raise ValueError("Unsupported collection input: " + str(kind))
                    check_health()
                    reply = {"kind": "ack", "seq": sequence}
                write_message(outgoing, reply)
                if report:
                    cameras.report_recording()
            except EOFError:
                return
            except Exception as exc:
                failed = True
                error = "{}: {}".format(type(exc).__name__, exc)
                try:
                    write_message(outgoing, {"kind": "error", "seq": sequence, "error": error})
                finally:
                    if isinstance(exc, _UnsafeROSShutdown):
                        raise exc
                if report or cameras is None:
                    traceback.print_exc(file=sys.stderr)
                return
    finally:
        try:
            close()
        except _UnsafeROSShutdown:
            raise
        except Exception:
            if not failed:
                traceback.print_exc(file=sys.stderr)
        finally:
            if cameras is not None:
                cameras.report_recording(final=True)


def main():
    incoming, outgoing = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    def poll_input(timeout):
        return bool(select.select([incoming], [], [], timeout)[0])

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        serve(incoming, outgoing, poll_input=poll_input)
    except _UnsafeROSShutdown:
        # rospy's exit hooks may also wait on a blocked synchronous publisher.
        # Let the OS tear down the sockets.
        os._exit(1)


if __name__ == "__main__":
    main()
