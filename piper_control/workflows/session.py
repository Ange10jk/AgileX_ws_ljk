"""Control loop: SDK motion and intervention remain independent of inference."""

import math
import queue
import select
import signal
import sys
import threading
import time

import numpy as np

from ..hardware.cameras import CameraRig, MockCameras
from ..control.arbiter import Controller, ControlError
from ..config import component_names
from ..observation import make_observation
from ..policies.client import PolicyClient
from ..policies.interfaces import PolicyError, PolicyWorkerError
from ..policies.lcgen_http import GRIPPER_RANGE_M, validate_trajectory
from ..recording.session import RecordingSession
from ..hardware.robot import SDKRobot, MockRobot, RobotError
from .definitions import get_workflow


def _stored_observation(snapshot, frames):
    return {
        "state": snapshot["state"].copy(),
        "stamp": snapshot["stamp"],
        "monotonic": snapshot["monotonic"],
        "frames": frames,
    }


class Session:
    """Compose SDK ownership, policy workers and an optional collection sink."""

    def __init__(
        self,
        config,
        mock=False,
        command_schedule=None,
        robot=None,
        cameras=None,
        policy=None,
        recording=None,
    ):
        self.cfg, self.mock = config, mock
        self.workflow = get_workflow(config["workflow"]["mode"])
        self.robot = robot or (MockRobot if mock else SDKRobot)(config["robot"])
        self.cameras = None
        if not config["record"].get("camera_in_ros_worker", False):
            self.cameras = cameras or (MockCameras if mock else CameraRig)(
                config["observation"]
            )
        self.policy = policy
        self._lcgen = config["policy"].get("backend") == "lcgen_http"
        self._lcgen_reply = None
        self._lcgen_observed_step = None
        self._lcgen_sdk_limits = None
        self.recording = recording or RecordingSession(config)
        self.controller = Controller(config, self.robot, self.event)
        self.commands = queue.Queue()
        self.schedule = sorted(command_schedule or [])
        self.pending = None
        self.episode_id = 0
        self.step_id = 0
        self.total_steps = 0
        self.episode_active = False
        self.episode_started = None
        self.needs_boundary = True
        self.learning = self.workflow.learns
        self.learning_pending = False
        self.learning_started = None
        self.learning_error = None
        self.seq = 0
        self.quit = False
        self.started = False
        self.last_observation = None
        self._human_action_ready = False
        self._signals = {}
        self._stop_wakeup = threading.Event()
        self._closing = threading.Event()
        self._stop_thread = None
        self._stop_error = None
        self._camera_recording = False

    def event(self, event):
        self.recording.write_event(event)
        kind = event.get("type")
        if kind == "mode":
            self._human_action_ready = False
            descriptions = {
                Controller.HUMAN: "主臂模式和转发已开启，等待主臂控制数据",
                Controller.POLICY: "模型控制已开启",
                Controller.HOLD: "已暂停并保持姿态",
            }
            state = event["control_state"]
            print("[控制] {}：{}".format(state, descriptions.get(state, state)), flush=True)
        elif kind in ("command_rejected", "fault", "policy_error"):
            print("[控制] {}：{}".format(kind, event.get("reason", "")), flush=True)

    def _log(self, kind, **fields):
        self.event(
            dict(type=kind, timestamp=time.time(), episode_id=self.episode_id, **fields)
        )

    def start(self):
        self.recording.start()
        if (self.cfg["record"].get("backend") in ("ros", "ros_publish")
                and hasattr(self.cameras, "record_to")):
            self.cameras.record_to(
                self.recording.publish_images,
                capacity=int(self.cfg["record"].get("camera_queue_size", 32)))
            self._camera_recording = True
        if self.cameras is not None:
            self.cameras.start()
        if self.workflow.uses_policy:
            if self.learning and self.recording.directory is not None:
                options = self.cfg["workflow"].setdefault("rlt", {})
                options.setdefault(
                    "output_dir", str(self.recording.directory / "training")
                )
            self._load_policy()
            if self._lcgen and not self.mock:
                self._lcgen_sdk_limits = self.robot.joint_limits()
                if (self._lcgen_sdk_limits.shape != (6, 2)
                        or not np.isfinite(self._lcgen_sdk_limits).all()
                        or np.any(self._lcgen_sdk_limits[:, 0] >= self._lcgen_sdk_limits[:, 1])):
                    raise RobotError("Invalid Piper SDK joint limits")
            capabilities = getattr(self.policy, "capabilities", {})
            self.episode_id = int(capabilities.get("episode_id", 0))
        self.robot.connect()
        manual_auto = self.cfg["workflow"].get("auto_start", False) and not self.workflow.uses_policy
        if manual_auto:
            if self.cameras is None:
                self.recording.check_health()
            if self.controller.abort.is_set():
                raise ControlError("Stop requested before enabling")
            print("[启动] 将主臂切回 follower 以读取状态，检查驱动并按需恢复电子停止。", flush=True)
            self.robot.prepare_human_start(stop_requested=self.controller.abort.is_set)
            print("[启动] 恢复检查已完成，正在使能并发送 JS 对齐目标。", flush=True)
        # Load process resources before starting SDK control.
        if self.cameras is None:
            self.recording.check_health()
        self.controller.start()
        self.started = True
        if manual_auto:
            print("[启动] 使能已确认，已发送一次 move_js()，正在等待运动稳定。", flush=True)
            alignment = self.robot.wait_for_alignment(
                self.controller.target, stop_requested=self.controller.abort.is_set
            )
            if alignment:
                self._log("startup_alignment", **alignment)
                print("[启动] 主从当前姿态差（不要求精确到位）：{}".format(alignment["pair_errors"]), flush=True)
                print("[启动] 相对初始目标的偏差（诊断）：{}".format(alignment["target_errors"]), flush=True)
            print("[启动] JS 运动已稳定，正在自动进入主从控制。", flush=True)
        self._log(
            "session_ready",
            mock=self.mock,
            policy_type=self.cfg["policy"]["type"],
            arms=self.cfg["robot"]["arms"],
            action_names=component_names(self.cfg["robot"]["arms"]),
            policy_capabilities=getattr(self.policy, "capabilities", None),
        )
        return self

    def _load_policy(self):
        self.policy = self.policy or PolicyClient(self.cfg)
        self.policy.load()

    def _camera_snapshot(self):
        return self.cameras.snapshot() if self.cameras is not None else {}

    def _observation(self, snapshot, frames):
        return make_observation(
            self.cfg, snapshot, frames, include_images=self.cameras is not None
        )

    def _invalidate_policy(self):
        self._lcgen_reply = None
        self._lcgen_observed_step = None
        if self.policy:
            self.policy.invalidate(self.controller.generation)
        self.needs_boundary = True

    def _finish_step(
        self, snapshot, frames, terminal=False, truncated=False, reward=0.0, observation=None
    ):
        if self.pending is None:
            return
        step = dict(self.pending)
        next_obs = observation if observation is not None else self._observation(snapshot, frames)
        step["next_observation"] = _stored_observation(snapshot, frames)
        step["duration_sec"] = max(0.0, time.monotonic() - step.pop("_started"))
        before_obs = step.pop("_policy_observation")
        requires_boundary = step.pop("_boundary_required")
        segment_start = step.pop("_segment_start")
        step["execution_boundary"] = requires_boundary
        step["segment_start"] = segment_start
        # Sparse task reward is stored once, in episode_end. Zero here avoids
        # counting it again when replay is reconstructed offline.
        step.update(terminated=bool(terminal), truncated=bool(truncated), reward=0.0)
        self.recording.write_step(step)
        self.pending = None
        if self.learning:
            if requires_boundary:
                # The learner receives the real execution boundary. Its default
                # collector defers costly VLA encoding until the robot is held.
                self.policy.boundary(
                    before_obs,
                    self.episode_id,
                    self.step_id,
                    segment_start=segment_start,
                )
            self.policy.record_step(
                step["action"],
                step["source"],
                intervention=step["intervention"],
                reward=0.0,
                next_observation=next_obs,
                ref_row=step.get("metadata", {}).get("rlt_ref_row"),
            )
        self.step_id += 1
        self.total_steps += 1

    def _begin_episode(self):
        if not self.episode_active:
            self.episode_active = True
            self.episode_started = time.monotonic()
            self.step_id = 0
            self.needs_boundary = True
            self._log("episode_start")

    def command(self, command, snapshot, frames):
        command = command.strip().lower()
        if command in ("p", "policy", "h", "human"):
            destination = (
                Controller.POLICY if command in ("p", "policy") else Controller.HUMAN
            )
            if destination == Controller.HUMAN and not self.robot.supports_human:
                self._log(
                    "command_rejected",
                    command=command,
                    reason="human control requires a leader for every selected arm",
                )
                return
            if self.controller.state == destination:
                return
            if self.learning_pending:
                self._log(
                    "command_rejected",
                    command=command,
                    reason="episode learner update still running",
                )
                return
            if command in ("p", "policy") and not self.workflow.uses_policy:
                self._log(
                    "command_rejected",
                    command=command,
                    reason="this manual workflow has no policy",
                )
                return
            if self._lcgen and self.policy is None:
                try:
                    self._load_policy()
                except (OSError, PolicyError) as exc:
                    self._close_policy()
                    self._log("policy_error", reason=str(exc))
                    return
            self._finish_step(snapshot, frames)
            if command in ("p", "policy"):
                self.controller.enter_policy()
            else:
                self.controller.enter_human()
            self._invalidate_policy()
            self._begin_episode()
        elif command in ("hold", "pause", " "):
            if self._lcgen and self.episode_active:
                self.end_episode(snapshot, frames, False, True, 0.0, False)
                return
            self._finish_step(snapshot, frames)
            self.controller.pause()
            self._invalidate_policy()
            self._log("episode_pause")
        elif command in ("s", "success", "f", "failure", "t", "truncate"):
            if not self.episode_active:
                return
            success = command in ("s", "success")
            truncated = command in ("t", "truncate")
            reward = (
                self.cfg["workflow"]["success_reward"]
                if success
                else 0.0 if truncated else self.cfg["workflow"]["failure_reward"]
            )
            self.end_episode(
                snapshot, frames, not truncated, truncated, reward, success
            )
        elif command in ("q", "quit"):
            if self.episode_active:
                self.end_episode(snapshot, frames, False, True, 0.0, False)
            self.quit = True
        elif command:
            self._log("command_rejected", command=command, reason="unknown command")

    def end_episode(self, snapshot, frames, terminal, truncated, reward, success):
        final_obs = {} if self._lcgen else self._observation(snapshot, frames)
        self._finish_step(
            snapshot, frames, terminal=terminal, truncated=truncated, reward=reward,
            observation=final_obs if self._lcgen else None,
        )
        self.controller.pause()
        self._invalidate_policy()
        self._log(
            "episode_end",
            terminal=terminal,
            truncated=truncated,
            reward=reward,
            success=success,
            steps=self.step_id,
            reward_in_final_step=False,
            final_observation=_stored_observation(snapshot, frames),
        )
        self.episode_active = False
        if self.learning:
            self.policy.end_episode(
                final_obs,
                terminal=terminal,
                reward=reward,
                success=success,
                truncated=truncated,
            )
            self.learning_pending = True
            self.learning_started = time.monotonic()
        else:
            self.episode_id += 1
        self.step_id = 0

    def _close_policy(self):
        policy, self.policy = self.policy, None
        if policy is not None:
            policy.close()

    def _lcgen_fail(self, exc, snapshot, frames):
        restart = isinstance(exc, PolicyWorkerError)
        if restart:
            # Prevent invalidation from enqueueing onto the broken worker.
            worker, self.policy = self.policy, None
        if self.episode_active:
            self.end_episode(snapshot, frames, False, True, 0.0, False)
        self._log("policy_error", reason=str(exc))
        if restart:
            if worker is not None:
                worker.close()
            try:
                self._load_policy()
            except (OSError, PolicyError) as load_error:
                self._close_policy()
                self._log("policy_error", reason="Worker restart failed: %s" % load_error)

    def _poll(self, snapshot=None, frames=None):
        if not self.policy:
            return
        while True:
            try:
                reply = self.policy.poll()
            except PolicyError as exc:
                if not self._lcgen:
                    raise
                self._lcgen_fail(exc, snapshot, frames)
                return
            if reply is None:
                return
            if reply.get("kind", "infer") == "infer":
                if not self._lcgen:
                    self.controller.offer(reply)
                elif (self.controller.state == Controller.POLICY
                      and reply.get("generation") == self.controller.generation
                      and self.controller.pending_seq is not None
                      and reply.get("seq") == self.controller.pending_seq):
                    self._lcgen_reply = reply
            elif reply.get("kind") == "end_episode":
                self._log("learner_result", result=reply.get("result"))
                self.learning_pending = False
                self.episode_id += 1

    def _lcgen_observation(self, snapshot, frames):
        max_skew = self.cfg["policy"]["lcgen_http"]["max_observation_skew_sec"]
        now = time.time()
        feedback = {}
        for side in self.cfg["robot"]["arms"]:
            follower = snapshot.get("arms", {}).get(side, {}).get("follower", {})
            stamps = {name: follower.get(name) for name in ("joint_timestamp", "gripper_timestamp")}
            if not follower.get("valid") or not all(
                stamp is not None and math.isfinite(stamp) and abs(now - stamp) <= max_skew
                for stamp in stamps.values()
            ):
                raise RobotError("Stale or unavailable follower feedback: " + side)
            feedback[side] = stamps
        state = np.asarray(snapshot["state"], dtype=float)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise RobotError("Invalid measured 14D follower state")
        try:
            observation = self._observation(snapshot, frames)
        except ValueError as exc:
            raise PolicyError(str(exc)) from exc
        cameras = {name: frames[name].get("stamp") for name in self.cfg["observation"]["camera_map"].values()}
        if not all(stamp is not None and math.isfinite(stamp) and abs(now - stamp) <= max_skew
                   for stamp in cameras.values()):
            raise PolicyError("Stale or unavailable RoboDojo camera frames")
        stamps = [*cameras.values(), *(stamp for side in feedback.values() for stamp in side.values())]
        if max(stamps) - min(stamps) > max_skew:
            raise PolicyError("RoboDojo camera and follower timestamps exceed the allowed skew")
        # Keep the measured grip values in float64 so 0.1 m serializes as exactly 1.0.
        if np.any((state[[6, 13]] < 0.0) | (state[[6, 13]] > GRIPPER_RANGE_M)):
            raise RobotError("Measured Piper gripper width is outside [0, 0.1] m")
        observation["observation.state"] = state.copy()
        observation["camera_timestamps"] = cameras
        observation["follower_timestamps"] = feedback
        return observation

    def _lcgen_tick(self, snapshot, frames):
        if self.controller.state != Controller.POLICY:
            return
        reply = self._lcgen_reply
        if reply is None and self.controller.pending_seq is not None:
            return
        try:
            observation = self._lcgen_observation(snapshot, frames)
            self._finish_step(snapshot, frames, observation=observation)
            self.last_observation = observation
            observation["wam_frame_index"] = self.step_id
            if reply is not None:
                self._lcgen_reply = None
                try:
                    validate_trajectory(reply["actions"], snapshot["state"],
                                        self.policy.capabilities, self._lcgen_sdk_limits)
                except PolicyError as exc:
                    meta = reply.get("metadata", {})
                    raise PolicyError(
                        "RoboDojo action chunk rejected status=%s elapsed=%s request_id=%s: %s"
                        % (meta.get("http_status"), meta.get("elapsed_sec"),
                           meta.get("request_id"), exc)
                    ) from exc
                self.controller.offer(reply)
            if self._lcgen_observed_step != self.step_id:
                self._lcgen_observed_step = self.step_id
                if not self.controller.actions:
                    self.seq += 1
                    self.controller.expect(self.seq)
                    self.policy.submit(observation, self.controller.generation, self.seq)
                    return
                self.policy.observe(observation, self.controller.generation)
        except PolicyError as exc:
            self._lcgen_fail(exc, snapshot, frames)
            return
        self._dispatch(snapshot, frames, observation)

    def _policy_tick(self, snapshot, frames):
        if self._lcgen:
            self._lcgen_tick(snapshot, frames)
            return
        self._finish_step(snapshot, frames)
        observation = self._observation(snapshot, frames)
        self.last_observation = observation
        self._dispatch(snapshot, frames, observation)
        # Refill after retaining the command that was actually sent.
        if self.controller.state == Controller.POLICY:
            if not self.controller.actions and self.controller.pending_seq is None:
                self.seq += 1
                self.controller.expect(self.seq)
                self.policy.submit(observation, self.controller.generation, self.seq)

    def _dispatch(self, snapshot, frames, observation):
        step = self.controller.step(snapshot)
        if step is not None:
            metadata = step.get("metadata", {})
            step.update(
                episode_id=self.episode_id,
                step_id=self.step_id,
                timestamp=time.time(),
                action_evidence=(
                    "forwarded_can_target"
                    if step["source"] == "HUMAN"
                    else "sdk_target"
                ),
                observation=_stored_observation(snapshot, frames),
                intervention=step["source"] == "HUMAN",
                reward=0.0,
                terminated=False,
                truncated=False,
                boundary=metadata.get("rlt_boundary"),
                _started=time.monotonic(),
                _policy_observation=observation,
                _segment_start=self.needs_boundary,
                _boundary_required=(self.needs_boundary or "rlt_boundary" in metadata),
            )
            self.needs_boundary = False
            self.pending = step
            self.recording.publish_action(
                step["action"],
                step["source"],
                {
                    "timestamp": step["timestamp"],
                    "episode_id": self.episode_id,
                    "step_id": self.step_id,
                    "generation": self.controller.generation,
                    "predicted_action": step["predicted_action"],
                    "action_evidence": step["action_evidence"],
                    "joint_names": component_names(self.cfg["robot"]["arms"]),
                },
            )
            if step["source"] == "HUMAN" and not self._human_action_ready:
                self._human_action_ready = True
                self._log(
                    "human_action_ready",
                    action_evidence=step["action_evidence"],
                    human_action_seq=snapshot.get("human_action_seq"),
                    human_action_stamp=snapshot.get("human_action_stamp"),
                )
                print("[采集] 已收到主臂动作，人工 action 数据已就绪。", flush=True)

    def _input(self):
        # Line-oriented input also supports pipes and scripted demonstrations.
        while not self.quit and not self.controller.abort.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                line = sys.stdin.readline()
                if not line:
                    print("[控制] 标准输入已关闭，无法接收键盘命令；程序仍继续运行。", flush=True)
                    return
                if line.strip().lower() in ("stop", "estop"):
                    self.controller.request_stop()
                    return
                self.commands.put(line.strip())
            except (OSError, ValueError) as exc:
                print("[控制] 无法读取终端命令：{}".format(exc), flush=True)
                return

    def _signal_stop(self, signum, frame):
        # A Python signal may interrupt an SDK send while it owns a TX lock.
        # Never re-enter SDK I/O from that handler; a normal thread stops it.
        self.controller.abort.set()
        self._stop_wakeup.set()

    def _relay_stop(self):
        self._stop_wakeup.wait()
        if self.controller.abort.is_set() and not self._closing.is_set():
            try:
                self.controller.request_stop()
            except Exception as exc:
                self._stop_error = str(exc)

    def run(self, max_steps=None, max_seconds=None, interactive=True):
        error = None
        try:
            if threading.current_thread() is threading.main_thread():
                self._stop_thread = threading.Thread(
                    target=self._relay_stop, name="signal-stop", daemon=True
                )
                self._stop_thread.start()
                for signum in (signal.SIGINT, signal.SIGTERM):
                    self._signals[signum] = signal.getsignal(signum)
                    signal.signal(signum, self._signal_stop)
            self.start()
            if interactive:
                commands = []
                if self.workflow.uses_policy:
                    commands.append("p=model")
                if self.robot.supports_human:
                    commands.append("h=human")
                commands.extend(
                    (
                        "hold=pause",
                        "s=success",
                        "f=failure",
                        "t=timeout",
                        "q=quit",
                        "stop=emergency stop",
                    )
                )
                print("Commands + Enter: " + ", ".join(commands), flush=True)
                threading.Thread(
                    target=self._input, name="operator-input", daemon=True
                ).start()
            started = time.monotonic()
            next_policy = started
            next_control = started
            policy_hz = self.cfg["workflow"]["policy_hz"]
            control_hz = self.cfg["robot"]["control_hz"]
            control_period = 1.0 / control_hz
            collect_on_control_tick = (
                self.workflow.name == "collect" and policy_hz == control_hz
            )
            auto = self.cfg["workflow"].get("auto_start", False)
            episode_limit = self.cfg["workflow"]["max_episode_sec"]
            while True:
                now = time.monotonic()
                policy_due = collect_on_control_tick or now >= next_policy
                self.recording.check_health()
                if self.cameras is not None:
                    self.cameras.check_health()
                if not self._lcgen:
                    self._poll()
                snapshot, frames = self.robot.snapshot(), self._camera_snapshot()
                if self._lcgen:
                    self._poll(snapshot, frames)
                self.recording.publish_state(snapshot)
                if self.cameras is not None and not self._camera_recording:
                    self.recording.publish_images(frames)
                self.controller.tick(snapshot)
                if auto:
                    self.command(self.workflow.initial_command, snapshot, frames)
                    auto = False
                    snapshot, frames = self.robot.snapshot(), self._camera_snapshot()
                while self.schedule and now - started >= self.schedule[0][0]:
                    _, command = self.schedule.pop(0)
                    self.commands.put(command)
                while not self.commands.empty():
                    self.command(self.commands.get_nowait(), snapshot, frames)
                    if self.quit:
                        break
                    snapshot, frames = self.robot.snapshot(), self._camera_snapshot()
                if (
                    self.episode_active
                    and episode_limit is not None
                    and now - self.episode_started >= episode_limit
                ):
                    self.command("t", snapshot, frames)
                limit = (
                    max_steps is not None
                    and (
                        self.total_steps >= max_steps
                        or (
                            self.pending is not None
                            and self.total_steps + 1 >= max_steps
                            and policy_due
                        )
                    )
                ) or (max_seconds is not None and now - started >= max_seconds)
                if limit and not self.quit:
                    self.command("q", snapshot, frames)
                if self.learning_pending:
                    timeout = self.cfg["policy"].get(
                        "update_timeout_sec",
                        self.cfg["workflow"]
                        .get("rlt", {})
                        .get("update_timeout_sec", 600.0),
                    )
                    if now - self.learning_started > timeout:
                        raise RuntimeError(
                            "RLT update timed out while robot was paused"
                        )
                if self.quit and not self.learning_pending:
                    break
                if policy_due and self.episode_active:
                    self._policy_tick(snapshot, frames)
                    if not collect_on_control_tick:
                        next_policy = now + 1.0 / policy_hz
                if self._lcgen:
                    if now >= next_control:
                        next_control = now + control_period
                else:
                    next_control += control_period
                if collect_on_control_tick:
                    remaining = next_control - time.monotonic()
                    if remaining < 0:
                        # Missed slots have no historical action to record.
                        skipped = math.ceil(-remaining / control_period)
                        next_control += skipped * control_period
                    time.sleep(max(0.0, next_control - time.monotonic()))
                else:
                    wakeup = next_control
                    if self._lcgen and self.episode_active:
                        wakeup = min(wakeup, next_policy)
                    time.sleep(max(0.0, min(wakeup - time.monotonic(), 0.05)))
                    if next_control < time.monotonic() - 0.1:
                        next_control = time.monotonic()
            self._log("session_end", steps=self.total_steps)
        except BaseException as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            if self.started:
                try:
                    # Motor stop must not depend on disk, ROS, or camera health.
                    self.controller.fail(error)
                except Exception:
                    pass
            if self.pending is not None:
                try:
                    # Keep evidence of a command that was actually sent even if
                    # no valid successor observation can be acquired.
                    incomplete = {
                        k: v for k, v in self.pending.items() if not k.startswith("_")
                    }
                    incomplete.update(
                        next_observation=None,
                        execution_complete=False,
                        duration_sec=max(
                            0.0, time.monotonic() - self.pending["_started"]
                        ),
                        fault=error,
                        truncated=True,
                    )
                    self.recording.write_step(incomplete)
                    self.pending = None
                except Exception:
                    pass
            raise
        finally:
            self.quit = True
            self._closing.set()
            self._stop_wakeup.set()
            errors = []
            for component in (self.robot, self.policy, self.cameras):
                if component is not None:
                    try:
                        component.close()
                    except Exception as exc:
                        errors.append(str(exc))
            if self._stop_thread is not None:
                self._stop_thread.join(timeout=1.0)
                if self._stop_thread.is_alive():
                    errors.append("Signal stop relay did not exit")
            if self._stop_error:
                errors.append(self._stop_error)
            try:
                self.recording.close(
                    error=error or ("; ".join(errors) if errors else None)
                )
            finally:
                for signum, handler in self._signals.items():
                    signal.signal(signum, handler)
            if not error and errors:
                raise RuntimeError("Shutdown failed: " + "; ".join(errors))
        return {
            "steps": self.total_steps,
            "episodes": self.episode_id,
            "directory": (
                str(self.recording.directory) if self.recording.directory else None
            ),
        }
