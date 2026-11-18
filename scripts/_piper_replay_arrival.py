"""Arrival evidence from replay SDK call boundaries and follower feedback."""

from bisect import bisect_right
from collections import Counter, defaultdict
import time

import numpy as np


class ReplayArrival:
    def __init__(self, policy_hz, chunk_size, joint_tol, gripper_tol):
        self.period = 1.0 / policy_hz
        self.chunk_size = chunk_size
        self.tolerance = [joint_tol] * 6 + [gripper_tol]
        self.frame = None
        self.channels = {}
        self.calls = defaultdict(list)
        self.samples = defaultdict(list)
        self.sample_times = defaultdict(list)
        self.stopped_at = None

    def begin_call(self, channel, kind, value):
        if self.frame is None and not self.calls:
            return None
        call = dict(frame=self.frame, kind=kind, target=value,
                    start_mono_sec=time.monotonic(), return_mono_sec=None)
        self.calls[channel].append(call)
        return call

    def observe(self, evidence):
        if not self.calls or self.stopped_at is not None:
            return
        observed = evidence["sampled_mono_sec"]
        offset = observed - evidence["sampled_wall_sec"]
        for index, side in enumerate(self.channels):
            stamps = tuple(evidence["sdk_wall_sec"][2 * index:2 * index + 2])
            samples = self.samples[side]
            if samples and stamps == samples[-1]["sdk_wall_sec"]:
                continue
            samples.append(dict(observed_mono_sec=observed, sdk_wall_sec=stamps,
                                estimated_sdk_mono_sec=[stamp + offset for stamp in stamps],
                                actual=evidence["actual"][7 * index:7 * index + 7]))
            self.sample_times[side].append(observed)

    def stop(self):
        if self.stopped_at is None:
            self.stopped_at = time.monotonic()

    def _window(self, side, calls, deadline):
        joint, grip = calls.get("joint"), calls.get("gripper")
        returned = [call["return_mono_sec"] if call else None for call in (joint, grip)]
        samples = []
        if None not in returned and deadline is not None:
            times = self.sample_times[side]
            begin = bisect_right(times, min(returned))
            end = bisect_right(times, deadline)
            for sample in self.samples[side][begin:end]:
                if all(sent < stamp <= sample["observed_mono_sec"]
                       for sent, stamp in zip(returned, sample["estimated_sdk_mono_sec"])):
                    samples.append(sample)
        result = dict(status="unknown", deadline_mono_sec=deadline,
                      distinct_samples=len(samples), first_reached_mono_sec=None,
                      closest_feedback=None, last_feedback=None)
        if not samples:
            return result
        target = joint["target"] + [grip["target"]]
        errors = np.asarray([sample["actual"] for sample in samples]) - target
        normalized = (np.abs(errors) / self.tolerance).max(axis=1)
        hits = np.flatnonzero(normalized <= 1.0)

        def describe(index):
            error = errors[index].tolist()
            return dict(samples[index], joint_error_rad=error[:6], gripper_error_m=error[6])

        result.update(status="reached" if len(hits) else "not_observed",
                      closest_feedback=describe(int(np.argmin(normalized))),
                      last_feedback=describe(-1))
        if len(hits):
            result["first_reached_mono_sec"] = samples[hits[0]]["observed_mono_sec"]
        return result

    def report(self):
        frames = {}
        for side, channel in self.channels.items():
            calls = self.calls[channel]
            index = 0
            while index < len(calls):
                call = calls[index]
                frame = call["frame"]
                if frame is None:
                    index += 1
                    continue
                start = call["start_mono_sec"]
                dispatch = {}
                while index < len(calls) and calls[index]["frame"] == frame:
                    send = calls[index]
                    dispatch[send["kind"]] = send
                    index += 1
                next_call = calls[index] if index < len(calls) else None
                end = next_call["start_mono_sec"] if next_call else self.stopped_at
                boundary = "next_replay_target" if next_call and next_call["frame"] is not None else (
                    "non_replay_send" if next_call else "stop")
                if self.stopped_at is not None and (end is None or self.stopped_at < end):
                    end, boundary = self.stopped_at, "stop"
                within = self._window(side, dispatch, min(start + self.period, end) if end else None)
                full = self._window(side, dispatch, end)
                frames.setdefault(frame, {})[side] = dict(
                    dispatch=dispatch, boundary=boundary,
                    next_frame_index=next_call["frame"] if boundary == "next_replay_target" else None,
                    window_sec=end - start if end is not None else None,
                    within_nominal_period=within, before_replacement_or_stop=full,
                    first_reached_after_nominal_period=(
                        full["first_reached_mono_sec"] is not None and
                        full["first_reached_mono_sec"] > start + self.period),
                )
        frame_rows = []
        counts = {key: Counter() for key in ("within_nominal_period", "before_replacement_or_stop")}
        late_interchunk = 0
        sorted_frames = sorted(frames)
        for frame in sorted_frames:
            sides = frames[frame]
            final = frame == sorted_frames[-1]
            interchunk = any(
                item["next_frame_index"] is not None and item["next_frame_index"] % self.chunk_size == 0
                for item in sides.values()
            )
            statuses = {}
            for key, counter in counts.items():
                states = [sides.get(side, {}).get(key, {}).get("status", "unknown")
                          for side in self.channels]
                status = ("unknown" if "unknown" in states else
                          "reached" if all(state == "reached" for state in states) else "not_observed")
                statuses[key] = status
                if not final:
                    counter[status] += 1
            late = statuses["before_replacement_or_stop"] == "reached" and any(
                item["first_reached_after_nominal_period"] for item in sides.values())
            late_interchunk += int(interchunk and late)
            frame_rows.append(dict(frame_index=frame, final_attempted_frame=final,
                                   interchunk=interchunk, status=statuses,
                                   reached_after_nominal_period=late, sides=sides))
        return dict(
            nominal_period_sec=self.period, joint_tolerance_rad=self.tolerance[0],
            gripper_tolerance_m=self.tolerance[-1],
            evidence="Distinct host-side SDK joint/gripper cache samples, compared with SDK inputs "
                     "before possible clamping or quantization. SDK call return is not device acknowledgement. "
                     "Each side needs a distinct pair of fresh joint/gripper timestamps after the respective "
                     "SDK call returns. Samples are from normal control-loop snapshots. Both sides must "
                     "reach independently to count a frame. Unknown means no eligible feedback or incomplete "
                     "dispatch. The final window ends on normal replacement or stop, with no added settling.",
            attempted_frames=len(frame_rows),
            nonfinal_frames={key: {status: count[status] for status in
                                   ("reached", "not_observed", "unknown")}
                             for key, count in counts.items()},
            interchunk_frames_reached_after_nominal_period=late_interchunk,
            final_attempted_frame=frame_rows[-1] if frame_rows else None,
            frames=frame_rows,
        )
