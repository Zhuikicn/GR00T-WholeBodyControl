"""Optional RTC playback scheduling, independent of sensors and robot transport."""

from dataclasses import dataclass
import math
import queue
import threading
import time

from gear_sonic.utils.inference.inference_recorder import utc_timestamp


class RTCUnsupportedError(RuntimeError):
    """The requested RTC contract was not acknowledged by the policy server."""


@dataclass(frozen=True)
class RTCSettings:
    horizon: int
    publish_rate: float
    execution_steps: int = 0
    delay_ms: float = 100.0
    ramp_rate: float = 5.0

    def __post_init__(self):
        for name in ("publish_rate", "delay_ms", "ramp_rate"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"RTC {name} must be finite and positive")
        if self.horizon < 1 or self.execution_steps < 0:
            raise ValueError("RTC horizon must be positive and execution_steps nonnegative")
        if not 1 <= self.execution_threshold <= self.horizon - self.frozen_steps - 2:
            raise ValueError(
                "RTC requires execution_steps + frozen_steps + 2 <= action_horizon "
                f"(got {self.execution_threshold} + {self.frozen_steps} + 2 > {self.horizon})"
            )

    @property
    def execution_threshold(self):
        return self.execution_steps or self.horizon // 2

    @property
    def frozen_steps(self):
        return math.ceil(self.delay_ms * self.publish_rate / 1000)


class RTCPlayback:
    """Own the original model chunk and its next playback index under one lock.

    Request snapshots are made after observation acquisition. Frame numbers never
    reset, even when the deployment protocol's frame counter does. A generation
    invalidates in-flight results on keyboard state changes.
    """

    def __init__(self, settings, recorder=None):
        self.settings = settings
        self.recorder = recorder
        self.lock = threading.RLock()
        self.generation = 0
        self.frame = 0
        self.cursor = 0
        self.chunk_id = 0
        self.raw_action = None
        self.pending = None
        self.exhausted = False
        self.exhaustion_count = 0
        self.retry_at = 0.0

    def event(self, event_type, **fields):
        if self.recorder is not None:
            self.recorder.record(
                event_type=event_type,
                timestamp_utc=utc_timestamp(),
                generation=self.generation,
                chunk_id=self.chunk_id,
                **fields,
            )

    def invalidate(self, reason):
        with self.lock:
            self.generation += 1
            self.raw_action = None
            self.cursor = 0
            self.exhausted = False
            self.retry_at = 0.0
            # Keep pending reserved until the worker's result is consumed.
            self.event("rtc_reset", reason=reason)

    def reserve(self, prompt, paused=False):
        with self.lock:
            if paused or self.pending is not None or time.monotonic() < self.retry_at:
                return None
            if self.raw_action is not None and self.cursor < self.settings.execution_threshold:
                return None
            request = {
                "request_id": time.monotonic_ns(),
                "generation": self.generation,
                "prompt": prompt,
            }
            self.pending = request["request_id"]
            return request

    def snapshot(self, request):
        with self.lock:
            if request["generation"] != self.generation:
                return None
            overlap = max(0, self.settings.horizon - self.cursor)
            # Too little committed motion remains to cover the delay budget.
            # Recover through a fresh, unconditioned chunk rather than freezing
            # actions that are no longer available.
            use_prefix = self.raw_action is not None and overlap >= self.settings.frozen_steps
            rtc = {
                "enabled": True,
                "overlap_steps": overlap if use_prefix else 0,
                "frozen_steps": self.settings.frozen_steps if use_prefix else 0,
                "ramp_rate": self.settings.ramp_rate,
            }
            if use_prefix:
                rtc["previous_actions"] = {k: v.copy() for k, v in self.raw_action.items()}
            metadata = {
                **request,
                "reference_frame": self.frame,
                "previous_chunk_id": self.chunk_id,
                "previous_cursor": self.cursor,
                "overlap_steps": rtc["overlap_steps"],
                "frozen_steps": rtc["frozen_steps"],
                "delay_budget_ms": self.settings.delay_ms,
                "started": time.monotonic(),
            }
            return {"rtc": rtc}, metadata

    def accept(self, result):
        """Consume exactly one completion; return (original action, start index)."""
        with self.lock:
            request = result["request"]
            if self.pending == request["request_id"]:
                self.pending = None
            if request["generation"] != self.generation:
                self.event("rtc_chunk_discarded", request_id=request["request_id"], reason="generation_changed")
                return None
            if result.get("fatal"):
                raise RTCUnsupportedError(result["fatal"])
            action = result.get("action")
            metadata = result.get("metadata")
            if action is None or metadata is None:
                self.retry_at = time.monotonic() + 0.1
                self.event("rtc_request_failed", request_id=request["request_id"], error=result.get("error"))
                return None
            elapsed = time.monotonic() - metadata["started"]
            index = self.frame - metadata["reference_frame"]
            over_budget = elapsed * 1000 > self.settings.delay_ms
            if index >= self.settings.horizon or elapsed >= self.settings.horizon / self.settings.publish_rate:
                self.event(
                    "rtc_chunk_discarded", request_id=request["request_id"], reason="expired", start_index=index
                )
                print(f"[Warning] RTC discarded expired chunk (delay {elapsed * 1000:.1f} ms)")
                self.retry_at = time.monotonic() + 0.1
                return None
            self.raw_action = {k: v.copy() for k, v in action.items()}
            self.cursor = index
            self.chunk_id += 1
            self.exhausted = False
            self.event(
                "rtc_chunk_received",
                request_id=request["request_id"],
                start_index=index,
                reference_frame=metadata["reference_frame"],
                overlap_steps=metadata["overlap_steps"],
                frozen_steps=metadata["frozen_steps"],
                actual_delay_ms=elapsed * 1000,
                delay_budget_exceeded=over_budget,
            )
            if over_budget:
                print(
                    f"[Warning] RTC delay {elapsed * 1000:.1f} ms exceeds budget {self.settings.delay_ms:.1f} ms"
                )
            return action, index

    def published(self):
        with self.lock:
            self.frame += 1
            if self.raw_action is not None:
                self.cursor += 1

    def playback_index(self):
        with self.lock:
            if self.raw_action is not None and self.cursor >= self.settings.horizon and not self.exhausted:
                self.exhausted = True
                self.exhaustion_count += 1
                print("[Warning] RTC chunk exhausted; repeating last frame until a fresh chunk arrives.")
                self.event("rtc_chunk_exhausted", exhaustion_count=self.exhaustion_count, frame=self.frame)
            return min(self.cursor, self.settings.horizon - 1)


def rtc_worker_loop(inference_queue, result_queue, stop_event, playback, prepare_obs_fn, inference_fn):
    """Single worker; always report completion, including errors and invalidations."""
    while not stop_event.is_set():
        try:
            request = inference_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        result = {"request": request, "action": None}
        try:
            observation = prepare_obs_fn(request["prompt"])
            snapshot = playback.snapshot(request) if observation is not None else None
            if snapshot is not None:
                options, metadata = snapshot
                result["metadata"] = metadata
                result["action"] = inference_fn(observation, options, metadata)
        except RTCUnsupportedError as exc:
            result["fatal"] = str(exc)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[Warning] RTC inference failed: {exc}")
        result_queue.put(result)
