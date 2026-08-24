# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Low-overhead correlation events for cross-role AFD profiling.

The recorder is disabled by default. When enabled, it streams every event
to a JSONL sidecar file immediately so that even a hard kill (SIGKILL)
preserves all data written so far. ``close()`` only writes the final clock
anchor and atomically renames the temp file. It never synchronizes an
accelerator or inspects tensor values.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import threading
import time
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, TextIO

AFDTraceRole = Literal["attention", "ffn"]
AFDTracePhase = Literal["begin", "end", "instant"]

AFD_TRACE_ENABLE_ENV: Final[str] = "AFD_TRACE_ENABLE"
AFD_TRACE_SESSION_ID_ENV: Final[str] = "AFD_TRACE_SESSION_ID"
AFD_TRACE_DIR_ENV: Final[str] = "AFD_TRACE_DIR"
AFD_TRACE_MAX_EVENTS_ENV: Final[str] = "AFD_TRACE_MAX_EVENTS"
AFD_TRACE_SCHEMA_VERSION: Final[int] = 1
DEFAULT_TRACE_DIR: Final[str] = "/tmp/afd-correlation-trace"
DEFAULT_MAX_EVENTS: Final[int] = 200_000
FLOW_DIGEST_BYTES: Final[int] = 8
TEMP_FILE_SUFFIX: Final[str] = ".tmp"
_TRUE_ENV_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSE_ENV_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})
_SAFE_FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class AFDCorrelationTraceConfig:
    """Environment-derived configuration for one trace recorder."""

    enabled: bool
    session_id: str | None
    trace_dir: Path
    max_events: int


@dataclass(frozen=True, slots=True)
class AFDTraceIdentity:
    """Stable process identity stored in every sidecar."""

    role: AFDTraceRole
    rank: int
    role_rank: int
    local_rank: int
    hostname: str
    pid: int


@dataclass(frozen=True, slots=True)
class AFDClockAnchor:
    """A near-simultaneous monotonic/raw and realtime clock sample."""

    label: str
    monotonic_ns: int
    realtime_ns: int
    capture_uncertainty_ns: int


@dataclass(frozen=True, slots=True)
class AFDCorrelationTraceEvent:
    """One scalar-only event emitted around an AFD execution boundary."""

    sequence: int
    monotonic_ns: int
    event: str
    phase: AFDTracePhase
    flow_id: str | None
    transaction_id: str | None
    layer_idx: int | None
    stage_idx: int | None
    num_tokens: int | None
    outcome: str | None


def afd_correlation_trace_config() -> AFDCorrelationTraceConfig:
    """Read correlation tracing settings without importing accelerator code."""

    enabled = _env_bool(AFD_TRACE_ENABLE_ENV, default=False)
    session_id = os.getenv(AFD_TRACE_SESSION_ID_ENV)
    if enabled and not session_id:
        raise ValueError(
            f"{AFD_TRACE_SESSION_ID_ENV} must be set when {AFD_TRACE_ENABLE_ENV}=1",
        )
    max_events = _env_positive_int(
        AFD_TRACE_MAX_EVENTS_ENV,
        default=DEFAULT_MAX_EVENTS,
    )
    return AFDCorrelationTraceConfig(
        enabled=enabled,
        session_id=session_id,
        trace_dir=Path(os.getenv(AFD_TRACE_DIR_ENV, DEFAULT_TRACE_DIR)),
        max_events=max_events,
    )


class AFDCorrelationTraceRecorder:
    """Buffer correlation events and atomically flush a JSONL sidecar."""

    def __init__(
        self,
        config: AFDCorrelationTraceConfig,
        identity: AFDTraceIdentity,
    ) -> None:
        self.config = config
        self.identity = identity
        self._lock = threading.Lock()
        self._sequence = 0
        self._dropped_events = 0
        self._closed = False
        self._output_path: Path | None = None
        self._temp_file: TextIO | None = None
        self._temp_path: Path | None = None
        if not self.enabled:
            return
        output_path = self._build_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._temp_path = output_path.with_name(
            output_path.name + TEMP_FILE_SUFFIX
        )
        # Open the sidecar file immediately and write the header
        # records.  Every subsequent event is streamed to disk so that
        # even a SIGKILL preserves everything written so far.
        try:
            self._temp_file = self._temp_path.open("w", encoding="utf-8")
        except OSError:
            self._temp_file = None
            return
        metadata = {
            "record_type": "metadata",
            "schema_version": AFD_TRACE_SCHEMA_VERSION,
            "session_id": self.config.session_id,
            "identity": asdict(self.identity),
            "clock": "CLOCK_MONOTONIC_RAW",
            "max_events": self.config.max_events,
            "dropped_events": 0,
        }
        _write_json_line(self._temp_file, metadata)
        start_anchor = _capture_clock_anchor("start")
        _write_json_line(
            self._temp_file,
            {"record_type": "clock_anchor", **asdict(start_anchor)},
        )
        self._temp_file.flush()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    def make_flow_id(
        self,
        transaction_id: str | None,
        *,
        layer_idx: int,
        stage_idx: int,
    ) -> str | None:
        """Build a stable logical exchange ID shared by Attention and FFN."""

        if not self.enabled or transaction_id is None:
            return None
        source = (
            f"{self.config.session_id}\0{transaction_id}\0"
            f"{int(layer_idx)}\0{int(stage_idx)}"
        )
        return hashlib.blake2b(
            source.encode("utf-8"),
            digest_size=FLOW_DIGEST_BYTES,
        ).hexdigest()

    def record(
        self,
        event: str,
        *,
        phase: AFDTracePhase = "instant",
        flow_id: str | None = None,
        transaction_id: str | None = None,
        layer_idx: int | None = None,
        stage_idx: int | None = None,
        num_tokens: int | None = None,
        outcome: str | None = None,
    ) -> None:
        """Append one event — streamed to the sidecar file immediately."""

        if not self.enabled:
            return
        with self._lock:
            if self._closed:
                return
            if self._sequence >= self.config.max_events:
                self._dropped_events += 1
                return
            sequence = self._sequence
            self._sequence += 1
            if self._temp_file is not None:
                try:
                    _write_json_line(
                        self._temp_file,
                        {
                            "record_type": "event",
                            "sequence": sequence,
                            "monotonic_ns": _monotonic_raw_ns(),
                            "event": event,
                            "phase": phase,
                            "flow_id": flow_id,
                            "transaction_id": transaction_id,
                            "layer_idx": layer_idx,
                            "stage_idx": stage_idx,
                            "num_tokens": num_tokens,
                            "outcome": outcome,
                        },
                    )
                    self._temp_file.flush()
                except OSError:
                    self._temp_file = None

    @contextmanager
    def record_range(
        self,
        event: str,
        *,
        flow_id: str | None = None,
        transaction_id: str | None = None,
        layer_idx: int | None = None,
        stage_idx: int | None = None,
        num_tokens: int | None = None,
    ) -> Generator[None, None, None]:
        """Record begin/end events and a profiler-visible named CPU range."""

        if not self.enabled:
            with nullcontext():
                yield
            return

        marker = _profiler_marker(event, flow_id)
        profiler_context = _record_function_context(marker)
        self.record(
            event,
            phase="begin",
            flow_id=flow_id,
            transaction_id=transaction_id,
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            num_tokens=num_tokens,
        )
        outcome = "ok"
        try:
            with profiler_context:
                yield
        except BaseException:
            outcome = "error"
            raise
        finally:
            self.record(
                event,
                phase="end",
                flow_id=flow_id,
                transaction_id=transaction_id,
                layer_idx=layer_idx,
                stage_idx=stage_idx,
                num_tokens=num_tokens,
                outcome=outcome,
            )

    def close(self) -> Path | None:
        """Finalize the sidecar and return its path.

        Data is already on disk (streamed by every ``record()`` call), so
        this only writes the closing clock anchor and renames the temp file.
        """
        if not self.enabled:
            return None
        with self._lock:
            if self._closed:
                return self._output_path
            self._closed = True
        output_path = self._build_output_path()
        if self._temp_file is not None:
            try:
                stop_anchor = _capture_clock_anchor("stop")
                _write_json_line(
                    self._temp_file,
                    {"record_type": "clock_anchor", **asdict(stop_anchor)},
                )
                self._temp_file.flush()
                self._temp_file.close()
            except OSError:
                pass
            self._temp_file = None
        if self._temp_path is not None and self._temp_path.exists():
            os.replace(self._temp_path, output_path)
        self._output_path = output_path
        return output_path

    def _build_output_path(self) -> Path:
        session_id = _safe_filename_component(str(self.config.session_id))
        hostname = _safe_filename_component(self.identity.hostname)
        filename = (
            f"afd-trace-{session_id}-{self.identity.role}-"
            f"rank{self.identity.role_rank}-pid{self.identity.pid}-{hostname}.jsonl"
        )
        return self.config.trace_dir / session_id / filename


def create_afd_correlation_trace_recorder(
    *,
    role: AFDTraceRole,
    rank: int,
    role_rank: int,
    local_rank: int,
) -> AFDCorrelationTraceRecorder:
    """Create one process-local recorder from environment configuration."""

    return AFDCorrelationTraceRecorder(
        afd_correlation_trace_config(),
        AFDTraceIdentity(
            role=role,
            rank=int(rank),
            role_rank=int(role_rank),
            local_rank=int(local_rank),
            hostname=socket.gethostname(),
            pid=os.getpid(),
        ),
    )


def _capture_clock_anchor(label: str) -> AFDClockAnchor:
    monotonic_before = _monotonic_raw_ns()
    realtime_ns = time.time_ns()
    monotonic_after = _monotonic_raw_ns()
    return AFDClockAnchor(
        label=label,
        monotonic_ns=(monotonic_before + monotonic_after) // 2,
        realtime_ns=realtime_ns,
        capture_uncertainty_ns=(monotonic_after - monotonic_before) // 2,
    )


def _monotonic_raw_ns() -> int:
    raw_clock = getattr(time, "CLOCK_MONOTONIC_RAW", None)
    if raw_clock is None:
        return time.monotonic_ns()
    return time.clock_gettime_ns(raw_clock)


def _record_function_context(marker: str) -> AbstractContextManager[object]:
    # torch_npu's profiler does NOT record torch.autograd.profiler
    # record_function names; it uses the MSTX user-range mechanism
    # (torch_npu.npu.mstx) instead, gated by ExperimentalConfig(mstx=True).
    # Use MSTX first so the marker name (including the flow_id suffix)
    # actually lands in the torch_npu trace and the merge tool can align
    # the device timeline to the correlation sidecar. mstx_range is a
    # decorator, not a context manager, so pair range_start/range_end.
    # Fall back to the torch CPU profiler API for non-NPU environments.
    try:
        import torch_npu

        @contextmanager
        def _mstx_range():
            range_id = torch_npu.npu.mstx.range_start(marker)
            try:
                yield
            finally:
                torch_npu.npu.mstx.range_end(range_id)

        return _mstx_range()
    except (ImportError, AttributeError, RuntimeError, TypeError):
        pass
    try:
        from torch.autograd.profiler import record_function
    except (ImportError, AttributeError):
        return nullcontext()
    return record_function(marker)


def _profiler_marker(event: str, flow_id: str | None) -> str:
    if flow_id is None:
        return event
    return f"{event} flow_id={flow_id}"


def _write_json_line(output_file: TextIO, payload: dict[str, object]) -> None:
    output_file.write(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    output_file.write("\n")


def _safe_filename_component(value: str) -> str:
    cleaned = _SAFE_FILENAME_PATTERN.sub("_", value).strip("._")
    return cleaned or "unnamed"


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in _TRUE_ENV_VALUES:
        return True
    if lowered in _FALSE_ENV_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _env_positive_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


__all__ = [
    "AFDCorrelationTraceConfig",
    "AFDCorrelationTraceRecorder",
    "AFDTraceIdentity",
    "afd_correlation_trace_config",
    "create_afd_correlation_trace_recorder",
]
