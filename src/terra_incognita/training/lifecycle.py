"""The operational lifecycle spine — timing, exit reason, and the ``training.run`` wide event.

This is the *operational* counterpart to MLflow (PLAN §6b, observability.md "Boundary with
MLflow"). It answers "did the run succeed, how long did it take, did spot get yanked?" — and
shares the join keys (``git_sha``, ``camtrap.dataset.version``, ``camtrap.model.version``)
with MLflow, but carries **no ML metrics**.

Two pieces, both pure so they're unit-tested in lean CI:

  - :func:`track_run` — a context manager that times the run and records its
    :class:`~terra_incognita.obs.events.ExitReason` (``completed`` on clean exit, ``error``
    if the body raises — then re-raises). The trainer fills in ``s3_bytes`` / ``model_version``
    on the yielded tracker as those facts become known.
  - :func:`build_training_run_event` — maps a tracker + settings into the typed
    :class:`~terra_incognita.obs.events.TrainingRunEvent`, which is then emitted as one fat
    root span via :func:`terra_incognita.obs.emit_event`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from terra_incognita.config import Settings
from terra_incognita.obs.events import Device, ExitReason, TrainingRunEvent

__all__ = ["RunTracker", "build_training_run_event", "track_run"]


@dataclass
class RunTracker:
    """Mutable operational record of one training run — the inputs to the wide event.

    ``exit_reason`` / ``duration_ms`` are set by :func:`track_run`; ``s3_bytes`` (bytes pulled
    materializing the dataset) and ``model_version`` (the registered version) are filled in by
    the trainer as it learns them. ``spot_interrupted`` is here for the deferred GPU-spot path
    (PLAN §9) — a signal handler would flip it and set ``exit_reason`` accordingly.
    """

    exit_reason: ExitReason = ExitReason.completed
    duration_ms: float = 0.0
    s3_bytes: int | None = None
    model_version: str | None = None
    spot_interrupted: bool = False


@contextmanager
def track_run() -> Iterator[RunTracker]:
    """Time the run body and record its exit reason; always re-raises on failure.

    On a clean exit the tracker keeps ``completed``; if the body raises, the reason becomes
    ``error`` and the exception propagates (we never swallow it — the wide event records that
    the run failed, then the failure still surfaces). ``duration_ms`` is set in ``finally`` so
    it's populated on both paths. Uses ``time.monotonic`` — wall-clock latency, not an MLflow
    metric (boundary discipline).
    """
    tracker = RunTracker()
    start = time.monotonic()
    try:
        yield tracker
    except BaseException:
        if not tracker.spot_interrupted:
            tracker.exit_reason = ExitReason.error
        raise
    finally:
        tracker.duration_ms = (time.monotonic() - start) * 1000.0


def build_training_run_event(
    settings: Settings,
    tracker: RunTracker,
    *,
    dataset_version: str,
    device: Device,
) -> TrainingRunEvent:
    """Map runtime settings + the run's operational record into the ``training.run`` event.

    The shared join key ``dataset_version`` (what data) is passed explicitly — a real run
    must have a pinned version, so it's never sourced from a possibly-``None`` config here.
    ``device`` is the *resolved* device (``auto`` already collapsed to mps/cuda/cpu by
    :func:`terra_incognita.training.device.resolve_device`). Operational-only: ML metrics are
    deliberately absent — they live in MLflow (observability.md).
    """
    return TrainingRunEvent(
        environment=settings.environment,
        service_name=settings.service_name,
        git_sha=settings.git_sha,
        dataset_version=dataset_version,
        device=device,
        instance_type=settings.instance_type,
        exit_reason=tracker.exit_reason,
        duration_ms=tracker.duration_ms,
        s3_bytes=tracker.s3_bytes,
        model_version=tracker.model_version,
    )
