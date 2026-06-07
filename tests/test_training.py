"""Unit tests for the pure training logic (slice 5): device resolution, provenance tags,
the run-lifecycle tracker, and the ``training.run`` wide-event builder.

These are the lean, CI-gated half of the training slice — they need no torch/ultralytics/
mlflow and no running stack. The heavy 1-epoch smoke (``scripts/train_smoke.py``) proves the
end-to-end wiring against the stack and is *not* run in CI (mirrors the dataset/stack smokes).
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from terra_incognita.config import Device as ConfigDevice
from terra_incognita.config import Environment, Settings
from terra_incognita.obs import emit_event
from terra_incognita.obs.events import Device, ExitReason
from terra_incognita.obs.registry import load_registry
from terra_incognita.training import (
    ARCHITECTURE_TAG,
    architecture_from_arch,
    build_provenance_tags,
    build_training_run_event,
    resolve_device,
    track_run,
    ultralytics_device,
)

# ML-metric attribute keys that must NEVER appear on the operational wide event nor in the
# provenance tags (observability.md "Boundary with MLflow" — they live in MLflow only).
_ML_METRIC_KEYS = ("mAP", "mAP50", "metrics/mAP50-95(B)", "precision", "recall", "loss")


def _settings(device: ConfigDevice = ConfigDevice.mps) -> Settings:
    # Hermetic: no env dependence (runtime/provenance only — hyperparameters are elsewhere).
    return Settings(
        environment=Environment.local,
        service_name="terra-incognita-training",
        git_sha="abc1234",
        device=device,
        instance_type="local-mps",
    )


# --- device resolution ------------------------------------------------------
@pytest.mark.parametrize(
    ("requested", "has_cuda", "has_mps", "expected"),
    [
        # `auto` picks the best accelerator present, in CUDA > MPS > CPU order.
        (ConfigDevice.auto, True, True, Device.cuda),
        (ConfigDevice.auto, True, False, Device.cuda),
        (ConfigDevice.auto, False, True, Device.mps),
        (ConfigDevice.auto, False, False, Device.cpu),
        # An explicit request is honored as-is, regardless of probed capabilities.
        (ConfigDevice.cuda, False, False, Device.cuda),
        (ConfigDevice.mps, False, False, Device.mps),
        (ConfigDevice.cpu, True, True, Device.cpu),
    ],
)
def test_resolve_device(requested: ConfigDevice, has_cuda: bool, has_mps: bool, expected: Device):
    assert resolve_device(requested, has_cuda=has_cuda, has_mps=has_mps) == expected


def test_ultralytics_device_mapping():
    # cpu/mps pass through as strings; cuda becomes the first-GPU index Ultralytics expects.
    assert ultralytics_device(Device.cpu) == "cpu"
    assert ultralytics_device(Device.mps) == "mps"
    assert ultralytics_device(Device.cuda) == 0


# --- provenance tags --------------------------------------------------------
def test_architecture_from_arch():
    assert architecture_from_arch("yolov8n.pt") == "yolov8n"
    assert architecture_from_arch("yolo11n.yaml") == "yolo11n"
    assert architecture_from_arch("yolov8n") == "yolov8n"


def test_build_provenance_tags_carries_required_metadata():
    tags = build_provenance_tags(
        git_sha="abc1234",
        device=Device.mps,
        instance_type="local-mps",
        dataset_version="ds-2026-06-07",
        architecture="yolov8n",
    )
    # The registry-required `architecture` (model-registry.md) plus the shared join keys.
    assert tags[ARCHITECTURE_TAG] == "yolov8n"
    assert tags["git_sha"] == "abc1234"
    assert tags["device"] == "mps"
    assert tags["instance_type"] == "local-mps"
    assert tags["dataset_version"] == "ds-2026-06-07"
    # All values stringified (MLflow tags are strings).
    assert all(isinstance(value, str) for value in tags.values())
    # Boundary discipline: no ML metrics smuggled into provenance.
    assert not any(key in tags for key in _ML_METRIC_KEYS)


# --- run-lifecycle tracker --------------------------------------------------
def test_track_run_completed_on_clean_exit():
    with track_run() as tracker:
        tracker.s3_bytes = 123
    assert tracker.exit_reason is ExitReason.completed
    assert tracker.duration_ms >= 0.0
    assert tracker.s3_bytes == 123


def test_track_run_records_error_and_reraises():
    with pytest.raises(RuntimeError), track_run() as tracker:
        raise RuntimeError("boom")
    # The exception propagates (never swallowed) *and* the failure is recorded for the event.
    assert tracker.exit_reason is ExitReason.error
    assert tracker.duration_ms >= 0.0


# --- training.run wide event (the acceptance schema test) -------------------
def test_build_training_run_event_emits_required_fields(
    span_exporter: InMemorySpanExporter, tracer: Tracer
):
    with track_run() as tracker:
        tracker.s3_bytes = 4096
        tracker.model_version = "3"
    device = resolve_device(_settings().device, has_cuda=False, has_mps=True)
    event = build_training_run_event(
        _settings(), tracker, dataset_version="ds-2026-06-07", device=device
    )
    emit_event(event, tracer=tracer, environment="local")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "training.run"
    attrs = dict(span.attributes or {})

    # Every required field per the registry (canonical settable + training.run domain).
    registry = load_registry()
    required = registry.settable_required_canonical() | set(registry.required_for("training.run"))
    for field in required:
        assert field in attrs, f"missing required field {field!r}"

    assert attrs["camtrap.device"] == "mps"
    assert attrs["camtrap.exit_reason"] == "completed"
    assert attrs["camtrap.dataset.version"] == "ds-2026-06-07"
    assert attrs["camtrap.s3.bytes"] == 4096
    assert attrs["camtrap.model.version"] == "3"

    # Boundary discipline: the operational event carries NO ML metrics.
    assert not any(key in attrs for key in _ML_METRIC_KEYS)


def test_build_training_run_event_records_error_exit():
    # A failed run still produces a well-formed event — with exit_reason=error.
    try:
        with track_run() as tracker:
            raise ValueError("train failed")
    except ValueError:
        pass
    event = build_training_run_event(_settings(), tracker, dataset_version="ds", device=Device.cpu)
    # WideEvent uses `use_enum_values=True`, so the stored value is the enum's string.
    assert event.exit_reason == ExitReason.error
