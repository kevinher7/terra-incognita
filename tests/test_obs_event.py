"""Acceptance test (issue #1): a trivial CLI command emits a wide event captured by an
in-memory OTel span exporter, with the required canonical + domain fields present."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer
from pydantic import ValidationError

from terra_incognita.cli import build_demo_training_run_event
from terra_incognita.config import Device, Environment, Settings
from terra_incognita.experiment import ExperimentConfig
from terra_incognita.obs import emit_event
from terra_incognita.obs.events import TrainingRunEvent
from terra_incognita.obs.registry import load_registry


def _settings() -> Settings:
    # Construct explicitly (no env dependence) so the test is hermetic. Note: runtime/
    # provenance only — hyperparameters now live in ExperimentConfig (see _experiment).
    return Settings(
        environment=Environment.local,
        service_name="terra-incognita-training",
        git_sha="abc1234",
        device=Device.mps,
        instance_type="local-mps",
    )


def _experiment() -> ExperimentConfig:
    # dataset_version is part of the experiment (what data), not the environment.
    return ExperimentConfig(dataset_version="ds-2026-06-07")


def test_demo_event_carries_required_fields(span_exporter: InMemorySpanExporter, tracer: Tracer):
    event = build_demo_training_run_event(_settings(), _experiment())
    result = emit_event(event, tracer=tracer, environment="local")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "training.run"
    assert result.event_name == "training.run"

    attrs = dict(span.attributes or {})

    # Settable canonical fields (registry-driven).
    registry = load_registry()
    for field in registry.settable_required_canonical():
        assert field in attrs, f"missing canonical field {field!r}"
    assert attrs["environment"] == "local"
    assert attrs["service.name"] == "terra-incognita-training"
    assert attrs["git_sha"] == "abc1234"

    # Required domain fields for training.run.
    for field in registry.required_for("training.run"):
        assert field in attrs, f"missing domain field {field!r}"
    assert attrs["camtrap.device"] == "mps"
    assert attrs["camtrap.dataset.version"] == "ds-2026-06-07"

    # Canonical trace_id / span_id are OTel-supplied (on the span, not the attributes).
    assert span.context.trace_id != 0
    assert span.context.span_id != 0
    assert result.trace_id == format(span.context.trace_id, "032x")


def test_missing_required_domain_field_is_rejected():
    # Domain fields are settable only through the typed model — building one from data
    # that omits a required field fails loudly rather than emitting a malformed event.
    payload = {
        "environment": "local",
        "service_name": "terra-incognita-training",
        "git_sha": "abc1234",
        "device": "mps",
        "instance_type": "local-mps",
        "exit_reason": "completed",
        # dataset_version intentionally omitted
    }
    with pytest.raises(ValidationError):
        TrainingRunEvent.model_validate(payload)
