"""Shared fixtures: an in-memory OTel exporter so tests capture wide events without SigNoz."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    """Captures every finished span for assertion (PLAN §10: in-memory exporter, no SigNoz)."""
    return InMemorySpanExporter()


@pytest.fixture
def tracer(span_exporter: InMemorySpanExporter) -> Tracer:
    """A tracer wired to the in-memory exporter via a synchronous (SimpleSpanProcessor) flush."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider.get_tracer("test")
