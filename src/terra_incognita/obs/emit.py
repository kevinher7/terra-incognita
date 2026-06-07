"""Configure tracing and emit typed wide events.

One call to :func:`emit_event` produces one fat root span — the canonical wide event
for a unit of work. Required-field presence is validated against the registry:
**loud in dev (raise), never fatal in prod (warn + emit anyway)** — telemetry must
never take down the thing it observes (observability.md "Schema enforcement").
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.trace import Tracer

from terra_incognita.obs.events import WideEvent
from terra_incognita.obs.registry import load_registry

log = logging.getLogger(__name__)

_TRACER_NAME = "terra-incognita"


class UrandomIdGenerator(IdGenerator):
    """Trace/span IDs from ``os.urandom`` — immune to a seeded global ``random``.

    OTel's default ``RandomIdGenerator`` draws IDs from Python's ``random`` module. A training
    run seeds that module for reproducibility (Ultralytics ``seed=``, PLAN §6) — which makes
    the default generator **deterministic**, so every run would emit the *same* ``trace_id``
    and collapse the cross-boundary join key the observability contract is built on
    (observability.md "Trace-context propagation"). ``os.urandom`` is independent of that seed,
    so IDs stay unique no matter what the process did to the RNG.
    """

    def generate_span_id(self) -> int:
        # 8-byte span id; retry on the (astronomically unlikely) all-zero INVALID id.
        span_id = int.from_bytes(os.urandom(8), "big")
        while span_id == trace.INVALID_SPAN_ID:
            span_id = int.from_bytes(os.urandom(8), "big")
        return span_id

    def generate_trace_id(self) -> int:
        # 16-byte trace id; retry on the all-zero INVALID id.
        trace_id = int.from_bytes(os.urandom(16), "big")
        while trace_id == trace.INVALID_TRACE_ID:
            trace_id = int.from_bytes(os.urandom(16), "big")
        return trace_id


# OTLP/HTTP signal path. Settings/.env carry the OTLP *base* endpoint (the collector root —
# the single env delta the observability contract promises), but the HTTP exporter's
# `endpoint` kwarg is the *signal-specific* URL used verbatim (it does NOT append a path the
# way the OTEL_EXPORTER_OTLP_ENDPOINT env var does). So we append it ourselves.
_OTLP_TRACES_PATH = "/v1/traces"


def _otlp_traces_endpoint(base: str) -> str:
    """Build the OTLP/HTTP traces URL from a base collector endpoint (idempotent)."""
    base = base.rstrip("/")
    if base.endswith(_OTLP_TRACES_PATH):
        return base
    return base + _OTLP_TRACES_PATH


@dataclass(frozen=True)
class EmitResult:
    """Identifiers of the emitted span — log/propagate these to correlate the trace."""

    event_name: str
    trace_id: str  # 32-hex W3C trace-id
    span_id: str  # 16-hex span-id


def configure_tracing(
    service_name: str,
    *,
    otlp_endpoint: str | None = None,
    span_processor: SpanProcessor | None = None,
) -> TracerProvider:
    """Install a global :class:`TracerProvider` and return it.

    Exactly one of the exporters is wired:
      - ``span_processor`` given (tests pass an in-memory exporter), or
      - ``otlp_endpoint`` given (SigNoz locally / Hetzner in prod — the only env delta), or
      - neither: a provider with no processor (spans are created but go nowhere).
    """
    resource = Resource.create({"service.name": service_name})
    # UrandomIdGenerator (not OTel's default random-module one) so a seeded training RNG can't
    # make every run share a trace_id — see the class docstring.
    provider = TracerProvider(resource=resource, id_generator=UrandomIdGenerator())

    if span_processor is not None:
        provider.add_span_processor(span_processor)
    elif otlp_endpoint:
        # Lazy import so the OTLP exporter (and its transitive deps) is only touched
        # when actually exporting — tests and pure-CLI runs don't need it.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=_otlp_traces_endpoint(otlp_endpoint))
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    return provider


def _validate_required(event: WideEvent, event_name: str, environment: str) -> None:
    """Loud in dev, never fatal in prod — assert required fields are present."""
    registry = load_registry()
    required = registry.settable_required_canonical() | set(registry.required_for(event_name))
    present = set(event.attributes())
    missing = required - present
    if not missing:
        return

    message = f"wide event {event_name!r} missing required fields: {sorted(missing)}"
    if environment == "local":
        raise ValueError(message)
    log.warning("%s (emitting anyway — telemetry must not crash prod)", message)


def emit_event(
    event: WideEvent,
    *,
    event_name: str | None = None,
    environment: str = "local",
    tracer: Tracer | None = None,
) -> EmitResult:
    """Emit ``event`` as one root span and return its trace/span identifiers."""
    name = event_name or getattr(type(event), "EVENT_NAME", type(event).__name__)
    _validate_required(event, name, environment)

    tracer = tracer or trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as span:
        for key, value in event.attributes().items():
            span.set_attribute(key, value)
        ctx = span.get_span_context()
        result = EmitResult(
            event_name=name,
            trace_id=format(ctx.trace_id, "032x"),
            span_id=format(ctx.span_id, "016x"),
        )
    return result
