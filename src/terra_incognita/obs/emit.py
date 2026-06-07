"""Configure tracing and emit typed wide events.

One call to :func:`emit_event` produces one fat root span — the canonical wide event
for a unit of work. Required-field presence is validated against the registry:
**loud in dev (raise), never fatal in prod (warn + emit anyway)** — telemetry must
never take down the thing it observes (observability.md "Schema enforcement").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer

from terra_incognita.obs.events import WideEvent
from terra_incognita.obs.registry import load_registry

log = logging.getLogger(__name__)

_TRACER_NAME = "terra-incognita"


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
    provider = TracerProvider(resource=resource)

    if span_processor is not None:
        provider.add_span_processor(span_processor)
    elif otlp_endpoint:
        # Lazy import so the OTLP exporter (and its transitive deps) is only touched
        # when actually exporting — tests and pure-CLI runs don't need it.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))

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
