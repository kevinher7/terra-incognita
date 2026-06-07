"""Typed OpenTelemetry wide-event helper (observability contract).

The discipline (observability.md): **one fat root span per unit of work**, carrying
a wide set of high-cardinality attributes. Domain fields can only be set through the
typed Pydantic models in :mod:`terra_incognita.obs.events`, whose field names are
checked against the authoritative registry (.plans/contracts/observability.attributes.yaml)
by :mod:`terra_incognita.obs.registry` — so a wrong or missing field is hard to write
and impossible to merge (the CI guard test fails).
"""

from terra_incognita.obs.emit import EmitResult, configure_tracing, emit_event
from terra_incognita.obs.events import (
    Device,
    Environment,
    ExitReason,
    TrainingRunEvent,
    WideEvent,
)
from terra_incognita.obs.registry import Registry, load_registry

__all__ = [
    "Device",
    "EmitResult",
    "Environment",
    "ExitReason",
    "Registry",
    "TrainingRunEvent",
    "WideEvent",
    "configure_tracing",
    "emit_event",
    "load_registry",
]
