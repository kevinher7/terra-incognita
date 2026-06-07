"""Load the authoritative attribute registry (observability.attributes.yaml).

This is the single source of truth for wide-event field names, types, required
status, and cardinality. The typed event models import their field names from here
(indirectly, via the CI guard test that asserts the models match this file), and the
emit helper uses it to validate required-field presence.

The registry is the vendored, read-only `.plans/` mirror of terra-carta. We resolve
it relative to this file so it works regardless of the process CWD; an explicit path
(e.g. from tests or odd deployments) overrides that.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

# trace_id / span_id are canonical *and* required, but OTel supplies them from the
# span context — they are never set as attributes by emit code. Everything else in
# `canonical` is a settable attribute.
SPAN_SUPPLIED: frozenset[str] = frozenset({"trace_id", "span_id"})

# registry.py -> obs -> terra_incognita -> src -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = _REPO_ROOT / ".plans" / "contracts" / "observability.attributes.yaml"


@dataclass(frozen=True)
class Registry:
    """In-memory view of observability.attributes.yaml."""

    schema_version: int
    canonical: tuple[str, ...]
    domain: tuple[str, ...]
    required_canonical: tuple[str, ...]
    events: dict[str, tuple[str, ...]]  # event name -> required domain fields

    def all_fields(self) -> set[str]:
        """Every known field name (canonical + domain)."""
        return set(self.canonical) | set(self.domain)

    def settable_required_canonical(self) -> set[str]:
        """Required canonical fields that emit code must set (excludes OTel-supplied ones)."""
        return set(self.required_canonical) - SPAN_SUPPLIED

    def required_for(self, event_name: str) -> tuple[str, ...]:
        """Required domain fields for a unit-of-work event type, e.g. ``training.run``."""
        if event_name not in self.events:
            raise KeyError(f"unknown event {event_name!r}; known: {sorted(self.events)}")
        return self.events[event_name]


@cache
def load_registry(path: Path | str | None = None) -> Registry:
    """Parse the YAML registry into a :class:`Registry` (cached per path)."""
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))

    canonical = tuple(field["name"] for field in raw.get("canonical", []))
    required_canonical = tuple(
        field["name"] for field in raw.get("canonical", []) if field.get("required")
    )
    domain = tuple(field["name"] for field in raw.get("domain", []))
    events = {
        event["name"]: tuple(event.get("requires", []) or []) for event in raw.get("events", [])
    }

    return Registry(
        schema_version=int(raw["schema_version"]),
        canonical=canonical,
        domain=domain,
        required_canonical=required_canonical,
        events=events,
    )
