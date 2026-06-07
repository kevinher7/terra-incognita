"""Guard test: the typed event models cannot drift from the authoritative registry
(.plans/contracts/observability.attributes.yaml). If someone renames a field in the
YAML or the model without the other, this fails — which is the whole point of the
'domain fields settable only through typed functions' enforcement."""

from __future__ import annotations

from terra_incognita.obs.events import (
    TrainingRunEvent,
    WideEvent,
    serialization_aliases,
)
from terra_incognita.obs.registry import load_registry


def test_all_model_aliases_exist_in_registry():
    registry = load_registry()
    known = registry.all_fields()
    for model in (WideEvent, TrainingRunEvent):
        for alias in serialization_aliases(model):
            assert alias in known, f"{model.__name__} field {alias!r} not in registry"


def test_training_run_event_covers_required_fields():
    registry = load_registry()
    aliases = serialization_aliases(TrainingRunEvent)

    # Required domain fields for training.run.
    for field in registry.required_for("training.run"):
        assert field in aliases, f"training.run requires {field!r}, model is missing it"

    # Settable canonical fields (trace_id/span_id are OTel-supplied, excluded).
    for field in registry.settable_required_canonical():
        assert field in aliases, f"canonical field {field!r} missing from event model"


def test_registry_schema_version_is_pinned():
    # A bump here means the contract changed shape — force a conscious update.
    assert load_registry().schema_version == 1
