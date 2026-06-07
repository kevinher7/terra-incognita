"""Typed wide-event models — the only way to set domain fields.

Each model maps Pythonic field names to the dotted OTel attribute keys from the
registry via ``serialization_alias`` (Python identifiers can't contain dots). The
CI guard test (tests/test_registry_guard.py) asserts every alias here exists in
observability.attributes.yaml and that each event covers its required fields — so the
models and the registry cannot silently drift.

Canonical ``trace_id`` / ``span_id`` are intentionally absent: OTel supplies them
from the span context, so they live on the span, not in these payloads.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel, ConfigDict, Field


class Environment(StrEnum):
    local = "local"
    prod = "prod"


class Device(StrEnum):
    mps = "mps"
    cuda = "cuda"
    cpu = "cpu"


class ExitReason(StrEnum):
    completed = "completed"
    error = "error"
    spot_interrupted = "spot_interrupted"


class WideEvent(BaseModel):
    """Base for every root-span payload: the settable canonical fields."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        use_enum_values=True,  # enums serialize to their str value (valid OTel attr type)
    )

    environment: Environment = Field(serialization_alias="environment")
    service_name: str = Field(serialization_alias="service.name")
    git_sha: str = Field(serialization_alias="git_sha")

    def attributes(self) -> dict[str, AttributeValue]:
        """Dotted-key attribute dict for the span; ``None``-valued fields are dropped."""
        dumped = self.model_dump(by_alias=True)
        return {key: value for key, value in dumped.items() if value is not None}


class TrainingRunEvent(WideEvent):
    """The ``training.run`` operational wide event (PLAN §6b, observability.md).

    Operational lifecycle ONLY — ML metrics (mAP, loss) live in MLflow. The two share
    keys (``camtrap.dataset.version``, ``git_sha``) but never duplicate payload.
    """

    EVENT_NAME: ClassVar[str] = "training.run"

    # Required domain fields for training.run (observability.attributes.yaml).
    dataset_version: str = Field(serialization_alias="camtrap.dataset.version")
    device: Device = Field(serialization_alias="camtrap.device")
    instance_type: str = Field(serialization_alias="camtrap.instance_type")
    exit_reason: ExitReason = Field(serialization_alias="camtrap.exit_reason")

    # Optional operational fields — dropped from the span when unset.
    duration_ms: float | None = Field(default=None, serialization_alias="camtrap.duration_ms")
    s3_bytes: int | None = Field(default=None, serialization_alias="camtrap.s3.bytes")
    model_version: str | None = Field(default=None, serialization_alias="camtrap.model.version")


def serialization_aliases(model: type[BaseModel]) -> set[str]:
    """The set of dotted OTel attribute keys a model serializes to (for the guard test)."""
    return {field.serialization_alias or name for name, field in model.model_fields.items()}
