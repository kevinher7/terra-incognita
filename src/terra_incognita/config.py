"""Typed, env-driven configuration — nothing hardcoded (PLAN §4/§9).

Local/GPU parity is "free" precisely because every endpoint, credential, path and
hyperparameter comes from the environment through this one typed object. The same
code runs against floci + local SigNoz locally and real S3 + Hetzner SigNoz in prod;
only the env values differ.

Conventions:
  - Project-specific settings use the ``TI_`` env prefix (e.g. ``TI_ENVIRONMENT``).
  - Settings that mirror a well-known third-party env var (OTel, MLflow, AWS) read
    that canonical var directly via an alias, so we never fight the upstream SDK.
"""

from __future__ import annotations

import subprocess
from enum import StrEnum
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Environment (local | prod) is both a config concept and an emitted wide-event value,
# so it has a single home in obs.events (the registry-aligned enums) and is reused here
# — no duplicate enum to drift out of sync.
from terra_incognita.obs.events import Environment

__all__ = ["Device", "Environment", "Settings", "current_git_sha"]


class Device(StrEnum):
    """Compute device. ``auto`` lets the trainer pick MPS/CUDA/CPU at runtime (PLAN §6).

    Distinct from :class:`terra_incognita.obs.events.Device` on purpose: ``auto`` is a
    runtime selection hint and is never an emitted attribute value, so the event enum
    omits it.
    """

    auto = "auto"
    mps = "mps"
    cuda = "cuda"
    cpu = "cpu"


def current_git_sha() -> str:
    """Resolve the short git SHA for provenance, or ``"unknown"`` if unavailable.

    Used as the ``git_sha`` canonical wide-event field and as the MLflow provenance
    tag — the shared join key across wide events, MLflow, and the deployed artifact.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


class Settings(BaseSettings):
    """Single typed config surface, hydrated from the environment (+ optional ``.env``)."""

    model_config = SettingsConfigDict(
        env_prefix="TI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Allow construction by field name too (not only the env aliases), so explicit
        # Settings(git_sha=...) and tests work alongside env-var hydration.
        populate_by_name=True,
    )

    # --- identity / provenance ------------------------------------------------
    environment: Environment = Environment.local
    # OTel semconv service.name for the training runtime (observability.attributes.yaml).
    service_name: str = "terra-incognita-training"
    git_sha: str = Field(
        default_factory=current_git_sha,
        validation_alias=AliasChoices("TI_GIT_SHA", "GIT_SHA", "GITHUB_SHA"),
    )

    # --- local paths ----------------------------------------------------------
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    # --- S3 (floci locally / real AWS S3 in prod) -----------------------------
    s3_bucket: str = "terra-incognita"
    s3_endpoint_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("TI_S3_ENDPOINT_URL", "MLFLOW_S3_ENDPOINT_URL"),
    )

    # --- MLflow ---------------------------------------------------------------
    mlflow_tracking_uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("TI_MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_URI"),
    )

    # --- OpenTelemetry --------------------------------------------------------
    # The whole local/prod sink delta is this endpoint (observability.md "Sinks").
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "TI_OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT"
        ),
    )
    otel_exporter_otlp_headers: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "TI_OTEL_EXPORTER_OTLP_HEADERS", "OTEL_EXPORTER_OTLP_HEADERS"
        ),
    )

    # --- runtime / provenance -------------------------------------------------
    # These are *environment* concerns, not experiment definition: `device` is whatever the
    # current machine offers (auto -> mps/cuda/cpu) and `instance_type` is where it ran.
    # The experiment hyperparameters (epochs, imgsz, batch, seed, model_arch,
    # dataset_version) deliberately live in a versioned config file instead — see
    # `terra_incognita.experiment.ExperimentConfig` and `configs/*.yaml`.
    device: Device = Device.auto
    instance_type: str = "local-mps"  # provenance tag: local-mps | g4dn.xlarge | ...

    # --- observability registry -----------------------------------------------
    # Override only for tests/odd layouts; defaults to the vendored .plans/ mirror.
    obs_registry_path: Path | None = None
