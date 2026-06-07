"""Experiment definition — the hyperparameters that make one run different from another.

Deliberately separate from :class:`terra_incognita.config.Settings`. The split is by
*concern*, decided with one litmus test:

  - "Does this differ between my laptop and the GPU box for the **same** experiment?"
    -> it's *environment* parity -> :class:`Settings` (env-driven, see ``config.py``).
  - "Does this define what makes **this** experiment different from another?"
    -> it's the *experiment* -> here, in a versioned ``configs/*.yaml`` (committed to git,
    logged to MLflow as params).

Why not env vars for these: a ``.env`` file is gitignored and single-state, so sourcing
hyperparameters from it means experiments are unversioned and you "switch" experiments by
editing ambient state — the env-var twin of the copy-pasted-script anti-pattern. A
committed YAML you point at (``mlflow run -P config=configs/baseline.yaml``) is diffable,
nameable, and reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ExperimentConfig", "load_experiment_config"]


class ExperimentConfig(BaseModel):
    """The hyperparameters that define one training experiment.

    ``extra="forbid"`` so a typo'd key in the YAML (e.g. ``epoch:`` for ``epochs:``)
    fails loudly instead of being silently ignored — a config you can't trust defeats the
    point. ``protected_namespaces=()`` lets us keep the domain term ``model_arch`` without
    Pydantic's ``model_*`` warning.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_arch: str = "yolov8n.pt"
    epochs: int = Field(default=50, gt=0)
    imgsz: int = Field(default=640, gt=0)
    batch: int = Field(default=16, gt=0)
    seed: int = 42
    # Pinned once the dataset is registered (datasets-experiment convention); part of the
    # experiment because *which data* is what makes a run reproducible.
    dataset_version: str | None = None

    def as_mlflow_params(self) -> dict[str, Any]:
        """Flat dict for a single ``mlflow.log_params`` call — the whole experiment, logged once.

        This is the low-boilerplate "log the full resolved config in one call" pattern:
        no per-field ``log_param`` and no risk of a hyperparameter silently going unlogged.
        """
        return self.model_dump()


def load_experiment_config(path: Path | None = None, **overrides: int | None) -> ExperimentConfig:
    """Load a versioned experiment YAML, applying any non-``None`` CLI overrides on top.

    The YAML (committed in ``configs/``) is the canonical, reproducible surface; the
    ``overrides`` are an ad-hoc human affordance (``--epochs 5`` for a quick local probe)
    and are intentionally *not* part of the MLproject contract. A ``None`` override means
    "unset on the CLI" and leaves the file's value untouched.
    """
    data: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"experiment config {path} must be a YAML mapping, got {type(raw).__name__}"
            )
        data = raw
    data.update({key: value for key, value in overrides.items() if value is not None})
    return ExperimentConfig(**data)
