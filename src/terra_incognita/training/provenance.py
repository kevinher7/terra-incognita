"""MLflow provenance — the *custom* half of the hybrid logging (PLAN §6, model-registry.md).

Ultralytics' built-in MLflow autolog covers params/metrics/mAP/per-class AP. It does **not**
know about *our* provenance — which device a run used, which box, which git SHA, which dataset
version, or the architecture string the registry contract requires. This module owns that
custom tag set as **pure, stringified data** (MLflow tag values are strings), so:

  - the exact keys/values are unit-tested in lean CI (no running MLflow needed), and
  - the heavy training script just calls ``mlflow.set_tags(build_provenance_tags(...))`` —
    no per-tag stringification scattered through I/O code (the same rationale as
    :mod:`terra_incognita.data.registration`'s ``build_dataset_tags``).

**Boundary discipline (observability.md).** These are provenance/identity tags only — they
are the *shared join keys* between MLflow and the ``training.run`` wide event (``git_sha``,
``device``, ``instance_type``, ``dataset_version``). ML metrics live in MLflow via autolog;
operational latency lives in the wide event. Neither is duplicated here.
"""

from __future__ import annotations

from pathlib import Path

from terra_incognita.obs.events import Device

__all__ = [
    "ARCHITECTURE_TAG",
    "CHAMPION_ALIAS",
    "REGISTERED_MODEL_NAME",
    "architecture_from_arch",
    "build_provenance_tags",
]

# The registered-model name the registry/serving contract targets as
# ``models:/<name>@champion`` (model-registry.md). A stable logical name; new training runs
# add new *versions* under it, and ``@champion`` moves to the chosen one.
REGISTERED_MODEL_NAME = "cct-detector"

# The promotion alias the dashboard + serving resolve (model-registry.md — alias is the
# source of truth, not a stage/tag).
CHAMPION_ALIAS = "champion"

# The registry-required metadata key (model-registry.md "Required model metadata"): the
# architecture string, logged as both a tag (for filtering) and a param (autolog-adjacent).
ARCHITECTURE_TAG = "architecture"


def architecture_from_arch(model_arch: str) -> str:
    """Derive the ``architecture`` string (e.g. ``"yolov8n"``) from the configured arch.

    ``ExperimentConfig.model_arch`` is the Ultralytics weight/spec name (``"yolov8n.pt"``,
    ``"yolo11n.yaml"``); the registry contract wants the bare architecture (``"yolov8n"``).
    ``Path.stem`` strips the one trailing suffix, which is exactly that.
    """
    return Path(model_arch).stem


def build_provenance_tags(
    *,
    git_sha: str,
    device: Device,
    instance_type: str,
    dataset_version: str,
    architecture: str,
) -> dict[str, str]:
    """The custom MLflow provenance tag dict (all values strings — MLflow tags are strings).

    Carries exactly the run's identity/provenance: ``git_sha`` (join key to the wide event,
    the deployed artifact, and the dataset run), the resolved ``device`` + ``instance_type``
    (local-mps vs g4dn.xlarge — so GPU and laptop runs are distinguishable), the pinned
    ``dataset_version`` (shared key with the ``datasets`` experiment), and the registry's
    required ``architecture``. **No ML metrics** — autolog owns those.
    """
    return {
        "git_sha": git_sha,
        "device": device.value,
        "instance_type": instance_type,
        "dataset_version": dataset_version,
        ARCHITECTURE_TAG: architecture,
    }
