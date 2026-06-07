"""Dataset-version registration helpers — the pure half of the `datasets`-experiment convention.

A dataset version is registered as **one run in a dedicated MLflow experiment named
``datasets``**, discovered by the dashboard via ``search_runs(experiment_names=["datasets"])``
and read through tags (``.plans/contracts/dataset-conventions.md``). This module owns the
*pure* parts of that convention — the S3 key layout and the exact tag dict — so they can be
unit-tested in the lean CI sync. The actual S3 upload + MLflow run live in
``scripts/dataset_smoke.py`` (the heavy ``ml`` extra: ``boto3`` + ``mlflow``), exactly as
slice 3's artifact round-trip lives in ``scripts/stack_smoke.py``.

**Why a string-keyed tag dict and not the typed objects directly:** MLflow tag *values are
strings*. Building the dict here (once, tested) means the smoke script just calls
``mlflow.set_tags(build_dataset_tags(...))`` — no per-tag stringification scattered across
I/O code where a missing or mistyped tag would only surface against a running stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from terra_incognita.data.coco_to_yolo import CategoryIndex
from terra_incognita.data.subset import SamplingConfig

__all__ = [
    "COCO_ARTIFACT_FILENAME",
    "DATASETS_EXPERIMENT",
    "DatasetVersion",
    "build_dataset_tags",
    "coco_annotation_key",
    "dataset_s3_prefix",
    "dataset_s3_uri",
]

# The dedicated experiment every dataset version is registered under (the convention's
# linchpin — the dashboard searches exactly this name).
DATASETS_EXPERIMENT = "datasets"

# The COCO file's name within a version's S3 prefix; also its run-artifact name.
COCO_ARTIFACT_FILENAME = "annotations.json"


def dataset_s3_prefix(dataset_name: str, version: str) -> str:
    """The key prefix (no bucket) holding a version's COCO file + images, trailing slash kept."""
    return f"datasets/{dataset_name}/{version}/"


def dataset_s3_uri(bucket: str, dataset_name: str, version: str) -> str:
    """The ``s3://`` URI of a version's prefix — the ``s3_uri`` tag the dashboard reads."""
    return f"s3://{bucket}/{dataset_s3_prefix(dataset_name, version)}"


def coco_annotation_key(dataset_name: str, version: str) -> str:
    """The exact S3 key of the COCO JSON in a version's prefix — the ``coco_annotation_key`` tag."""
    return f"{dataset_s3_prefix(dataset_name, version)}{COCO_ARTIFACT_FILENAME}"


@dataclass(frozen=True)
class DatasetVersion:
    """Everything needed to register one dataset version (the inputs to the tag dict)."""

    name: str
    version: str
    sampling_config: SamplingConfig
    category_index: CategoryIndex
    num_images: int
    num_annotations: int


def build_dataset_tags(
    dataset: DatasetVersion, *, s3_uri: str, coco_annotation_key: str
) -> dict[str, str]:
    """The full ``datasets``-convention tag dict (all values stringified — MLflow tags are strings).

    Carries exactly the contract's required tags. ``class_map_json`` is a **mirror only**:
    the COCO file is the authoritative class list; this is the convenience index→name map
    for quick listing. ``s3_uri`` / ``coco_annotation_key`` are passed in (derived once by
    the caller from bucket+name+version) so the tag and the actual upload location can't drift.
    """
    class_map = {str(index): name for index, name in dataset.category_index.names.items()}
    return {
        "dataset_name": dataset.name,
        "version": dataset.version,
        "s3_uri": s3_uri,
        "coco_annotation_key": coco_annotation_key,
        "sampling_config_json": dataset.sampling_config.model_dump_json(),
        "seed": str(dataset.sampling_config.seed),
        "class_map_json": json.dumps(class_map),
        "num_images": str(dataset.num_images),
        "num_annotations": str(dataset.num_annotations),
    }
