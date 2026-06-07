"""Dataset-registration tag tests — the `datasets`-experiment convention (issue #4).

The dashboard discovers dataset versions via ``search_runs(experiment_names=["datasets"])``
and reads tags (dataset-conventions.md). These tests pin the *pure* half of that contract —
the S3 key layout and the exact tag dict — so a missing/mistyped tag fails in lean CI rather
than only against a running stack. The S3-upload + MLflow-run I/O is exercised separately by
``scripts/dataset_smoke.py``.
"""

from __future__ import annotations

import json

from terra_incognita.data.coco_to_yolo import CategoryIndex, CocoCategory
from terra_incognita.data.registration import (
    COCO_ARTIFACT_FILENAME,
    DatasetVersion,
    build_dataset_tags,
    coco_annotation_key,
    dataset_s3_prefix,
    dataset_s3_uri,
)
from terra_incognita.data.subset import SamplingConfig

# The contract's required tag set (dataset-conventions.md).
REQUIRED_TAGS = {
    "dataset_name",
    "version",
    "s3_uri",
    "coco_annotation_key",
    "sampling_config_json",
    "seed",
    "class_map_json",
    "num_images",
    "num_annotations",
}


def _dataset_version() -> DatasetVersion:
    # Sparse, out-of-order category ids → exercises the contiguous index remap (3/7/12).
    category_index = CategoryIndex.from_categories(
        [
            CocoCategory(id=12, name="bobcat"),
            CocoCategory(id=3, name="raccoon"),
            CocoCategory(id=7, name="coyote"),
        ]
    )
    return DatasetVersion(
        name="cct-subset",
        version="v3",
        sampling_config=SamplingConfig(seed=42),
        category_index=category_index,
        num_images=5,
        num_annotations=8,
    )


# ---------------------------------------------------------------------------
# S3 key layout — pure functions of bucket/name/version.
# ---------------------------------------------------------------------------
def test_s3_path_derivation():
    assert dataset_s3_prefix("cct-subset", "v3") == "datasets/cct-subset/v3/"
    assert (
        dataset_s3_uri("my-bucket", "cct-subset", "v3") == "s3://my-bucket/datasets/cct-subset/v3/"
    )
    assert (
        coco_annotation_key("cct-subset", "v3")
        == f"datasets/cct-subset/v3/{COCO_ARTIFACT_FILENAME}"
    )


# ---------------------------------------------------------------------------
# The tag dict — all required tags, all stringified, well-formed.
# ---------------------------------------------------------------------------
def test_build_dataset_tags_has_all_required_tags_as_strings():
    dv = _dataset_version()
    s3_uri = dataset_s3_uri("my-bucket", dv.name, dv.version)
    coco_key = coco_annotation_key(dv.name, dv.version)
    tags = build_dataset_tags(dv, s3_uri=s3_uri, coco_annotation_key=coco_key)

    assert set(tags) == REQUIRED_TAGS
    assert all(isinstance(value, str) for value in tags.values())  # MLflow tags are strings


def test_build_dataset_tags_values_are_well_formed():
    dv = _dataset_version()
    s3_uri = dataset_s3_uri("my-bucket", dv.name, dv.version)
    coco_key = coco_annotation_key(dv.name, dv.version)
    tags = build_dataset_tags(dv, s3_uri=s3_uri, coco_annotation_key=coco_key)

    assert tags["dataset_name"] == "cct-subset"
    assert tags["version"] == "v3"
    assert tags["s3_uri"].startswith("s3://") and tags["s3_uri"].endswith("/")
    assert tags["coco_annotation_key"] == "datasets/cct-subset/v3/annotations.json"

    # Stat tags are stringified ints, and seed mirrors the sampling config.
    assert int(tags["num_images"]) == 5
    assert int(tags["num_annotations"]) == 8
    assert int(tags["seed"]) == dv.sampling_config.seed

    # sampling_config_json round-trips to the config it was built from.
    assert json.loads(tags["sampling_config_json"]) == dv.sampling_config.model_dump()

    # class_map_json is the index→name mirror (string keys; sorted by category_id).
    assert json.loads(tags["class_map_json"]) == {"0": "raccoon", "1": "coyote", "2": "bobcat"}
