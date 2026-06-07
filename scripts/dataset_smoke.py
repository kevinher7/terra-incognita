"""Dataset-pipeline smoke: prove the full `datasets`-experiment convention on fixtures.

This is the slice-4 acceptance harness — the I/O counterpart to the pure sampler/tag logic
in ``src/``. Like ``scripts/stack_smoke.py`` it is deliberately **not** a pytest/CI test:
it needs the running docker stack (``just up``) and the heavy ``ml`` extra (``just
sync-ml``), neither of which CI has. CI proves the pure logic (sampler determinism + tag
well-formedness) with lean unit tests instead.

What it exercises end to end (dataset-conventions.md):
  1. sample a stratified, location-split subset of a synthetic fixture,
  2. **upload** the subset images + COCO file directly to S3 (floci) under the version prefix,
  3. **register** one run in the ``datasets`` experiment with the contract's tags + the COCO
     logged as a run artifact,
  4. **verify** acceptance: ``mlflow.search_runs(experiment_names=["datasets"])`` returns the
     run with well-formed tags, and the COCO downloads from ``s3_uri``/``coco_annotation_key``
     and parses.

All endpoints/credentials come from the environment via ``Settings`` (loaded from ``.env``
by the ``just`` recipe's dotenv) — nothing localhost is hardcoded here.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import boto3
import mlflow
from botocore.client import BaseClient, Config

# `python scripts/x.py` puts only scripts/ on sys.path; add the repo root so the synthetic
# fixture generator (under tests/) is importable. terra_incognita itself is installed
# (editable), so it needs no help.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.synthetic import generate_synthetic_dataset  # noqa: E402

from terra_incognita.config import Settings  # noqa: E402
from terra_incognita.data import (  # noqa: E402
    CategoryIndex,
    CocoDataset,
    DatasetVersion,
    build_dataset_tags,
    coco_annotation_key,
    dataset_s3_prefix,
    dataset_s3_uri,
    sample_subset,
    write_subset_coco,
)
from terra_incognita.data.registration import DATASETS_EXPERIMENT  # noqa: E402

# A stable logical name; reruns create a fresh MLflow run (new run id) under the same name.
_DATASET_NAME = "cct-subset-synthetic"
_DATASET_VERSION = "v1"


def _s3_client(settings: Settings) -> BaseClient:
    """A boto3 S3 client pointed at floci (or real S3) with path-style addressing for floci."""
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        config=Config(s3={"addressing_style": "path"}),
    )


def run_smoke() -> bool:
    """Sample → upload → register → verify; return whether the full round-trip held."""
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    s3 = _s3_client(settings)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. fixture → stratified, location-split subset → faithful subset COCO file.
        fixture = generate_synthetic_dataset(tmp_path / "src")
        dataset = CocoDataset.from_path(fixture.coco_path)
        subset = sample_subset(dataset)
        subset_coco = write_subset_coco(
            fixture.coco_path, list(subset.image_ids), tmp_path / "subset" / "annotations.json"
        )
        category_index = CategoryIndex.from_categories(dataset.categories)

        # 2. upload subset images + COCO directly to S3 under the version prefix.
        prefix = dataset_s3_prefix(_DATASET_NAME, _DATASET_VERSION)
        s3_uri = dataset_s3_uri(settings.s3_bucket, _DATASET_NAME, _DATASET_VERSION)
        coco_key = coco_annotation_key(_DATASET_NAME, _DATASET_VERSION)
        s3.upload_file(str(subset_coco), settings.s3_bucket, coco_key)
        for image_id in subset.image_ids:
            file_name = next(img.file_name for img in dataset.images if img.id == image_id)
            s3.upload_file(
                str(fixture.images_dir / file_name),
                settings.s3_bucket,
                f"{prefix}images/{file_name}",
            )

        # 3. register one run in the `datasets` experiment with the contract's tags + COCO artifact.
        dataset_version = DatasetVersion(
            name=_DATASET_NAME,
            version=_DATASET_VERSION,
            sampling_config=subset.config,
            category_index=category_index,
            num_images=subset.num_images,
            num_annotations=subset.num_annotations,
        )
        tags = build_dataset_tags(dataset_version, s3_uri=s3_uri, coco_annotation_key=coco_key)
        mlflow.set_experiment(DATASETS_EXPERIMENT)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.set_tags(tags)
            mlflow.log_artifact(str(subset_coco))  # provenance copy → mlflow-artifacts/

        # 4a. acceptance: the version is discoverable via the convention, tags well-formed.
        runs = mlflow.search_runs(experiment_names=[DATASETS_EXPERIMENT])
        found = runs[runs["run_id"] == run_id]
        discoverable = not found.empty
        tags_ok = discoverable and all(
            found.iloc[0].get(f"tags.{key}") == value for key, value in tags.items()
        )

        # 4b. acceptance: COCO downloads from s3_uri/coco_annotation_key and parses.
        download_path = tmp_path / "roundtrip.json"
        s3.download_file(settings.s3_bucket, coco_key, str(download_path))
        parsed = CocoDataset.from_path(download_path)
        coco_ok = len(parsed.images) == subset.num_images and len(parsed.annotations) == (
            subset.num_annotations
        )

    ok = bool(discoverable and tags_ok and coco_ok)
    print(f"tracking_uri    = {settings.mlflow_tracking_uri}")
    print(f"s3_endpoint     = {settings.s3_endpoint_url}")
    print(f"run_id          = {run_id}")
    print(f"s3_uri          = {s3_uri}")
    print(f"coco_key        = {coco_key}")
    print(f"subset          = {subset.num_images} images / {subset.num_annotations} annotations")
    print(f"  per-class     = {subset.per_class_counts}")
    print(f"  train/val loc = {subset.train_locations} / {subset.val_locations}")
    print(f"discoverable    = {discoverable}")
    print(f"tags_ok         = {tags_ok}")
    print(f"coco_ok         = {coco_ok}")
    if discoverable:
        print(f"tags            = {json.dumps(tags, indent=2)}")
    return ok


def main() -> int:
    ok = run_smoke()
    if ok:
        print(
            "\nSMOKE PASS: dataset registered in `datasets`, tags well-formed, COCO round-tripped."
        )
        return 0
    print("\nSMOKE FAIL: dataset registration / COCO round-trip did not verify.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
