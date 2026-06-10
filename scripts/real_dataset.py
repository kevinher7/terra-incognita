"""Real Caltech Camera Traps dataset build (slice 8): LILA → subset → S3 → `datasets` run.

The on-demand, **one-time, not-in-CI** counterpart to ``scripts/dataset_smoke.py``: where the
smoke samples a *synthetic fixture*, this points the **same** proven pipeline at the real
`Caltech Camera Traps <https://lila.science/datasets/caltech-camera-traps>`_ set. It is heavy
(a ~38 MB annotation download + a few GB of images) so — like the other ``scripts/*`` — it
needs the running stack (``just up``) and the ``ml`` extra (``just sync-ml``); CI proves the
*pure* logic it relies on (``clean_bbox_coco``, the sampler, the tag builder) with lean tests.

What it does (PLAN §5; dataset-conventions.md), all driven from the env via ``Settings``:
  1. **Download** ``caltech_bboxes_20200316.json`` from LILA into the gitignored data dir
     (cached — re-runs skip the fetch).
  2. **Clean** it (:func:`~terra_incognita.data.clean_bbox_coco`): drop the bbox-less "empty"
     markers so empty images become zero-annotation images — the sampler's "empty" case.
  3. **Sample** the seeded, stratified, location-split subset (the real
     :func:`~terra_incognita.data.sample_subset`) → ~5K image fixed list, ~25% empty.
  4. **Download only the selected images** per-image from LILA's unzipped folder (threaded,
     cached) — never the 6 GB / 105 GB archives.
  5. **Upload** the subset COCO + images to S3 (floci) under the version prefix.
  6. **Register** one run in the ``datasets`` experiment with the contract's tags + the COCO
     as a run artifact, and verify it is discoverable.

Then pin the printed ``version`` into ``configs/cct_real.yaml`` and run ``just real-train``.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import mlflow
from botocore.client import BaseClient, Config

# `python scripts/x.py` puts only scripts/ on sys.path; add the repo root so anything under
# tests/ would import. terra_incognita itself is installed (editable), so it needs no help.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from terra_incognita.config import Settings  # noqa: E402
from terra_incognita.data import (  # noqa: E402
    CCT_BBOX_FILENAME,
    CCT_BBOX_URL,
    CategoryIndex,
    CocoDataset,
    DatasetVersion,
    SamplingConfig,
    build_dataset_tags,
    clean_bbox_coco,
    coco_annotation_key,
    dataset_s3_prefix,
    dataset_s3_uri,
    image_url,
    sample_subset,
    write_subset_coco,
)
from terra_incognita.data.registration import DATASETS_EXPERIMENT  # noqa: E402

# The logical dataset name + version this run registers. Bump the version for a re-sample with
# different sampling knobs; the dashboard discovers it by (dataset_name, version).
_DATASET_NAME = "cct-subset"
_DATASET_VERSION = "v1"

# The sampling policy for the real subset (CONTEXT §4). `min_per_class` is the **subset-size
# lever**: the sampler keeps up to this many images per class (all of a rarer class), so a
# higher cap = a bigger subset. ~300 lands the subset in the deliverable's ~5-10K band while
# down-sampling the long-tail head and keeping rares (badger/fox) whole; empties are reduced
# from ~70% of source to ~25%, and val holds out ~20% of camera locations (disjoint).
_SAMPLING = SamplingConfig(
    min_per_class=300,
    target_empty_ratio=0.25,
    val_location_fraction=0.2,
    seed=42,
)

# Parallel image pulls. The unzipped LILA folder serves each image individually; a modest pool
# saturates a home connection without hammering the host.
_DOWNLOAD_WORKERS = 16
_DOWNLOAD_RETRIES = 3


def _s3_client(settings: Settings) -> BaseClient:
    """A boto3 S3 client pointed at floci (or real S3) with path-style addressing for floci."""
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        config=Config(s3={"addressing_style": "path"}),
    )


def _download(url: str, dest: Path, *, retries: int = _DOWNLOAD_RETRIES) -> None:
    """Download ``url`` → ``dest`` (atomic via a ``.part`` rename), retrying transient errors."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for _attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                tmp.write_bytes(response.read())
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url} after {retries} attempts: {last_error}")


def _fetch_annotations(cache_dir: Path) -> Path:
    """Download the LILA bbox annotation JSON into ``cache_dir`` (cached across runs)."""
    raw_path = cache_dir / CCT_BBOX_FILENAME
    if raw_path.exists():
        print(f"annotations cached     = {raw_path} ({raw_path.stat().st_size / 1e6:.1f} MB)")
        return raw_path
    print(f"downloading annotations from {CCT_BBOX_URL} …")
    _download(CCT_BBOX_URL, raw_path)
    print(f"annotations downloaded = {raw_path} ({raw_path.stat().st_size / 1e6:.1f} MB)")
    return raw_path


def _download_images(file_names: list[str], images_dir: Path) -> int:
    """Pull each selected image from LILA into ``images_dir`` (threaded, cached); return bytes.

    Skips files already on disk so a re-run (or a resume after an interrupt) is cheap. The
    bytes downloaded are surfaced for the operational record but are *not* the registered
    stat (the dashboard cares about image/annotation counts, not transfer size).
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    todo = [name for name in file_names if not (images_dir / name).exists()]
    cached = len(file_names) - len(todo)
    print(f"images: {cached} cached, downloading {len(todo)} …")

    done = 0
    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(_download, image_url(name), images_dir / name): name for name in todo
        }
        for future in as_completed(futures):
            future.result()  # re-raise any download failure loudly
            done += 1
            if done % 250 == 0 or done == len(todo):
                print(f"  downloaded {done}/{len(todo)}")
    return sum((images_dir / name).stat().st_size for name in file_names)


def _preflight(s3: BaseClient, settings: Settings) -> None:
    """Fail fast (before the multi-GB download) if S3 / MLflow aren't reachable + credentialed.

    The heavy download runs first, so without this a missing ``.env`` cred or a stack that isn't
    ``just up`` only surfaces *after* minutes of transfer (and at the very end for MLflow). A
    cheap ``head_bucket`` + a tracking ping turn those late failures into one immediate, actionable
    message. We catch broadly on purpose — any setup problem (no creds, floci down, bucket missing,
    tracking server down) should produce the same "fix your stack/.env" guidance, not a stack trace.
    """
    try:
        s3.head_bucket(Bucket=settings.s3_bucket)
    except Exception as error:
        raise SystemExit(
            f"S3 pre-flight failed: cannot reach bucket {settings.s3_bucket!r} at "
            f"{settings.s3_endpoint_url} ({type(error).__name__}: {error}).\n"
            "Fix: run `just up` (floci :4566) and ensure `.env` carries the floci AWS_* creds + "
            "MLFLOW_S3_ENDPOINT_URL — copy them from .env.example."
        ) from error
    try:
        mlflow.MlflowClient().search_experiments(max_results=1)
    except Exception as error:
        raise SystemExit(
            f"MLflow pre-flight failed: cannot reach the tracking server at "
            f"{settings.mlflow_tracking_uri} ({type(error).__name__}: {error}).\n"
            "Fix: run `just up` (mlflow :5000) and ensure `.env` sets MLFLOW_TRACKING_URI "
            "— see .env.example."
        ) from error


def run() -> bool:
    """Build → upload → register the real subset; return whether it registered + verified."""
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    s3 = _s3_client(settings)

    cache_dir = settings.data_dir / "lila"
    images_dir = cache_dir / "cct_images"

    # Fail fast on a missing `.env` / down stack BEFORE the heavy download (not after it).
    _preflight(s3, settings)

    # 1-2. download + clean the real annotation file (empties -> zero-annotation images).
    raw_path = _fetch_annotations(cache_dir)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    cleaned = clean_bbox_coco(raw)
    cleaned_path = cache_dir / "cct_bbox_clean.json"
    cleaned_path.write_text(json.dumps(cleaned) + "\n", encoding="utf-8")
    dataset = CocoDataset.from_path(cleaned_path)
    print(
        f"source                 = {len(raw.get('annotations', []))} raw annotations → "
        f"{len(dataset.annotations)} boxed, {len(dataset.images)} images, "
        f"{len(dataset.categories)} categories"
    )

    # 3. seeded, stratified, location-split subset (the real sampler, same as the fixture path).
    subset = sample_subset(dataset, _SAMPLING)
    subset_coco = write_subset_coco(
        cleaned_path, list(subset.image_ids), cache_dir / "subset" / "annotations.json"
    )
    category_index = CategoryIndex.from_categories(dataset.categories)

    # 4. download only the selected subset's images (per-image, threaded, cached).
    file_name_by_id = {img.id: img.file_name for img in dataset.images}
    file_names = [file_name_by_id[iid] for iid in subset.image_ids]
    bytes_pulled = _download_images(file_names, images_dir)

    # 5. upload the subset COCO + its images to S3 under the version prefix.
    prefix = dataset_s3_prefix(_DATASET_NAME, _DATASET_VERSION)
    s3_uri = dataset_s3_uri(settings.s3_bucket, _DATASET_NAME, _DATASET_VERSION)
    coco_key = coco_annotation_key(_DATASET_NAME, _DATASET_VERSION)
    s3.upload_file(str(subset_coco), settings.s3_bucket, coco_key)
    for name in file_names:
        s3.upload_file(str(images_dir / name), settings.s3_bucket, f"{prefix}images/{name}")

    # 6. register one run in the `datasets` experiment with the contract's tags + COCO artifact.
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
    with mlflow.start_run() as run_ctx:
        run_id = run_ctx.info.run_id
        mlflow.set_tags(tags)
        mlflow.log_artifact(str(subset_coco))

    # acceptance: the version is discoverable via the convention with well-formed tags.
    runs = mlflow.search_runs(experiment_names=[DATASETS_EXPERIMENT])
    found = runs[runs["run_id"] == run_id]
    discoverable = not found.empty
    tags_ok = discoverable and all(
        found.iloc[0].get(f"tags.{key}") == value for key, value in tags.items()
    )

    ok = bool(discoverable and tags_ok)
    print(f"tracking_uri           = {settings.mlflow_tracking_uri}")
    print(f"s3_endpoint            = {settings.s3_endpoint_url}")
    print(f"run_id                 = {run_id}")
    print(f"s3_uri                 = {s3_uri}")
    print(f"coco_key               = {coco_key}")
    print(
        f"subset                 = {subset.num_images} images "
        f"({subset.num_empty_images} empty, ~{subset.num_empty_images / subset.num_images:.0%}) / "
        f"{subset.num_annotations} annotations / {bytes_pulled / 1e6:.0f} MB images"
    )
    print(f"  per-class images     = {subset.per_class_counts}")
    print(
        f"  train/val locations  = {len(subset.train_locations)}/{len(subset.val_locations)} "
        f"cameras (disjoint)"
    )
    print(f"discoverable           = {discoverable}")
    print(f"tags_ok                = {tags_ok}")
    return ok


def main() -> int:
    ok = run()
    if ok:
        print(
            f"\nDONE: registered '{_DATASET_NAME}' {_DATASET_VERSION} in `datasets`. "
            f'Pin `dataset_version: "{_DATASET_VERSION}"` in configs/cct_real.yaml, then '
            "`just real-train`."
        )
        return 0
    print("\nFAILED: dataset registration did not verify (see output above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
