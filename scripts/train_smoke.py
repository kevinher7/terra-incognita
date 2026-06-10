"""Training smoke (slice 5): a 1-epoch device-agnostic train that produces a registered
``@champion`` model **and** emits the ``training.run`` operational wide event — end to end.

Like ``scripts/dataset_smoke.py`` / ``scripts/stack_smoke.py`` this is deliberately **not** a
pytest/CI test: it needs the running docker stack (``just up``) and the heavy ``ml`` extra
(``just sync-ml``), neither of which CI has. CI proves the *pure* training logic (device
resolution, provenance tags, the wide-event builder) with lean unit tests in
``tests/test_training.py``; this smoke proves the heavy wiring against the real stack.

What it exercises end to end (PLAN §6/§6b; model-registry.md; observability.md):
  1. **dataset** — sample a stratified subset of a synthetic fixture, upload it to S3 (floci)
     under a version prefix, and register it in the ``datasets`` experiment.
  2. **materialize from ``s3_uri``** — download the COCO + images *from S3* (no hardcoded
     path) and build the Ultralytics YOLO layout, so a local and a GPU run are identical.
  3. **device-agnostic train** — probe the device (``auto`` → mps/cuda/cpu) and 1-epoch train
     with Ultralytics' **built-in MLflow autolog** handling params/metrics/mAP.
  4. **custom hybrid logging** — reopen the autolog run to add provenance tags
     (``git_sha``/``device``/``instance_type``/``dataset_version``/``architecture``), log the
     model with a **signature**, register it, and move the **``@champion``** alias to it.
  5. **operational wide event** — emit ``training.run`` (lifecycle only: device, instance,
     exit reason, duration, s3 bytes, model version) via the typed OTel helper.
  6. **verify** — the registered version resolves via ``@champion``, carries ``architecture``,
     and the run has ML metrics; the wide event carries its required fields.

**Boundary discipline (observability.md):** ML metrics live only in MLflow (autolog); the
wide event carries only operational lifecycle — they share keys, never payload.

All endpoints/credentials come from the environment via ``Settings`` (loaded from ``.env`` by
the ``just`` recipe's dotenv) — nothing localhost is hardcoded here.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import boto3
import mlflow
from botocore.client import BaseClient, Config

# `python scripts/x.py` puts only scripts/ on sys.path; add the repo root so the synthetic
# fixture generator (under tests/) is importable. terra_incognita itself is installed
# (editable), so it needs no help.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# The real serving pyfunc (slice 6) — a sibling script, importable because scripts/ is on
# sys.path when this runs as `python scripts/train_smoke.py`. It carries the heavy ml deps
# (mlflow + ultralytics), exactly like this file.
import serving_pyfunc  # noqa: E402
from serving_pyfunc import (  # noqa: E402
    CATEGORY_MAP_ARTIFACT,
    WEIGHTS_ARTIFACT,
    CCTDetector,
    build_serving_signature,
)
from tests.fixtures.synthetic import generate_synthetic_dataset  # noqa: E402

import terra_incognita  # noqa: E402
from terra_incognita.config import Settings  # noqa: E402
from terra_incognita.data import (  # noqa: E402
    CategoryIndex,
    CocoDataset,
    DatasetVersion,
    Split,
    build_dataset_tags,
    coco_annotation_key,
    convert_coco_to_yolo,
    dataset_s3_prefix,
    dataset_s3_uri,
    sample_subset,
    split_by_fraction,
    write_subset_coco,
)
from terra_incognita.data.registration import DATASETS_EXPERIMENT  # noqa: E402
from terra_incognita.obs import configure_tracing, emit_event  # noqa: E402
from terra_incognita.obs.events import Device  # noqa: E402
from terra_incognita.training import (  # noqa: E402
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    architecture_from_arch,
    build_provenance_tags,
    build_training_run_event,
    resolve_device,
    track_run,
    ultralytics_device,
)

# A stable logical name; reruns create a fresh MLflow run / model version under the same name.
_DATASET_NAME = "cct-subset-synthetic"
_DATASET_VERSION = "v1"
_TRAINING_EXPERIMENT = "training"

# Deliberately tiny so the smoke is fast: 1 epoch, small imgsz (multiple of 32), small batch.
# These are *smoke* values, not the committed experiment in configs/baseline.yaml.
_MODEL_ARCH = "yolov8n.pt"
_EPOCHS = 1
_IMGSZ = 64
_BATCH = 4
_SEED = 42


def _s3_client(settings: Settings) -> BaseClient:
    """A boto3 S3 client pointed at floci (or real S3) with path-style addressing for floci."""
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        config=Config(s3={"addressing_style": "path"}),
    )


def detect_device(settings: Settings) -> Device:
    """Probe the real hardware and resolve the configured (possibly ``auto``) device.

    The torch probe is the *only* device-specific code, and it lives here (heavy ``ml`` extra)
    — the pure mapping is :func:`terra_incognita.training.resolve_device`, tested in CI.
    """
    import torch

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    return resolve_device(settings.device, has_cuda=has_cuda, has_mps=has_mps)


def _upload_and_register_dataset(
    s3: BaseClient, settings: Settings, fixture_dir: Path
) -> tuple[str, str, str]:
    """Sample → upload → register a dataset version; return (s3_uri, coco_key, image_prefix).

    The dataset half mirrors ``scripts/dataset_smoke.py`` so this smoke is self-contained —
    it needs a *registered* dataset to then materialize from, which is the slice-5 deliverable.
    """
    fixture = generate_synthetic_dataset(fixture_dir)
    dataset = CocoDataset.from_path(fixture.coco_path)
    subset = sample_subset(dataset)
    subset_coco = write_subset_coco(
        fixture.coco_path, list(subset.image_ids), fixture_dir / "subset" / "annotations.json"
    )
    category_index = CategoryIndex.from_categories(dataset.categories)

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
    with mlflow.start_run():
        mlflow.set_tags(tags)
        mlflow.log_artifact(str(subset_coco))
    return s3_uri, coco_key, f"{prefix}images/"


def _materialize_from_s3(
    s3: BaseClient,
    bucket: str,
    coco_key: str,
    image_prefix: str,
    out_dir: Path,
    *,
    split: Callable[[CocoDataset], dict[str, Split]] | None = None,
) -> tuple[Path, Path, int]:
    """Download the COCO + images from S3 and build the YOLO layout.

    Returns ``(data.yaml, category_map, bytes)``. This is the "materialize the dataset from its
    ``s3_uri`` into the local YOLO layout" deliverable — driven entirely by the dataset's S3
    keys, never a hardcoded local path, so a laptop run and a GPU-box run are identical. The
    ``category_map`` (the converter's index→``category_id`` artifact) is baked into the served
    model so serving can translate YOLO indices back to COCO ids (serving-io.md). ``bytes`` (sum
    of objects pulled) becomes the wide event's ``camtrap.s3.bytes`` operational field.

    ``split`` derives the train/val map from the materialized COCO (the split is a *training*
    input, not stored in S3 — dataset-conventions.md). The fixture smokes leave it ``None`` →
    the placeholder fraction split; the real run (``real_train.py``) passes the location-disjoint
    :func:`~terra_incognita.data.split_selected_by_location` so train/val cameras are disjoint.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    local_coco = out_dir / "annotations.json"
    s3.download_file(bucket, coco_key, str(local_coco))
    bytes_pulled = local_coco.stat().st_size

    dataset = CocoDataset.from_path(local_coco)
    local_images = out_dir / "images"
    local_images.mkdir(parents=True, exist_ok=True)
    for image in dataset.images:
        dest = local_images / image.file_name
        s3.download_file(bucket, f"{image_prefix}{image.file_name}", str(dest))
        bytes_pulled += dest.stat().st_size

    if split is not None:
        image_splits = split(dataset)
    else:
        image_splits = split_by_fraction(
            [img.id for img in dataset.images], val_fraction=0.3, seed=_SEED
        )
    result = convert_coco_to_yolo(dataset, local_images, out_dir / "yolo", image_splits)
    return result.data_yaml_path, result.category_map_path, bytes_pulled


def _train_and_register(
    data_yaml: Path,
    category_map: Path,
    device: Device,
    dataset_version: str,
    settings: Settings,
    *,
    model_arch: str = _MODEL_ARCH,
    epochs: int = _EPOCHS,
    imgsz: int = _IMGSZ,
    batch: int = _BATCH,
    seed: int = _SEED,
) -> tuple[str, dict[str, float]]:
    """Autolog train, then custom provenance + the real serving pyfunc + ``@champion``.

    Returns (registered version, metrics). Ultralytics' MLflow callback owns the run for
    params/metrics (the *built-in* half of the hybrid); we reopen it by id to add the *custom*
    half autolog can't do — provenance tags, the model+signature, registration, and the alias.
    The logged model is the real :class:`~serving_pyfunc.CCTDetector` (slice 6): the trained
    weights **and** the index→``category_id`` map (``category_map``, from this run's dataset) are
    baked in as artifacts, with this file + the ``terra_incognita`` package carried via
    ``code_paths`` so ``mlflow models serve`` / ``build-docker`` reconstruct it with no editable
    install and no S3 at runtime (serving-io.md).

    The ``model_arch``/``epochs``/``imgsz``/``batch``/``seed`` keyword args default to the tiny
    smoke values; the real run (``real_train.py``) passes the committed ``configs/cct_real.yaml``
    experiment so the *same* registration path produces a genuine champion (PLAN §6).
    """
    from ultralytics import YOLO
    from ultralytics import settings as ultralytics_settings

    # Turn on Ultralytics' built-in MLflow autolog; it reads MLFLOW_TRACKING_URI / experiment
    # name from the env (already provided by .env via the `just` recipe's dotenv).
    ultralytics_settings.update({"mlflow": True})
    os.environ["MLFLOW_EXPERIMENT_NAME"] = _TRAINING_EXPERIMENT

    architecture = architecture_from_arch(model_arch)
    model = YOLO(model_arch)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=ultralytics_device(device),
        seed=seed,
        plots=False,
        verbose=False,
    )
    best_weights = Path(model.trainer.best)  # best.pt from the just-finished run

    # Autolog started AND ended its own run; reopen it by id to graft on the custom half.
    autolog_run = mlflow.last_active_run()
    if autolog_run is None:
        raise RuntimeError("Ultralytics MLflow autolog did not produce a run")
    run_id = autolog_run.info.run_id

    with mlflow.start_run(run_id=run_id):
        tags = build_provenance_tags(
            git_sha=settings.git_sha,
            device=device,
            instance_type=settings.instance_type,
            dataset_version=dataset_version,
            architecture=architecture,
        )
        mlflow.set_tags(tags)
        # `architecture` as a param too (model-registry.md: tag/param) for MLflow-UI compare.
        mlflow.log_param("architecture", architecture)

        # The real serving signature (slice 6): base64 image in + the inference params
        # (conf/iou/max_det), per serving-io.md. No input_example — MLflow validates it by
        # *running* predict at log time, which needs the loaded weights + a real image; the
        # serving round-trip (serve_smoke.py) is where that wire contract is exercised.
        signature = build_serving_signature()

        # code_paths bakes THIS pyfunc source + the terra_incognita package (which carries the
        # pure terra_incognita.serving logic) into the model artifact, so the served container
        # reconstructs CCTDetector with no editable install — true to the baked-in image.
        code_paths = [serving_pyfunc.__file__, str(Path(terra_incognita.__file__).parent)]

        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=CCTDetector(),
            artifacts={
                WEIGHTS_ARTIFACT: str(best_weights),
                CATEGORY_MAP_ARTIFACT: str(category_map),
            },
            signature=signature,
            code_paths=code_paths,
            registered_model_name=REGISTERED_MODEL_NAME,
            pip_requirements=["ultralytics", "torch", "torchvision", "pillow", "mlflow"],
        )

    version = str(model_info.registered_model_version)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)

    metrics = dict(client.get_run(run_id).data.metrics)
    return version, metrics


def run_smoke() -> bool:
    """Run the full slice-5 chain on fixtures and return whether every acceptance check held."""
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    provider = configure_tracing(
        settings.service_name, otlp_endpoint=settings.otel_exporter_otlp_endpoint
    )
    s3 = _s3_client(settings)
    device = detect_device(settings)

    with track_run() as tracker, tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. dataset: sample → upload → register (so there is an s3_uri to materialize from).
        _s3_uri, coco_key, image_prefix = _upload_and_register_dataset(
            s3, settings, tmp_path / "fixture"
        )

        # 2. materialize the dataset FROM s3_uri into the YOLO layout (no hardcoded path).
        data_yaml, category_map, bytes_pulled = _materialize_from_s3(
            s3, settings.s3_bucket, coco_key, image_prefix, tmp_path / "materialized"
        )
        tracker.s3_bytes = bytes_pulled

        # 3+4. device-agnostic 1-epoch train (autolog) + custom provenance/signature/registry.
        version, metrics = _train_and_register(
            data_yaml, category_map, device, _DATASET_VERSION, settings
        )
        tracker.model_version = version

    # 5. operational wide event — emitted after the lifecycle block so duration is final.
    event = build_training_run_event(
        settings, tracker, dataset_version=_DATASET_VERSION, device=device
    )
    emit_result = emit_event(event, environment=settings.environment.value)
    provider.shutdown()  # flush the BatchSpanProcessor before the script exits

    # 6. acceptance checks.
    client = mlflow.MlflowClient()
    champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    champion_ok = champion.version == version
    run = client.get_run(champion.run_id)
    architecture = architecture_from_arch(_MODEL_ARCH)
    arch_ok = (
        run.data.tags.get("architecture") == architecture
        or run.data.params.get("architecture") == architecture
    )
    metrics_ok = len(metrics) > 0  # autolog logged ML metrics (mAP/precision/recall/loss)

    ok = bool(champion_ok and arch_ok and metrics_ok)
    print(f"tracking_uri   = {settings.mlflow_tracking_uri}")
    print(f"s3_endpoint    = {settings.s3_endpoint_url}")
    print(f"device         = {device.value} (instance={settings.instance_type})")
    print(f"s3_bytes       = {tracker.s3_bytes}")
    print(f"duration_ms    = {tracker.duration_ms:.1f}")
    print(f"model          = {REGISTERED_MODEL_NAME} v{version} @{CHAMPION_ALIAS}")
    print(f"architecture   = {architecture}")
    print(f"metrics        = {sorted(metrics)}")
    print(f"wide event     = {emit_result.event_name} trace_id={emit_result.trace_id}")
    print(f"champion_ok    = {champion_ok}")
    print(f"arch_ok        = {arch_ok}")
    print(f"metrics_ok     = {metrics_ok}")
    return ok


def main() -> int:
    ok = run_smoke()
    if ok:
        print(
            "\nSMOKE PASS: 1-epoch train registered a @champion with metrics + architecture, "
            "and training.run was emitted."
        )
        return 0
    print("\nSMOKE FAIL: registration / @champion / metrics / architecture did not verify.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
