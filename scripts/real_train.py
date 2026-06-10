"""Real local MPS training (slice 8): the registered `cct-subset` → a genuine `@champion`.

The **one-time, not-in-CI** real counterpart to ``scripts/train_smoke.py``. Where the smoke
1-epoch-trains on a synthetic fixture, this trains the committed ``configs/cct_real.yaml``
experiment on the **real** Caltech Camera Traps subset that ``just real-dataset`` registered —
producing the first real champion the dashboard can consume. It reuses the smoke's proven
``_materialize_from_s3`` / ``_train_and_register`` *verbatim* (the same anti-drift reuse
``serve_smoke`` relies on: "what we serve is exactly what training produced"), differing only
by the inputs — the real experiment config and the location-disjoint split.

What it does (PLAN §6/§6b; model-registry.md; observability.md), all env-driven via ``Settings``:
  1. **resolve the experiment** from ``configs/cct_real.yaml`` (``--config`` to override); the
     pinned ``dataset_version`` is required — an unpinned run fails loudly.
  2. **discover the dataset** the *dashboard's* way — ``search_runs(experiment_names=["datasets"])``
     filtered by ``dataset_name``/``version`` → read ``s3_uri`` / ``coco_annotation_key`` /
     ``sampling_config_json`` (exercises dataset-conventions.md from the consumer side).
  3. **materialize from S3** into the YOLO layout, re-deriving the **location-disjoint** split
     with the dataset's own sampling knobs (``split_selected_by_location``).
  4. **device-agnostic train** at the config's epochs/imgsz/batch with Ultralytics MLflow
     autolog, then the custom provenance + real ``CCTDetector`` + register + ``@champion``.
  5. **operational wide event** — emit ``training.run`` (lifecycle only) via the OTel helper.
  6. **serving check** — load ``@champion`` and predict on a held-out **real** val image,
     asserting the serving-io.md wire contract holds (the slice's final acceptance).
"""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any

import mlflow
import pandas as pd
import typer

# `python scripts/x.py` puts only scripts/ on sys.path; add the repo root for any tests/ import.
# terra_incognita itself is installed (editable), so it needs no help.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Reuse the smoke's materialize + train/register chain verbatim, exactly as serve_smoke does.
from train_smoke import (  # noqa: E402
    _materialize_from_s3,
    _s3_client,
    _train_and_register,
    detect_device,
)

from terra_incognita.config import Settings  # noqa: E402
from terra_incognita.data import (  # noqa: E402
    SamplingConfig,
    dataset_s3_prefix,
    split_selected_by_location,
)
from terra_incognita.data.registration import DATASETS_EXPERIMENT  # noqa: E402
from terra_incognita.experiment import load_experiment_config  # noqa: E402
from terra_incognita.obs import configure_tracing, emit_event  # noqa: E402
from terra_incognita.serving import IMAGE_FIELD  # noqa: E402
from terra_incognita.training import (  # noqa: E402
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    architecture_from_arch,
    build_training_run_event,
    track_run,
)

# Must match the logical name `scripts/real_dataset.py` registered the subset under.
_DATASET_NAME = "cct-subset"
_DEFAULT_CONFIG = Path("configs/cct_real.yaml")


def _discover_dataset_run(version: str) -> dict[str, str]:
    """Find the registered ``cct-subset`` version and return its tags (the dashboard's path).

    Resolves the dataset via the ``datasets``-experiment convention rather than any local
    state, so training is coupled to the *registered* dataset (the same thing the dashboard
    and a GPU box would resolve), not to whatever happens to be on this disk.
    """
    runs = mlflow.search_runs(experiment_names=[DATASETS_EXPERIMENT])
    if runs.empty:
        raise RuntimeError(
            f"no runs in the `{DATASETS_EXPERIMENT}` experiment — run `just real-dataset`"
        )
    match = runs[
        (runs.get("tags.dataset_name") == _DATASET_NAME) & (runs.get("tags.version") == version)
    ]
    if match.empty:
        raise RuntimeError(
            f"no `{_DATASET_NAME}` version {version!r} in the `{DATASETS_EXPERIMENT}` experiment — "
            "run `just real-dataset` and pin its printed version in configs/cct_real.yaml"
        )
    row = match.iloc[0]
    return {key[len("tags.") :]: row[key] for key in match.columns if key.startswith("tags.")}


def _serving_check(model_uri: str, val_images_dir: Path) -> tuple[int, str]:
    """Load ``@champion`` and predict on one held-out real val image; assert the wire contract.

    Returns (#detections, file_name). Asserts serving-io.md shape (width/height, xyxy boxes
    inside the image, real COCO ``category_id``, class_name/score) but **not** a non-zero count
    — a deliberately weak nano model may legitimately find nothing on a disjoint-location frame,
    which is still a *sane* (well-formed) response.
    """
    val_image = next(iter(sorted(val_images_dir.glob("*.jpg"))), None)
    if val_image is None:
        raise RuntimeError(f"no val images materialized under {val_images_dir}")

    model = mlflow.pyfunc.load_model(model_uri)
    image_b64 = base64.b64encode(val_image.read_bytes()).decode("ascii")
    outputs = model.predict(pd.DataFrame({IMAGE_FIELD: [image_b64]}))
    output: dict[str, Any] = outputs[0]

    assert isinstance(output["width"], int) and output["width"] > 0, output
    assert isinstance(output["height"], int) and output["height"] > 0, output
    assert isinstance(output["detections"], list), output
    for det in output["detections"]:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        eps = 1.0
        assert -eps <= x1 <= x2 <= output["width"] + eps, det
        assert -eps <= y1 <= y2 <= output["height"] + eps, det
        assert isinstance(det["category_id"], int), det
        assert isinstance(det["class_name"], str) and det["class_name"], det
        assert isinstance(det["score"], float), det
    return len(output["detections"]), val_image.name


def run(config_path: Path) -> bool:
    """Run the full real train chain and return whether the @champion + serving check held."""
    settings = Settings()
    experiment = load_experiment_config(config_path)
    if not experiment.dataset_version:
        raise SystemExit(
            f"{config_path} has no `dataset_version` pinned — run `just real-dataset` first, "
            "then set the printed version in the config."
        )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    provider = configure_tracing(
        settings.service_name, otlp_endpoint=settings.otel_exporter_otlp_endpoint
    )
    s3 = _s3_client(settings)
    device = detect_device(settings)

    # Discover the registered dataset + the sampling knobs it was built with (for the split).
    tags = _discover_dataset_run(experiment.dataset_version)
    coco_key = tags["coco_annotation_key"]
    sampling_config = SamplingConfig.model_validate_json(tags["sampling_config_json"])
    image_prefix = f"{dataset_s3_prefix(_DATASET_NAME, experiment.dataset_version)}images/"

    with track_run() as tracker, tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Materialize FROM s3_uri, re-deriving the location-disjoint split (not the placeholder).
        data_yaml, category_map, bytes_pulled = _materialize_from_s3(
            s3,
            settings.s3_bucket,
            coco_key,
            image_prefix,
            tmp_path / "materialized",
            split=lambda ds: split_selected_by_location(ds, sampling_config),
        )
        tracker.s3_bytes = bytes_pulled

        # Train the committed experiment via the SAME registration path the smoke proves.
        version, metrics = _train_and_register(
            data_yaml,
            category_map,
            device,
            experiment.dataset_version,
            settings,
            model_arch=experiment.model_arch,
            epochs=experiment.epochs,
            imgsz=experiment.imgsz,
            batch=experiment.batch,
            seed=experiment.seed,
        )
        tracker.model_version = version

        # Held-out real-image serving check while the layout is still on disk.
        n_detections, val_name = _serving_check(
            f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}", data_yaml.parent / "images" / "val"
        )

    # Operational wide event — after the lifecycle block so duration is final.
    event = build_training_run_event(
        settings, tracker, dataset_version=experiment.dataset_version, device=device
    )
    emit_result = emit_event(event, environment=settings.environment.value)
    provider.shutdown()  # flush the BatchSpanProcessor before the script exits

    # Acceptance: @champion resolves to this version, carries architecture, and has ML metrics.
    client = mlflow.MlflowClient()
    champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    architecture = architecture_from_arch(experiment.model_arch)
    run_data = client.get_run(champion.run_id).data
    champion_ok = champion.version == version
    arch_ok = (
        run_data.tags.get("architecture") == architecture
        or run_data.params.get("architecture") == architecture
    )
    metrics_ok = len(metrics) > 0

    ok = bool(champion_ok and arch_ok and metrics_ok)
    print(f"tracking_uri   = {settings.mlflow_tracking_uri}")
    print(f"device         = {device.value} (instance={settings.instance_type})")
    print(f"dataset        = {_DATASET_NAME} {experiment.dataset_version} ({coco_key})")
    print(f"experiment     = {experiment.as_mlflow_params()}")
    print(f"s3_bytes       = {tracker.s3_bytes}")
    print(f"duration_ms    = {tracker.duration_ms:.1f}")
    print(f"model          = {REGISTERED_MODEL_NAME} v{version} @{CHAMPION_ALIAS}")
    print(f"architecture   = {architecture}")
    print(f"metrics        = {sorted(metrics)}")
    print(f"serving check  = {n_detections} detections on held-out {val_name}")
    print(f"wide event     = {emit_result.event_name} trace_id={emit_result.trace_id}")
    print(f"champion_ok={champion_ok} arch_ok={arch_ok} metrics_ok={metrics_ok}")
    return ok


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[
        Path, typer.Option(help="Experiment config YAML (the reproducible unit).")
    ] = _DEFAULT_CONFIG,
) -> None:
    """Train the real subset → a genuine @champion, emit training.run, and serve-check it."""
    ok = run(config)
    if ok:
        print(
            "\nDONE: real @champion registered with metrics + architecture; serving check passed."
        )
        raise SystemExit(0)
    print("\nFAILED: @champion / architecture / metrics / serving check did not verify.")
    raise SystemExit(1)


if __name__ == "__main__":
    app()
