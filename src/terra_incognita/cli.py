"""Typer CLI — one entrypoint stub per pipeline step (PLAN §5/§13).

Slice 1 ships the command surface (stubs) plus one real vertical: ``demo-event``
emits a registry-validated ``training.run`` wide event through the typed OTel helper,
proving the observability path end to end. Later slices fill in each stub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from terra_incognita.config import Settings
from terra_incognita.data import (
    CocoDataset,
    SamplingConfig,
    convert_coco_to_yolo,
    sample_subset,
    split_by_fraction,
    write_subset_coco,
)
from terra_incognita.experiment import ExperimentConfig, load_experiment_config
from terra_incognita.obs import TrainingRunEvent, configure_tracing, emit_event
from terra_incognita.training import RunTracker, build_training_run_event, resolve_device

# The default experiment when none is named — a committed file, never ambient env state.
DEFAULT_CONFIG = Path("configs/baseline.yaml")

app = typer.Typer(
    name="terra-incognita",
    help="Camera-trap failure-mode CV training pipeline.",
    no_args_is_help=True,
    add_completion=False,
)


def _todo(step: str) -> None:
    """Uniform 'not implemented yet' marker for the pipeline-step stubs."""
    typer.echo(f"[stub] '{step}' is not implemented yet (scaffold slice 1).")


# --- data pipeline steps (PLAN §5) ------------------------------------------
@app.command()
def download() -> None:
    """Download Caltech Camera Traps annotations + images from LILA."""
    # The real LILA download + S3 upload + dataset registration is a heavy, one-time run that
    # needs the `ml` stack (boto3 + mlflow) and the running stack — so, like `upload`/
    # `register-dataset`, it lives in a runnable script (`scripts/real_dataset.py`), not this
    # lean CLI. The pure logic it uses (URL building, COCO clean-up, sampling) is in `src/`.
    typer.echo(
        "[stub] the real LILA download + subset + register runs via the ml stack — "
        "use `just real-dataset` (needs `just up` + `just sync-ml`)."
    )


@app.command()
def subset(
    coco: Annotated[Path, typer.Option(help="Source COCO annotations JSON to sample from.")],
    out: Annotated[Path, typer.Option(help="Output dir for the subset COCO + split.json.")],
    seed: Annotated[int, typer.Option(help="Seed for the one-time stratified sampling.")] = 42,
    min_per_class: Annotated[
        int, typer.Option(help="Floor of images kept per class (all available if fewer).")
    ] = 20,
    target_empty_ratio: Annotated[
        float, typer.Option(help="Target fraction of empty (no-box) images in the subset.")
    ] = 0.25,
    val_location_fraction: Annotated[
        float,
        typer.Option(help="Fraction of camera locations held out for val (split is disjoint)."),
    ] = 0.2,
) -> None:
    """Build the seeded, location-split stratified subset (PLAN §5.3).

    Writes ``annotations.json`` (the faithful subset COCO — the registered source of truth)
    and ``split.json`` (the location-disjoint train/val map) into ``--out``. The split is a
    *training* input for ``convert --split``, not a registration tag. ``seed`` is a
    data-pipeline input (it fixes the one-time subset), distinct from the training seed in
    ExperimentConfig — hence a plain CLI option, not the experiment file.
    """
    dataset = CocoDataset.from_path(coco)
    config = SamplingConfig(
        seed=seed,
        min_per_class=min_per_class,
        target_empty_ratio=target_empty_ratio,
        val_location_fraction=val_location_fraction,
    )
    result = sample_subset(dataset, config)

    out.mkdir(parents=True, exist_ok=True)
    subset_coco = write_subset_coco(coco, list(result.image_ids), out / "annotations.json")
    split_path = out / "split.json"
    split_path.write_text(json.dumps(result.image_splits, indent=2) + "\n", encoding="utf-8")

    typer.echo(
        f"sampled {result.num_images} images "
        f"({result.num_annotations} annotations, {result.num_empty_images} empty) "
        f"from {len(dataset.images)} -> {out}"
    )
    typer.echo(f"  per-class images: {result.per_class_counts}")
    typer.echo(f"  locations: train={result.train_locations} val={result.val_locations} (disjoint)")
    typer.echo(f"  subset COCO: {subset_coco}")
    typer.echo(f"  split: {split_path}")


@app.command()
def convert(
    coco: Annotated[Path, typer.Option(help="COCO annotations JSON to convert.")],
    images: Annotated[Path, typer.Option(help="Directory holding the source images.")],
    out: Annotated[Path, typer.Option(help="Output dir for the Ultralytics layout.")],
    split: Annotated[
        Path | None,
        typer.Option(
            help="split.json from `subset` (location-disjoint). Omit for the placeholder."
        ),
    ] = None,
    val_fraction: Annotated[
        float,
        typer.Option(help="Placeholder split: fraction of images to val (ignored if --split)."),
    ] = 0.2,
    seed: Annotated[int, typer.Option(help="Seed for the placeholder split.")] = 42,
) -> None:
    """Convert COCO annotations to the Ultralytics YOLO layout.

    Prefer ``--split split.json`` (the real location-disjoint split from the ``subset`` step,
    PLAN §5.3); without it we fall back to the *placeholder* fraction split
    (``split_by_fraction``). The converter itself is policy-free — it just honors the split
    it's handed.
    """
    dataset = CocoDataset.from_path(coco)
    if split is not None:
        image_splits = json.loads(Path(split).read_text(encoding="utf-8"))
    else:
        image_splits = split_by_fraction([img.id for img in dataset.images], val_fraction, seed)
    result = convert_coco_to_yolo(dataset, images, out, image_splits)
    typer.echo(
        f"converted {result.num_annotations} annotations across {result.num_images} images "
        f"({result.num_categories} classes) -> {result.output_dir}"
    )
    typer.echo(
        f"  images/labels per split: {result.images_per_split} | "
        f"empty={result.num_empty_images} out_of_bounds={result.num_out_of_bounds}"
    )
    typer.echo(f"  data.yaml: {result.data_yaml_path}")
    typer.echo(f"  index->category_id map: {result.category_map_path}")


# `upload` and `register-dataset` need the heavy `ml` deps (boto3 + mlflow), which the lean
# CI sync (and `ty`) doesn't have — so, like `train`/`package`/`serve`, the real path lives
# in a runnable script (`scripts/dataset_smoke.py`, `just dataset-smoke`) that exercises the
# whole sample→upload→register→verify chain against the running stack. The pure logic it
# uses (sampler, tag-building, S3-path derivation) is in `src/` and unit-tested in CI.
@app.command()
def upload() -> None:
    """Upload the subset (images + COCO file) to S3 (floci/real)."""
    typer.echo(
        "[stub] 'upload' runs via the ml stack — use `just dataset-smoke` (needs `just up`)."
    )


@app.command("register-dataset")
def register_dataset() -> None:
    """Register the dataset version in MLflow (datasets-experiment convention)."""
    typer.echo(
        "[stub] 'register-dataset' runs via the ml stack — use `just dataset-smoke` "
        "(needs `just up`)."
    )


# --- training / serving (PLAN §6/§7) ----------------------------------------
@app.command()
def train(
    config: Annotated[
        Path,
        typer.Option(help="Experiment config YAML — the reproducible unit of an experiment."),
    ] = DEFAULT_CONFIG,
    epochs: Annotated[int | None, typer.Option(help="Override config epochs (ad-hoc).")] = None,
    imgsz: Annotated[int | None, typer.Option(help="Override config imgsz (ad-hoc).")] = None,
    batch: Annotated[int | None, typer.Option(help="Override config batch (ad-hoc).")] = None,
    seed: Annotated[int | None, typer.Option(help="Override config seed (ad-hoc).")] = None,
) -> None:
    """Resolve + echo the experiment; the heavy train loop runs via the ml-extra script.

    The experiment is defined by ``--config`` (a committed ``configs/*.yaml``); the
    ``--override`` flags are an ad-hoc convenience for quick probes. Environment/parity
    (device, tracking URI, S3) comes from :class:`Settings`. This command is the *lean*
    config-resolution surface (no torch/ultralytics/mlflow import, so it stays in the CI
    sync); the actual device-agnostic train → autolog → register ``@champion`` → emit
    ``training.run`` runs in ``scripts/train_smoke.py`` (``just train-smoke``), the same
    lean/heavy split as ``upload``/``register-dataset`` (see those commands' note).
    """
    settings = Settings()
    experiment = load_experiment_config(config, epochs=epochs, imgsz=imgsz, batch=batch, seed=seed)
    typer.echo(f"resolved experiment from {config}: {experiment.as_mlflow_params()}")
    typer.echo(f"runtime: device={settings.device.value} instance={settings.instance_type}")
    typer.echo(
        "[stub] the train loop runs via the ml stack — use `just train-smoke` "
        "(needs `just up` + `just sync-ml`)."
    )


# `package` and `serve` drive the mlflow CLI against the ml stack (build-docker / models serve),
# so — like `upload`/`register-dataset` — the real path is a `just` recipe, not this lean command
# (no mlflow/ultralytics import, so it stays in the CI sync). The pyfunc they package is the real
# `CCTDetector` (scripts/serving_pyfunc.py); its pure logic lives in `terra_incognita.serving`.
@app.command()
def package() -> None:
    """Build the byte-identical serving image (model baked in) via `mlflow models build-docker`."""
    typer.echo(
        "[stub] 'package' runs via the ml stack — use `just package` "
        "(needs Docker + `just sync-ml`; builds the @champion serving image)."
    )


@app.command()
def serve() -> None:
    """Serve the @champion model via `mlflow models serve` (REST)."""
    typer.echo(
        "[stub] 'serve' runs via the ml stack — use `just serve` "
        "(needs `just up` + `just sync-ml`); round-trip it with `just serve-smoke`."
    )


@app.command()
def smoke() -> None:
    """Run the connected fixture smoke pipeline end to end (PLAN §10, slice 7)."""
    typer.echo(
        "[stub] 'smoke' is the connected end-to-end pytest pipeline (it needs the ml stack) — "
        "use `just smoke` (needs `just sync-ml`). It chains COCO→YOLO → 1-epoch train → pyfunc "
        "package → serving round-trip → training.run wide-event schema, server-free."
    )


# --- observability vertical (the slice-1 testable path) ----------------------
def build_demo_training_run_event(
    settings: Settings, experiment: ExperimentConfig
) -> TrainingRunEvent:
    """Map runtime config + experiment into a placeholder ``training.run`` event.

    Pure (no I/O) so the acceptance test can build the same event it asserts on. This now
    delegates to the *real* builder (:func:`terra_incognita.training.build_training_run_event`)
    so the observability smoke and an actual run can't drift — the only stand-ins are the
    operational values (a fresh, untouched :class:`RunTracker`: ``completed``, zero duration).

    The shared join key ``dataset_version`` comes from the *experiment* (what data), while
    ``device``/``instance_type`` come from *settings* (where it ran).
    """
    # The event enum has no `auto` — that's a runtime hint, not an emitted value. The demo
    # has no hardware to probe, so it resolves with both capabilities false (auto -> cpu);
    # an explicit device is honored. A real run passes the probed capabilities instead.
    device = resolve_device(settings.device, has_cuda=False, has_mps=False)
    return build_training_run_event(
        settings,
        RunTracker(),
        dataset_version=experiment.dataset_version or "demo-dataset",
        device=device,
    )


@app.command("demo-event")
def demo_event() -> None:
    """Emit a registry-validated training.run wide event (observability smoke)."""
    settings = Settings()
    configure_tracing(
        settings.service_name,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
    )
    # Defaults are fine for the observability smoke — it exercises the event path, not a
    # real experiment, so it never needs configs/baseline.yaml to exist.
    event = build_demo_training_run_event(settings, ExperimentConfig())
    result = emit_event(event, environment=settings.environment.value)
    typer.echo(
        f"emitted '{result.event_name}' wide event "
        f"trace_id={result.trace_id} span_id={result.span_id}"
    )


if __name__ == "__main__":
    app()
