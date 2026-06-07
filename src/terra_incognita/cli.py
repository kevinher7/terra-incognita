"""Typer CLI — one entrypoint stub per pipeline step (PLAN §5/§13).

Slice 1 ships the command surface (stubs) plus one real vertical: ``demo-event``
emits a registry-validated ``training.run`` wide event through the typed OTel helper,
proving the observability path end to end. Later slices fill in each stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from terra_incognita.config import Device as ConfigDevice
from terra_incognita.config import Settings
from terra_incognita.data import CocoDataset, convert_coco_to_yolo, split_by_fraction
from terra_incognita.experiment import ExperimentConfig, load_experiment_config
from terra_incognita.obs import TrainingRunEvent, configure_tracing, emit_event
from terra_incognita.obs.events import Device, ExitReason

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
    _todo("download")


@app.command()
def subset(
    seed: Annotated[int, typer.Option(help="Seed for the one-time stratified sampling.")] = 42,
) -> None:
    """Build the seeded, location-split stratified subset (~5-10K images)."""
    # `seed` is a data-pipeline input (it fixes the one-time subset), distinct from the
    # training seed in ExperimentConfig — hence a plain CLI option, not the experiment file.
    typer.echo(f"[stub] 'subset' (seed={seed}) is not implemented yet (scaffold slice 1).")


@app.command()
def convert(
    coco: Annotated[Path, typer.Option(help="COCO annotations JSON to convert.")],
    images: Annotated[Path, typer.Option(help="Directory holding the source images.")],
    out: Annotated[Path, typer.Option(help="Output dir for the Ultralytics layout.")],
    val_fraction: Annotated[
        float, typer.Option(help="Placeholder split: fraction of images to val.")
    ] = 0.2,
    seed: Annotated[int, typer.Option(help="Seed for the placeholder split.")] = 42,
) -> None:
    """Convert COCO annotations to the Ultralytics YOLO layout.

    The split here is the *placeholder* fraction split (``split_by_fraction``); the real
    location-disjoint split is the subset step's job (PLAN §5.3). The converter itself is
    policy-free — it just honors the split it's handed.
    """
    dataset = CocoDataset.from_path(coco)
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


@app.command()
def upload() -> None:
    """Upload the subset (images + COCO file) to S3 (floci/real)."""
    _todo("upload")


@app.command("register-dataset")
def register_dataset() -> None:
    """Register the dataset version in MLflow (datasets-experiment convention)."""
    _todo("register-dataset")


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
    """Train the device-agnostic Ultralytics model and log to MLflow.

    The experiment is defined by ``--config`` (a committed ``configs/*.yaml``); the
    ``--override`` flags are an ad-hoc convenience for quick probes. Environment/parity
    (device, tracking URI, S3) comes from :class:`Settings`. Slice 1 loads + echoes the
    resolved experiment to prove the config path; the train loop lands in a later slice.
    """
    settings = Settings()
    experiment = load_experiment_config(config, epochs=epochs, imgsz=imgsz, batch=batch, seed=seed)
    typer.echo(f"resolved experiment from {config}: {experiment.as_mlflow_params()}")
    typer.echo(f"runtime: device={settings.device.value} instance={settings.instance_type}")
    _todo("train")


@app.command()
def package() -> None:
    """Package the trained model as an mlflow.pyfunc and register @champion."""
    _todo("package")


@app.command()
def serve() -> None:
    """Serve the @champion model via `mlflow models serve` (REST)."""
    _todo("serve")


@app.command()
def smoke() -> None:
    """Run the fixture smoke pipeline end to end (PLAN §10)."""
    _todo("smoke")


# --- observability vertical (the slice-1 testable path) ----------------------
def build_demo_training_run_event(
    settings: Settings, experiment: ExperimentConfig
) -> TrainingRunEvent:
    """Map runtime config + experiment into a placeholder ``training.run`` event.

    Pure (no I/O) so the acceptance test can build the same event it asserts on. The
    operational values are stand-ins; a real run fills them from the run lifecycle. Note
    the shared join key ``dataset_version`` comes from the *experiment* (what data), while
    ``device``/``instance_type`` come from *settings* (where it ran) — the same split the
    config refactor enforces.
    """
    # The event enum has no `auto` — that's a runtime hint, not an emitted value. For
    # the demo we resolve `auto` to `cpu` (a real run records the device actually used).
    device = Device.cpu if settings.device is ConfigDevice.auto else Device(settings.device.value)
    return TrainingRunEvent(
        environment=settings.environment,
        service_name=settings.service_name,
        git_sha=settings.git_sha,
        dataset_version=experiment.dataset_version or "demo-dataset",
        device=device,
        instance_type=settings.instance_type,
        exit_reason=ExitReason.completed,
        duration_ms=0.0,
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
