"""Typer CLI — one entrypoint stub per pipeline step (PLAN §5/§13).

Slice 1 ships the command surface (stubs) plus one real vertical: ``demo-event``
emits a registry-validated ``training.run`` wide event through the typed OTel helper,
proving the observability path end to end. Later slices fill in each stub.
"""

from __future__ import annotations

import typer

from terra_incognita.config import Device as ConfigDevice
from terra_incognita.config import Settings
from terra_incognita.obs import TrainingRunEvent, configure_tracing, emit_event
from terra_incognita.obs.events import Device, ExitReason

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
def subset() -> None:
    """Build the seeded, location-split stratified subset (~5-10K images)."""
    _todo("subset")


@app.command()
def convert() -> None:
    """Convert COCO annotations to the Ultralytics YOLO layout."""
    _todo("convert")


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
def train() -> None:
    """Train the device-agnostic Ultralytics model and log to MLflow."""
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
def build_demo_training_run_event(settings: Settings) -> TrainingRunEvent:
    """Map config into a placeholder ``training.run`` event.

    Pure (no I/O) so the acceptance test can build the same event it asserts on. The
    operational values are stand-ins; a real run fills them from the run lifecycle.
    """
    # The event enum has no `auto` — that's a runtime hint, not an emitted value. For
    # the demo we resolve `auto` to `cpu` (a real run records the device actually used).
    device = Device.cpu if settings.device is ConfigDevice.auto else Device(settings.device.value)
    return TrainingRunEvent(
        environment=settings.environment,
        service_name=settings.service_name,
        git_sha=settings.git_sha,
        dataset_version=settings.dataset_version or "demo-dataset",
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
    event = build_demo_training_run_event(settings)
    result = emit_event(event, environment=settings.environment.value)
    typer.echo(
        f"emitted '{result.event_name}' wide event "
        f"trace_id={result.trace_id} span_id={result.span_id}"
    )


if __name__ == "__main__":
    app()
