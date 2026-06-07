"""Training pipeline — the *pure* logic, split from the heavy Ultralytics/MLflow run.

PLAN §6/§6b. The riskiest-to-get-right and most-worth-testing parts of training are pure
and live here so lean CI covers them without torch/ultralytics/mlflow:

  - :mod:`device` — device-agnostic resolution (``auto`` → mps/cuda/cpu) + the Ultralytics
    device-arg mapping.
  - :mod:`provenance` — the custom MLflow provenance tags (the *hybrid* logging's custom
    half) and the registered-model name / ``@champion`` alias constants.
  - :mod:`lifecycle` — the operational spine: run timing + exit reason + the ``training.run``
    wide-event builder.

The actual run — materialize the dataset from its ``s3_uri``, 1-epoch device-agnostic train
with autolog, register ``@champion`` — needs the ``ml`` extra and lives in the runnable
``scripts/train_smoke.py`` (``just train-smoke``), exactly as the dataset pipeline's heavy
half lives in ``scripts/dataset_smoke.py``. CI proves this pure logic; the smoke proves the
end-to-end wiring against the running stack.
"""

from terra_incognita.training.device import resolve_device, ultralytics_device
from terra_incognita.training.lifecycle import (
    RunTracker,
    build_training_run_event,
    track_run,
)
from terra_incognita.training.provenance import (
    ARCHITECTURE_TAG,
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    architecture_from_arch,
    build_provenance_tags,
)

__all__ = [
    "ARCHITECTURE_TAG",
    "CHAMPION_ALIAS",
    "REGISTERED_MODEL_NAME",
    "RunTracker",
    "architecture_from_arch",
    "build_provenance_tags",
    "build_training_run_event",
    "resolve_device",
    "track_run",
    "ultralytics_device",
]
