"""Connected end-to-end smoke (slice 7) — the regression backbone that chains every stage.

Slices 2/4/5/6 each test their own stage in isolation; this test wires them into **one
connected pipeline on synthetic fixtures** and is the PR gate (``.github/workflows/ci.yml``
``smoke`` job). It runs, in sequence:

  1. **COCO→YOLO (assert math)** — convert a synthetic fixture to the Ultralytics layout and
     assert the anchor label equals the hand-computed normalized box. Breaking the
     normalization math fails here — the first deliberate-break gate (issue #7 acceptance).
  2. **1-epoch train + pyfunc package + register ``@champion``** — reuse the *real*
     ``scripts/train_smoke.py`` train chain verbatim (autolog → provenance → ``log_model`` →
     register), proving training + MLflow logging fire.
  3. **serving round-trip** — load the champion in-process via the registry alias and assert
     the wire contract (base64 in → ``xyxy`` + real COCO ``category_id`` out). Breaking the
     serving contract fails here — the second deliberate-break gate.
  4. **wide-event schema** — emit ``training.run`` through an in-memory OTel exporter and
     assert every registry-required field is present (no SigNoz, no OTLP).

**Why this is a CI test (unlike the per-stage ``scripts/*_smoke.py``):** it runs **server-free**
— MLflow on a tmp **SQLite backend + tmp local artifact dir** (the registry works without a
server on SQLite), the serving round-trip **in-process** via ``mlflow.pyfunc.load_model`` (no
REST, no Docker), and the wide event through the conftest in-memory exporter. So it needs no
``just up`` stack — only the heavy ``ml`` extra, which the dedicated CI ``smoke`` job installs
(with CPU torch). Without that extra the whole module is ``importorskip``-skipped, so lean CI's
``pytest`` stays green.

It deliberately **reuses** the heavy logic from ``scripts/`` (the train chain, the serving
round-trip helpers) rather than refactoring it into ``src/`` — pulling the ``ml`` imports into
``src/`` would break the lean CI boundary the repo maintains. ``serve_smoke.py`` already imports
from ``train_smoke.py`` via ``sys.path``; this test follows the same established pattern.
"""

from __future__ import annotations

import pytest

# Guard FIRST: skip the whole module (cleanly, at collection) before any heavy import, so the
# lean CI `pytest` (no `--extra ml`) never errors importing torch/ultralytics/mlflow.
pytest.importorskip("torch")
pytest.importorskip("ultralytics")
pytest.importorskip("mlflow")

import base64
import math
import sys
from pathlib import Path

import mlflow
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import Tracer

# `scripts/` is not a package on sys.path; add it so the real train/serve smoke logic is
# importable here, exactly as `serve_smoke.py` imports `train_smoke.py`. `tests.fixtures` is
# already importable via pytest's rootdir (other tests import it the same way).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from serve_smoke import (  # noqa: E402
    _HIGH_CONF,
    _LOW_CONF,
    _ROUNDTRIP_IMAGE,
    _assert_wire_contract,
    _round_trip,
)
from train_smoke import (  # noqa: E402
    _DATASET_VERSION,
    _train_and_register,
    detect_device,
)

from terra_incognita.config import Settings  # noqa: E402
from terra_incognita.data import CategoryIndex, CocoDataset, convert_coco_to_yolo  # noqa: E402
from terra_incognita.obs import emit_event  # noqa: E402
from terra_incognita.obs.registry import load_registry  # noqa: E402
from terra_incognita.training import (  # noqa: E402
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    build_training_run_event,
    track_run,
)
from tests.fixtures.synthetic import SyntheticDataset, generate_synthetic_dataset  # noqa: E402

# The default conf the serving signature fills when a caller omits it (serving-io.md).
_DEFAULT_CONF = 0.25


def _assert_anchor_math(yolo_dir: Path, fixture: SyntheticDataset) -> None:
    """Assert the converted anchor label equals the hand-computed normalized box.

    Mirrors ``tests/test_coco_to_yolo.py::test_label_values_match_hand_computed_anchor`` so the
    connected smoke fails if anyone perturbs the COCO→YOLO normalization (issue #7 acceptance).
    """
    anchor = fixture.anchor
    split = fixture.image_splits[anchor.image_stem]
    label_text = (yolo_dir / "labels" / split / f"{anchor.image_stem}.txt").read_text(
        encoding="utf-8"
    )
    lines = [line for line in label_text.splitlines() if line.strip()]
    assert len(lines) == 1, lines
    parts = lines[0].split()
    assert int(parts[0]) == anchor.yolo_index
    values = [float(p) for p in parts[1:]]
    for got, expected in zip(values, anchor.expected_norm, strict=True):
        assert math.isclose(got, expected, abs_tol=1e-6), (got, expected)


def test_smoke_pipeline_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    tracer: Tracer,
) -> None:
    """The connected pipeline on fixtures: convert → train → register → serve → wide event."""
    # Capture runtime/provenance from the real env + git BEFORE we chdir into the tmp sandbox.
    settings = Settings()
    device = detect_device(settings)  # CPU in CI; MPS/CPU locally — same code (PLAN §6).

    # --- server-free MLflow: tmp SQLite backend + tmp local artifact dir (no server) --------
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    # Ultralytics' autolog callback reads MLFLOW_TRACKING_URI from the env; the registry client
    # calls below read mlflow's global. Set both so they agree on the SQLite store. setenv via
    # monkeypatch so a dotenv-loaded `just smoke` can't leave the real localhost:5000 in os.environ.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    # Pre-create the training experiment with a tmp artifact_location so the pyfunc artifacts
    # land in the tmp dir (not ./mlruns); autolog's set_experiment then reuses this experiment.
    mlflow.create_experiment("training", artifact_location=artifacts_dir.as_uri())

    # Keep Ultralytics' run dirs (./runs/...) and any relative writes inside the tmp sandbox.
    monkeypatch.chdir(tmp_path)

    # --- 1. fixture → COCO→YOLO (+ assert the normalization math) ---------------------------
    fixture = generate_synthetic_dataset(tmp_path / "fixture")
    yolo_dir = tmp_path / "yolo"
    convert = convert_coco_to_yolo(
        fixture.coco_path, fixture.images_dir, yolo_dir, fixture.image_splits
    )
    _assert_anchor_math(yolo_dir, fixture)

    # Ultralytics deliberately disables its MLflow autolog callback when it detects pytest or
    # GitHub Actions (`utils.TESTS_RUNNING` guard in callbacks/mlflow.py) — i.e. exactly the two
    # environments this smoke runs in, which would silently skip the MLflow logging the smoke
    # exists to prove. Flip the flag before training so the hybrid autolog actually fires. Safe
    # here because the callback module is imported lazily during `model.train()` (after this
    # patch), not at `import ultralytics`.
    import ultralytics.utils as ultralytics_utils

    monkeypatch.setattr(ultralytics_utils, "TESTS_RUNNING", False)

    # --- 2. 1-epoch train → pyfunc package → register @champion (reused verbatim, S3-free) ---
    # track_run() times the lifecycle and records exit_reason for the wide event (step 4).
    with track_run() as tracker:
        version, metrics = _train_and_register(
            convert.data_yaml_path, convert.category_map_path, device, _DATASET_VERSION, settings
        )
        tracker.model_version = version
    assert metrics, "Ultralytics autolog must log ML metrics (mAP/precision/recall/loss)"

    # --- 3. serving round-trip: load @champion in-process, base64 in → detections out -------
    # A standalone fixture for the round-trip input (same category scheme as training), so we
    # know the real COCO ids the join key must be drawn from and have an image to encode.
    roundtrip = generate_synthetic_dataset(tmp_path / "roundtrip")
    rt_dataset = CocoDataset.from_path(roundtrip.coco_path)
    expected_category_ids = set(
        CategoryIndex.from_categories(rt_dataset.categories).index_to_category_id.values()
    )
    image_b64 = base64.b64encode((roundtrip.images_dir / _ROUNDTRIP_IMAGE).read_bytes()).decode(
        "ascii"
    )

    # Loaded exactly as the serving runtime does — from the registry alias, reconstructed from
    # the baked-in code + artifacts (no editable install, no S3 at inference time).
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")

    # Default params (conf=0.25): assert the wire contract, then that every score clears the floor.
    default_out = _round_trip(model, image_b64, conf=None)
    _assert_wire_contract(default_out, expected_category_ids)
    assert all(det["score"] >= _DEFAULT_CONF for det in default_out["detections"])

    # conf param genuinely honored: low conf surfaces >= as many detections as high conf.
    low_out = _round_trip(model, image_b64, conf=_LOW_CONF)
    high_out = _round_trip(model, image_b64, conf=_HIGH_CONF)
    assert len(low_out["detections"]) >= len(high_out["detections"])

    # --- 4. wide-event schema, integrated (in-memory exporter — no SigNoz/OTLP) -------------
    event = build_training_run_event(
        settings, tracker, dataset_version=_DATASET_VERSION, device=device
    )
    # environment="local" makes emit_event raise on any missing required field (validate-loud).
    emit_event(event, tracer=tracer, environment="local")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "training.run"
    attrs = dict(span.attributes or {})

    registry = load_registry()
    required = registry.settable_required_canonical() | set(registry.required_for("training.run"))
    for field in required:
        assert field in attrs, f"missing required wide-event field {field!r}"
    # The shared join key back to the registered model — the wide event and MLflow agree on it.
    assert attrs["camtrap.model.version"] == version
