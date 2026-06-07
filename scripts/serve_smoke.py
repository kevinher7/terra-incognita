"""Serving round-trip acceptance (slice 6): base64 image in → detections out — the loop closed.

Like the other smokes (``train_smoke``/``dataset_smoke``/``stack_smoke``) this needs the running
docker stack (``just up``) + the heavy ``ml`` extra (``just sync-ml``), so it is **not** a
pytest/CI test. CI proves the *pure* serving logic (the YOLO index→COCO ``category_id``
translation, the wire-format assembly, the inference-param defaults) in
``tests/test_serving.py``; this smoke proves the heavy round-trip against the real registry +
the real :class:`~serving_pyfunc.CCTDetector` artifact.

What it exercises end to end (serving-io.md; bbox-format.md; PLAN §7):
  1. **produce a champion** — run the slice-5/6 train chain (materialize → 1-epoch train →
     register), which now logs the *real* CCTDetector (base64 in / xyxy + COCO id out).
  2. **load it the way serving does** — ``mlflow.pyfunc.load_model("models:/...@champion")``,
     reconstructed from the baked-in code + artifacts (no editable install).
  3. **round-trip** — base64-encode a synthetic image, ``predict`` it, and assert the wire
     contract: ``width``/``height`` present; boxes are absolute-pixel ``xyxy``; ``category_id``
     is the **real COCO id** (the join key), not the YOLO index; ``class_name``/``score`` present.
  4. **params honored** — a low ``conf`` returns at least as many detections as a high ``conf``,
     and every returned ``score`` clears the ``conf`` floor (the param is genuinely applied).
  5. **pure/stateless** — the only input at inference is the base64 string + the baked-in model;
     no S3 client is touched (serving-io.md: serving runtime IAM ≈ none).

Endpoints/credentials come from the environment via ``Settings`` (loaded from ``.env`` by the
``just`` recipe's dotenv) — nothing localhost is hardcoded here.
"""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

# `python scripts/x.py` puts only scripts/ on sys.path; add the repo root so the synthetic
# fixture generator (under tests/) is importable. terra_incognita itself is installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.synthetic import generate_synthetic_dataset  # noqa: E402

# Reuse the train chain verbatim so "what serving serves" is exactly "what training produced".
from train_smoke import (  # noqa: E402
    _DATASET_VERSION,
    _materialize_from_s3,
    _s3_client,
    _train_and_register,
    _upload_and_register_dataset,
    detect_device,
)

from terra_incognita.config import Settings  # noqa: E402
from terra_incognita.data import CategoryIndex, CocoDataset  # noqa: E402
from terra_incognita.serving import IMAGE_FIELD  # noqa: E402
from terra_incognita.training import CHAMPION_ALIAS, REGISTERED_MODEL_NAME  # noqa: E402

# The anchor image is a fixed 800x600 canvas carrying a drawn shape (tests/fixtures/synthetic),
# so the round-trip has a real picture to detect on and known dimensions to assert.
_ROUNDTRIP_IMAGE = "img-000.png"
_ANCHOR_WIDTH = 800
_ANCHOR_HEIGHT = 600

# Bracketing confidences to show the param is honored: ~floor surfaces raw boxes, ~ceiling
# suppresses almost everything. Detection count must be monotonic across them.
_LOW_CONF = 0.01
_HIGH_CONF = 0.99


def _round_trip(
    model: mlflow.pyfunc.PyFuncModel, image_b64: str, conf: float | None
) -> dict[str, Any]:
    """Predict one base64 image and return its single per-image output dict.

    ``conf=None`` exercises the signature's *default* params (no params sent); a float overrides
    just ``conf`` — the path the dashboard uses to trade recall for precision per request.
    """
    frame = pd.DataFrame({IMAGE_FIELD: [image_b64]})
    params = None if conf is None else {"conf": conf}
    outputs = model.predict(frame, params=params)
    return outputs[0]


def _assert_wire_contract(output: dict[str, Any], expected_category_ids: set[int]) -> None:
    """Assert one image's output matches serving-io.md / bbox-format.md, or raise AssertionError."""
    assert output["width"] == _ANCHOR_WIDTH, output["width"]
    assert output["height"] == _ANCHOR_HEIGHT, output["height"]
    assert isinstance(output["detections"], list)
    for det in output["detections"]:
        box = det["bbox_xyxy"]
        assert len(box) == 4, box
        x1, y1, x2, y2 = box
        # Absolute-pixel xyxy: ordered corners, inside the image (small epsilon for box slop).
        eps = 1.0
        assert -eps <= x1 <= x2 <= _ANCHOR_WIDTH + eps, box
        assert -eps <= y1 <= y2 <= _ANCHOR_HEIGHT + eps, box
        # The join key is the REAL COCO category_id (e.g. 3/7/12), never the YOLO index (0/1/2).
        assert det["category_id"] in expected_category_ids, det
        assert isinstance(det["class_name"], str) and det["class_name"], det
        assert isinstance(det["score"], float), det


def run_smoke() -> bool:
    """Run the full serving round-trip on fixtures and return whether every check held."""
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    s3 = _s3_client(settings)
    device = detect_device(settings)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. produce a @champion: the same train chain train_smoke runs, logging the real pyfunc.
        _s3_uri, coco_key, image_prefix = _upload_and_register_dataset(
            s3, settings, tmp_path / "fixture"
        )
        data_yaml, category_map, _bytes = _materialize_from_s3(
            s3, settings.s3_bucket, coco_key, image_prefix, tmp_path / "materialized"
        )
        version, _metrics = _train_and_register(
            data_yaml, category_map, device, _DATASET_VERSION, settings
        )

        # A standalone fixture for the round-trip *input* (same category scheme as training), so
        # we know the real COCO ids the join key must be drawn from and have an image to encode.
        roundtrip = generate_synthetic_dataset(tmp_path / "roundtrip")
        dataset = CocoDataset.from_path(roundtrip.coco_path)
        expected_category_ids = set(
            CategoryIndex.from_categories(dataset.categories).index_to_category_id.values()
        )
        image_b64 = base64.b64encode((roundtrip.images_dir / _ROUNDTRIP_IMAGE).read_bytes()).decode(
            "ascii"
        )

        # 2. load it exactly as the serving runtime does — from the registry alias, reconstructed
        #    from baked-in code + artifacts (no editable install, no S3 at inference time).
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
        model = mlflow.pyfunc.load_model(model_uri)

        # 3. round-trip with the signature DEFAULT params (conf=0.25), and assert the contract.
        default_out = _round_trip(model, image_b64, conf=None)
        _assert_wire_contract(default_out, expected_category_ids)
        default_scores_ok = all(d["score"] >= 0.25 for d in default_out["detections"])

        # 4. params honored: low conf >= high conf detections, and every score clears its floor.
        low_out = _round_trip(model, image_b64, conf=_LOW_CONF)
        high_out = _round_trip(model, image_b64, conf=_HIGH_CONF)
        low_n, high_n = len(low_out["detections"]), len(high_out["detections"])
        monotonic_ok = low_n >= high_n
        floor_ok = all(d["score"] >= _LOW_CONF for d in low_out["detections"]) and all(
            d["score"] >= _HIGH_CONF for d in high_out["detections"]
        )

    contract_ok = default_scores_ok  # _assert_wire_contract already raised if shape was wrong
    params_ok = monotonic_ok and floor_ok
    ok = bool(contract_ok and params_ok)

    print(f"tracking_uri    = {settings.mlflow_tracking_uri}")
    print(f"model           = {REGISTERED_MODEL_NAME} v{version} @{CHAMPION_ALIAS}")
    print(f"image           = {_ROUNDTRIP_IMAGE} ({_ANCHOR_WIDTH}x{_ANCHOR_HEIGHT}) via base64")
    print(f"expected coco ids = {sorted(expected_category_ids)}")
    print(f"default conf=0.25 -> {len(default_out['detections'])} detections")
    print(f"conf={_LOW_CONF} -> {low_n} detections | conf={_HIGH_CONF} -> {high_n} detections")
    print(f"contract_ok     = {contract_ok} (xyxy absolute pixels + real COCO category_id)")
    print(f"params_ok       = {params_ok} (monotonic={monotonic_ok} score-floor={floor_ok})")
    print("stateless       = predicted from base64 + baked-in artifacts (no S3 at inference)")
    return ok


def main() -> int:
    ok = run_smoke()
    if ok:
        print(
            "\nSMOKE PASS: base64 round-trip returned xyxy + real COCO category_id detections, "
            "and the conf inference param was honored."
        )
        return 0
    print("\nSMOKE FAIL: serving round-trip / wire contract / inference params did not verify.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
