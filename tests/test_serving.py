"""Unit tests for the pure serving logic (slice 6): the YOLO index→COCO ``category_id``
translation + wire-format assembly, and the inference-param defaults/validation.

These are the lean, CI-gated half of the serving slice — they need no torch/ultralytics/mlflow
and no running stack, exactly because the serving-*critical* logic was deliberately kept pure
(``terra_incognita.serving``). The heavy base64 round-trip (``scripts/serve_smoke.py``) proves
the end-to-end wiring against the real registry + CCTDetector and is *not* run in CI (mirrors the
train/dataset/stack smokes).
"""

from __future__ import annotations

import pytest

from terra_incognita.data import CategoryIndex, CocoCategory
from terra_incognita.serving import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    InferenceParams,
    RawDetection,
    build_image_output,
)


def _category_index() -> CategoryIndex:
    # Deliberately sparse, out-of-order COCO ids so the test exercises the real remap
    # (sorted by id -> index 0=3 raccoon, 1=7 coyote, 2=12 bobcat) not an identity mapping.
    return CategoryIndex.from_categories(
        [
            CocoCategory(id=12, name="bobcat"),
            CocoCategory(id=3, name="raccoon"),
            CocoCategory(id=7, name="coyote"),
        ]
    )


# --- inference params --------------------------------------------------------
def test_inference_params_defaults_match_contract():
    # The serving-io.md defaults, both as constants and as the None-params resolution.
    assert (DEFAULT_CONF, DEFAULT_IOU, DEFAULT_MAX_DET) == (0.25, 0.45, 300)
    params = InferenceParams.from_params(None)
    assert (params.conf, params.iou, params.max_det) == (0.25, 0.45, 300)


def test_inference_params_partial_override_keeps_other_defaults():
    params = InferenceParams.from_params({"conf": 0.5})
    assert params.conf == 0.5
    assert params.iou == DEFAULT_IOU
    assert params.max_det == DEFAULT_MAX_DET


def test_inference_params_coerces_types():
    # MLflow may hand numpy/str-ish scalars over the wire; we coerce to float/int.
    params = InferenceParams.from_params({"conf": 1, "iou": 0, "max_det": 10.0})
    assert isinstance(params.conf, float) and params.conf == 1.0
    assert isinstance(params.max_det, int) and params.max_det == 10


@pytest.mark.parametrize(
    "bad",
    [{"conf": -0.1}, {"conf": 1.5}, {"iou": 2.0}, {"max_det": 0}],
)
def test_inference_params_rejects_out_of_range(bad: dict[str, float]):
    with pytest.raises(ValueError):
        InferenceParams.from_params(bad)


# --- wire-format assembly + index->category_id translation ------------------
def test_build_image_output_translates_index_to_real_coco_id():
    raw = [
        RawDetection(bbox_xyxy=(10.0, 20.0, 110.0, 220.0), class_index=0, score=0.9),
        RawDetection(bbox_xyxy=(0.0, 0.0, 5.0, 5.0), class_index=2, score=0.5),
    ]
    out = build_image_output(width=2048, height=1536, raw=raw, category_index=_category_index())

    assert out["width"] == 2048
    assert out["height"] == 1536
    first, second = out["detections"]
    # Index 0 -> real COCO id 3 (raccoon); the join key is the COCO id, never the YOLO index.
    assert first["category_id"] == 3
    assert first["class_name"] == "raccoon"
    assert first["bbox_xyxy"] == [10.0, 20.0, 110.0, 220.0]  # absolute-pixel xyxy, passed through
    assert isinstance(first["score"], float) and first["score"] == 0.9
    # Index 2 -> real COCO id 12 (bobcat).
    assert second["category_id"] == 12
    assert second["class_name"] == "bobcat"


def test_build_image_output_empty_detections_keeps_dimensions():
    out = build_image_output(width=640, height=480, raw=[], category_index=_category_index())
    assert out == {"width": 640, "height": 480, "detections": []}


def test_build_image_output_rejects_unknown_index():
    # A model index absent from the map means weights/map mismatch — fail loud, not a wrong join.
    raw = [RawDetection(bbox_xyxy=(0.0, 0.0, 1.0, 1.0), class_index=99, score=0.4)]
    with pytest.raises(ValueError, match="mismatch"):
        build_image_output(width=10, height=10, raw=raw, category_index=_category_index())
