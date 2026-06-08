"""Serving — the *pure* half of the pyfunc inference contract (serving-io.md, PLAN §7).

The pyfunc wrapper is the single most important artifact in the repo: it turns a trained model
into the REST service the dashboard consumes (base64 image in → ``xyxy`` + COCO ``category_id``
out). Following the repo's lean/heavy split, the serving-critical *pure* logic — the YOLO
index→``category_id`` translation, the wire-format assembly, and the inference-param defaults —
lives here so lean CI proves it without torch/ultralytics/mlflow. The heavy
:class:`CCTDetector` pyfunc that loads weights, decodes base64, and runs Ultralytics inference
lives in the runnable ``scripts/serving_pyfunc.py`` (logged via ``code_paths`` so the served
image carries this source), exactly as the heavy training run lives in ``scripts/train_smoke.py``.
"""

from terra_incognita.serving.detections import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    IMAGE_FIELD,
    PARAM_CONF,
    PARAM_IOU,
    PARAM_MAX_DET,
    InferenceParams,
    RawDetection,
    build_image_output,
)

__all__ = [
    "DEFAULT_CONF",
    "DEFAULT_IOU",
    "DEFAULT_MAX_DET",
    "IMAGE_FIELD",
    "PARAM_CONF",
    "PARAM_IOU",
    "PARAM_MAX_DET",
    "InferenceParams",
    "RawDetection",
    "build_image_output",
]
