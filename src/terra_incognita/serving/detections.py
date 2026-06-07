"""Serving output assembly + inference params — the *pure* half of the pyfunc (serving-io.md).

The two serving-critical pieces of logic live here, kept free of torch/ultralytics/mlflow so
lean CI can assert them exhaustively (the same lean/heavy split as the COCO→YOLO converter and
the training provenance — heavy wiring lives in ``scripts/``):

1. **The index→``category_id`` translation + wire-format assembly.** The model emits a
   *contiguous YOLO class index*; the dashboard joins on the *real COCO ``category_id``*. This
   module reverses the converter's class-index transform using the same
   :class:`~terra_incognita.data.CategoryIndex` map stored with the model artifact, so a
   prediction's ``category_id`` is the COCO join key, not the YOLO index (serving-io.md,
   bbox-format.md). Getting this wrong silently mis-labels every detection — so it is pure and
   directly unit-tested.

2. **The inference params** (``conf``/``iou``/``max_det``) with the contract's defaults
   (0.25 / 0.45 / 300), resolved from a possibly-partial params dict into a typed, validated
   :class:`InferenceParams`.

The heavy :class:`CCTDetector` pyfunc (``scripts/serving_pyfunc.py``) owns only the parts that
*need* the ml stack — base64 decode, Ultralytics inference honoring these params, pulling raw
boxes off the result — then hands plain :class:`RawDetection`\\s here to be shaped into the wire
format. So the box→``category_id`` join is testable without a single heavy import.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from terra_incognita.data import CategoryIndex

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

# The single base64-image input column name — both the signature (heavy side) and the input
# extraction agree on it via this one constant, so they can never silently disagree.
IMAGE_FIELD = "image_b64"

# Inference-param names + the contract's defaults (serving-io.md). Names are constants so the
# MLflow ParamSchema (heavy) and the params resolution (here) reference the same strings.
PARAM_CONF = "conf"
PARAM_IOU = "iou"
PARAM_MAX_DET = "max_det"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_MAX_DET = 300


@dataclass(frozen=True)
class InferenceParams:
    """Resolved, validated per-request inference knobs (serving-io.md signature ``params``).

    ``conf``/``iou`` are confidence/NMS-IoU thresholds in ``[0, 1]``; ``max_det`` caps the
    detections returned per image. Built via :meth:`from_params` so a caller (or MLflow's
    signature defaulting) can pass any subset and the contract defaults fill the rest.
    """

    conf: float = DEFAULT_CONF
    iou: float = DEFAULT_IOU
    max_det: int = DEFAULT_MAX_DET

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None) -> InferenceParams:
        """Resolve a possibly-``None``/partial params mapping into typed, validated params.

        ``None`` (no params sent) yields the contract defaults. Values are coerced to
        ``float``/``int`` (MLflow may hand us numpy scalars over the wire) and range-checked so
        a nonsensical request fails loud here rather than silently mis-thresholding inference.
        """
        params = params or {}
        conf = float(params.get(PARAM_CONF, DEFAULT_CONF))
        iou = float(params.get(PARAM_IOU, DEFAULT_IOU))
        max_det = int(params.get(PARAM_MAX_DET, DEFAULT_MAX_DET))
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"{PARAM_CONF} must be in [0, 1], got {conf}")
        if not 0.0 <= iou <= 1.0:
            raise ValueError(f"{PARAM_IOU} must be in [0, 1], got {iou}")
        if max_det < 1:
            raise ValueError(f"{PARAM_MAX_DET} must be >= 1, got {max_det}")
        return cls(conf=conf, iou=iou, max_det=max_det)


@dataclass(frozen=True)
class RawDetection:
    """One detection as it comes off the model: an absolute-pixel ``xyxy`` box, the *YOLO*
    class index, and the confidence score.

    The plain handoff type between the heavy pyfunc (which reads these off an Ultralytics
    ``Result``) and the pure assembly here — so the index→``category_id`` join needs no
    ultralytics types and is unit-testable.
    """

    bbox_xyxy: tuple[float, float, float, float]
    class_index: int
    score: float


def build_image_output(
    *,
    width: int,
    height: int,
    raw: Sequence[RawDetection],
    category_index: CategoryIndex,
) -> dict[str, Any]:
    """Shape one image's raw detections into the serving wire format (serving-io.md).

    Returns ``{"width", "height", "detections": [...]}`` where each detection carries
    ``bbox_xyxy`` (absolute-pixel ``[x1, y1, x2, y2]`` — bbox-format.md), the **real COCO
    ``category_id``** (translated from the YOLO index via ``category_index``), the
    ``class_name`` (convenience), and the ``score``. ``category_id`` is the dashboard's join
    key; ``class_name`` is for humans.
    """
    index_to_category_id = category_index.index_to_category_id
    names = category_index.names
    detections = [
        {
            "bbox_xyxy": list(det.bbox_xyxy),
            "category_id": _category_id_for(det.class_index, index_to_category_id),
            "class_name": _class_name_for(det.class_index, names),
            "score": float(det.score),
        }
        for det in raw
    ]
    return {"width": int(width), "height": int(height), "detections": detections}


def _category_id_for(class_index: int, index_to_category_id: Mapping[int, int]) -> int:
    """Translate a YOLO index → real COCO ``category_id``; fail loud on an unknown index.

    An index the map doesn't know means the served weights and the category map disagree (the
    map didn't come from the dataset this model trained on — serving-io.md). Surfacing it beats
    returning a wrong join key the dashboard would silently mis-attribute.
    """
    try:
        return index_to_category_id[class_index]
    except KeyError:
        raise ValueError(
            f"model emitted YOLO class index {class_index} absent from the category map "
            f"(known indices: {sorted(index_to_category_id)}) — weights/map mismatch"
        ) from None


def _class_name_for(class_index: int, names: Mapping[int, str]) -> str:
    try:
        return names[class_index]
    except KeyError:
        raise ValueError(
            f"model emitted YOLO class index {class_index} absent from the category map "
            f"(known indices: {sorted(names)}) — weights/map mismatch"
        ) from None
