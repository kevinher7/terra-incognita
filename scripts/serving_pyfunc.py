"""The serving pyfunc — *the single most important artifact in the repo* (PLAN §7, serving-io.md).

This is the heavy half of serving: it needs the ml stack (mlflow + ultralytics + torch +
Pillow), so — like ``scripts/train_smoke.py`` — it lives in ``scripts/`` (excluded from the lean
CI typecheck), not in the typed ``src/`` surface. The serving-*critical* but ml-free logic (the
YOLO index→COCO ``category_id`` translation, the wire-format assembly, the inference-param
defaults) is in :mod:`terra_incognita.serving`, unit-tested in CI; this file only owns what
genuinely requires the stack.

**How it reaches the serving runtime.** The training run logs this model with
``code_paths=[this file, the terra_incognita package]`` (see ``train_smoke.py``), so MLflow bakes
this source *and* :mod:`terra_incognita.serving` into the model artifact. ``mlflow models serve``
/ ``build-docker`` then reconstruct :class:`CCTDetector` with no editable install and no S3 at
runtime — pure, stateless inference: base64 image in → detections out.

The wire contract (serving-io.md):
  - **Input:** one base64-encoded image per record, column ``image_b64``.
  - **Params** (MLflow signature ``params``): ``conf`` (0.25), ``iou`` (0.45), ``max_det`` (300).
  - **Output:** per image, ``width``/``height`` + ``detections[]`` with ``bbox_xyxy``
    (absolute-pixel ``xyxy``), the real COCO ``category_id`` (join key), ``class_name``, ``score``.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

if TYPE_CHECKING:
    # Annotation-only: PIL is imported lazily at inference (ml extra), never at module load.
    from PIL import Image

from terra_incognita.data import load_category_index
from terra_incognita.serving import (
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

# Artifact keys the training run logs (see train_smoke.py) and load_context reads back. The
# weights are the trained YOLO checkpoint; the category map is the index→category_id artifact
# the converter wrote (coco_to_yolo.CATEGORY_MAP_FILENAME) — it MUST be the one from the dataset
# this model trained on, or the served category_id join key is wrong (serving-io.md).
WEIGHTS_ARTIFACT = "weights"
CATEGORY_MAP_ARTIFACT = "category_map"


def build_serving_signature() -> ModelSignature:
    """The MLflow model signature: base64-image input + the contract's inference ``params``.

    Inputs are a single ``image_b64`` string column (batchable: one record per image). The
    params carry the serving-io.md defaults (``conf``/``iou``/``max_det``) so a caller can omit
    any of them and MLflow fills the default before ``predict`` runs. The *output* schema is
    intentionally left unspecified: the response is a per-image nested object
    (``width``/``height``/``detections[]``) that MLflow's columnar schema can't express cleanly,
    and pinning a brittle nested schema would buy nothing the wire contract + round-trip test
    don't already guarantee.
    """
    inputs = Schema([ColSpec("string", IMAGE_FIELD)])
    params = ParamSchema(
        [
            ParamSpec(PARAM_CONF, "double", DEFAULT_CONF),
            ParamSpec(PARAM_IOU, "double", DEFAULT_IOU),
            ParamSpec(PARAM_MAX_DET, "long", DEFAULT_MAX_DET),
        ]
    )
    return ModelSignature(inputs=inputs, params=params)


class CCTDetector(mlflow.pyfunc.PythonModel):
    """Custom pyfunc: base64 image(s) in → detections out, honoring per-request params.

    Stateless by construction — everything it needs (weights + the index→``category_id`` map) is
    baked into the artifact at log time and loaded once in :meth:`load_context`. No S3/IAM at
    inference time (serving-io.md): images arrive base64, the model is baked in.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the YOLO weights + the category map once, at container/model startup."""
        from ultralytics import YOLO

        self._model = YOLO(context.artifacts[WEIGHTS_ARTIFACT])
        # The serving join key: YOLO contiguous index → real COCO category_id (serving-io.md).
        self._category_index = load_category_index(context.artifacts[CATEGORY_MAP_ARTIFACT])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: object,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run inference on each base64 image and return its wire-format detections.

        ``model_input`` is a batch of base64 images (a DataFrame with the ``image_b64`` column
        over REST; a list/dict is also accepted for direct in-process calls). ``params`` are the
        signature's inference knobs — resolved + validated through :class:`InferenceParams`, then
        passed straight to Ultralytics so ``conf``/``iou``/``max_det`` are genuinely honored.
        """
        resolved = InferenceParams.from_params(params)
        images = [self._decode(b64) for b64 in _iter_image_b64(model_input)]
        if not images:
            return []

        results = self._model.predict(
            images,
            conf=resolved.conf,
            iou=resolved.iou,
            max_det=resolved.max_det,
            verbose=False,
        )

        outputs: list[dict[str, Any]] = []
        for result in results:
            height, width = result.orig_shape  # Ultralytics gives (h, w) of the ORIGINAL image
            raw = [
                RawDetection(
                    bbox_xyxy=tuple(box.xyxy[0].tolist()),
                    class_index=int(box.cls[0]),
                    score=float(box.conf[0]),
                )
                for box in result.boxes
            ]
            outputs.append(
                build_image_output(
                    width=width,
                    height=height,
                    raw=raw,
                    category_index=self._category_index,
                )
            )
        return outputs

    @staticmethod
    def _decode(image_b64: str) -> Image.Image:
        """Base64 string → an RGB PIL image to hand to Ultralytics (width/height come off the
        inference result's ``orig_shape``, so we never need the dimensions here)."""
        from PIL import Image

        return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _iter_image_b64(model_input: object) -> list[str]:
    """Pull the list of base64 strings out of whatever shape the input arrived in.

    REST serving hands ``predict`` a pandas DataFrame (the ``image_b64`` column); direct
    in-process / round-trip calls may pass a list of records, a list of strings, or a column
    dict. We normalize all of them to a plain ``list[str]`` so the contract is forgiving at the
    edge without leaking pandas into the pure assembly.
    """
    # pandas DataFrame (the REST path) — duck-typed to avoid importing pandas here.
    if hasattr(model_input, "columns"):
        return model_input[IMAGE_FIELD].tolist()
    if isinstance(model_input, dict):
        return list(model_input[IMAGE_FIELD])
    if isinstance(model_input, list):
        return [item[IMAGE_FIELD] if isinstance(item, dict) else item for item in model_input]
    raise TypeError(f"unsupported model_input type for serving: {type(model_input).__name__}")
