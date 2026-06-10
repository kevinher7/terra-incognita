"""LILA Caltech Camera Traps access — the on-demand real-data source (PLAN §5.1, issue #8).

The whole pipeline (sampler, converter, registration, training, serving) is proven on
synthetic fixtures in CI; slice 8 points it at the **real** Caltech Camera Traps set hosted
on `LILA <https://lila.science/datasets/caltech-camera-traps>`_. This module owns the only
*pure* bit of new logic that the real path needs — the public download URLs and the COCO
"clean-up" that makes the real annotation file fit the typed pipeline — so it stays lean,
import-light (no boto3/mlflow/torch), and unit-testable in CI. The heavy I/O (the actual
HTTP download, S3 upload, MLflow registration) lives in ``scripts/real_dataset.py``
(``just real-dataset``), the same lean/heavy split as ``registration`` vs ``dataset_smoke``.

**Why a clean-up step at all.** The real ``caltech_bboxes_20200316.json`` is a single COCO
file covering ~63K images that "only have one species label **or are empty**". Empty images
are encoded as ``category_id: 30`` ("empty") annotations **without a real bounding box**. Our
typed :class:`~terra_incognita.data.coco_to_yolo.CocoAnnotation` requires a 4-number ``bbox``
(it is a *detection* annotation), so the raw file would fail to parse. :func:`clean_bbox_coco`
drops every annotation lacking a valid box, which turns each empty image into a
**zero-annotation image** — exactly the form :func:`~terra_incognita.data.subset.sample_subset`
already treats as "empty" and the converter writes as an empty (background) label file. So
the empties the deliverable wants (~20-30% of the subset) come from this one file for free;
no second image-level metadata file is needed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CCT_BBOX_FILENAME",
    "CCT_BBOX_URL",
    "CCT_IMAGE_BASE_URL",
    "clean_bbox_coco",
    "image_url",
]

# The bounding-box annotation file (info / categories / annotations / images), ~38 MB, public,
# no auth. Hosted on the LILA GCS mirror. Its images carry ``location`` (the camera/site id the
# sampler splits on) and ``file_name`` (the per-image key below).
CCT_BBOX_FILENAME = "caltech_bboxes_20200316.json"
CCT_BBOX_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    f"caltechcameratraps/labels/{CCT_BBOX_FILENAME}"
)

# The "unzipped" image folder: every image is individually addressable as
# ``{CCT_IMAGE_BASE_URL}{file_name}``, so we download **only the selected subset** (~5K) rather
# than the 6 GB resized / 105 GB full archives. Verified to return HTTP 206 with range support.
CCT_IMAGE_BASE_URL = (
    "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/"
)


def image_url(file_name: str) -> str:
    """The public URL of a single CCT image by its COCO ``file_name`` (e.g. ``<uuid>.jpg``)."""
    return f"{CCT_IMAGE_BASE_URL}{file_name}"


def clean_bbox_coco(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a faithful COCO dict keeping only annotations that carry a real bounding box.

    Faithful = every original ``images``/``categories`` field is preserved verbatim (the
    registered COCO is the dashboard's source of truth, so it must not be silently lossy);
    only the ``annotations`` list is filtered, dropping the bbox-less "empty" markers
    (see module docstring). Images that had *only* such markers become zero-annotation
    images — the sampler's "empty" case — so the empties survive as images while never
    reaching the typed detection model. Pure (no I/O), so CI tests it against a tiny dict.
    """
    annotations = [ann for ann in raw.get("annotations", []) if _has_bbox(ann)]
    return {**raw, "annotations": annotations}


def _has_bbox(annotation: dict[str, Any]) -> bool:
    """True iff the annotation has a 4-number ``bbox`` (a detection, not an empty marker)."""
    bbox = annotation.get("bbox")
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4
