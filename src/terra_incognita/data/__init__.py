"""Data pipeline: COCO ingest, the COCO→YOLO converter, and the Ultralytics layout.

The riskiest pure logic in the repo lives here (PLAN §13.3): the COCO→YOLO
normalized-center transform and the YOLO contiguous-index ↔ COCO ``category_id``
map that serving depends on. It is deliberately policy-free and side-effect-light so
it can be tested exhaustively against synthetic fixtures (PLAN §10).
"""

from terra_incognita.data.coco_to_yolo import (
    CategoryIndex,
    CocoAnnotation,
    CocoCategory,
    CocoDataset,
    CocoImage,
    ConversionResult,
    Split,
    YoloLabel,
    convert_coco_to_yolo,
    load_category_index,
    normalize_bbox,
    split_by_fraction,
)

__all__ = [
    "CategoryIndex",
    "CocoAnnotation",
    "CocoCategory",
    "CocoDataset",
    "CocoImage",
    "ConversionResult",
    "Split",
    "YoloLabel",
    "convert_coco_to_yolo",
    "load_category_index",
    "normalize_bbox",
    "split_by_fraction",
]
