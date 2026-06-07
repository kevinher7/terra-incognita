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
from terra_incognita.data.registration import (
    COCO_ARTIFACT_FILENAME,
    DATASETS_EXPERIMENT,
    DatasetVersion,
    build_dataset_tags,
    coco_annotation_key,
    dataset_s3_prefix,
    dataset_s3_uri,
)
from terra_incognita.data.subset import (
    SamplingConfig,
    SubsetResult,
    sample_subset,
    write_subset_coco,
)

__all__ = [
    "COCO_ARTIFACT_FILENAME",
    "DATASETS_EXPERIMENT",
    "CategoryIndex",
    "CocoAnnotation",
    "CocoCategory",
    "CocoDataset",
    "CocoImage",
    "ConversionResult",
    "DatasetVersion",
    "SamplingConfig",
    "Split",
    "SubsetResult",
    "YoloLabel",
    "build_dataset_tags",
    "coco_annotation_key",
    "convert_coco_to_yolo",
    "dataset_s3_prefix",
    "dataset_s3_uri",
    "load_category_index",
    "normalize_bbox",
    "sample_subset",
    "split_by_fraction",
    "write_subset_coco",
]
