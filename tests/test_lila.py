"""LILA Caltech access tests — the pure real-data adapter (issue #8, PLAN §5.1).

The heavy download/upload/registration is a not-in-CI script (``scripts/real_dataset.py``);
what CI can and must prove is the one bit of *pure* logic the real path adds: that the real
annotation file is reshaped into something the typed pipeline accepts **without losing
images** — specifically that the bbox-less "empty" markers become zero-annotation images (the
sampler's "empty" case) while every real box and every image/category survives verbatim.
"""

from __future__ import annotations

from terra_incognita.data import (
    CCT_IMAGE_BASE_URL,
    CocoDataset,
    SamplingConfig,
    clean_bbox_coco,
    image_url,
    sample_subset,
)

# A miniature of the real file's shape: 22-style sparse category ids incl. the "empty" (30)
# category, two boxed images, and two images whose only annotation is a bbox-less empty marker.
_RAW = {
    "info": {"version": "test"},
    "categories": [
        {"id": 6, "name": "bobcat"},
        {"id": 1, "name": "opossum"},
        {"id": 30, "name": "empty"},
    ],
    "annotations": [
        {"id": "a1", "image_id": "i1", "category_id": 6, "bbox": [10.0, 20.0, 30.0, 40.0]},
        {"id": "a2", "image_id": "i2", "category_id": 1, "bbox": [1.0, 2.0, 3.0, 4.0]},
        # Empty markers: category 30, NO bbox — would fail CocoAnnotation (bbox required).
        {"id": "a3", "image_id": "i3", "category_id": 30},
        {"id": "a4", "image_id": "i4", "category_id": 30},
    ],
    "images": [
        {"id": "i1", "file_name": "i1.jpg", "width": 100, "height": 100, "location": "5"},
        {"id": "i2", "file_name": "i2.jpg", "width": 100, "height": 100, "location": "5"},
        {"id": "i3", "file_name": "i3.jpg", "width": 100, "height": 100, "location": "9"},
        {"id": "i4", "file_name": "i4.jpg", "width": 100, "height": 100, "location": "9"},
    ],
}


def test_image_url_joins_base_and_file_name():
    assert image_url("abc.jpg") == f"{CCT_IMAGE_BASE_URL}abc.jpg"


def test_clean_drops_only_bbox_less_annotations():
    cleaned = clean_bbox_coco(_RAW)
    # The two empty markers are gone; the two real boxes remain.
    assert {ann["id"] for ann in cleaned["annotations"]} == {"a1", "a2"}


def test_clean_preserves_all_images_and_categories_verbatim():
    cleaned = clean_bbox_coco(_RAW)
    # Faithful: images + categories untouched (the registered COCO is the source of truth),
    # so the empty images survive as images even though their annotations were dropped.
    assert cleaned["images"] == _RAW["images"]
    assert cleaned["categories"] == _RAW["categories"]
    assert cleaned["info"] == _RAW["info"]


def test_clean_does_not_mutate_the_input():
    clean_bbox_coco(_RAW)
    assert len(_RAW["annotations"]) == 4  # original list untouched


def test_cleaned_output_parses_and_empty_images_are_zero_annotation():
    """The crux: the cleaned dict feeds the typed pipeline; empties are the sampler's empty case."""
    dataset = CocoDataset.model_validate(clean_bbox_coco(_RAW))
    assert len(dataset.images) == 4
    boxed = {ann.image_id for ann in dataset.annotations}
    assert boxed == {"i1", "i2"}  # i3/i4 carry no annotation → "empty"

    # And the sampler treats i3/i4 as empties (location-split needs the location we preserved).
    result = sample_subset(
        dataset, SamplingConfig(min_per_class=10, target_empty_ratio=1.0, val_location_fraction=0.5)
    )
    assert result.num_empty_images == 2
