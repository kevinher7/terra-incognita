"""Stratified subset sampler tests — the seeded dataset-construction policy (issue #4).

What must hold (PLAN §5.3, CONTEXT §4, dataset-conventions.md acceptance):
  1. Deterministic for a fixed seed (the reproducibility the `datasets` convention promises).
  2. Per-class floor honored — at least ``min_per_class`` images per class, or all if rarer.
  3. Empty-ratio reduced toward ``target_empty_ratio`` (source is ~70% empty).
  4. Train/val camera locations are **disjoint** and both splits are non-empty.
  5. The written subset COCO is faithful and re-parses (it is the registered source of truth).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terra_incognita.data.coco_to_yolo import CocoDataset
from terra_incognita.data.subset import SamplingConfig, sample_subset, write_subset_coco
from tests.fixtures.synthetic import generate_synthetic_dataset


def _make_dataset(
    *,
    nonempty_per_class: dict[int, int],
    num_empty: int,
    num_locations: int = 4,
    categories: dict[int, str] | None = None,
) -> CocoDataset:
    """Build a COCO dataset with precise per-class / empty / location structure (test control).

    Each non-empty image carries exactly one annotation of a single class; images are spread
    round-robin across ``num_locations`` cameras so the location split has something to do.
    """
    categories = categories or {cid: f"class-{cid}" for cid in nonempty_per_class}
    images: list[dict] = []
    annotations: list[dict] = []
    idx = 0
    ann_id = 0
    for category_id, count in nonempty_per_class.items():
        for _ in range(count):
            image_id = f"img-{idx:03d}"
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{image_id}.png",
                    "width": 640,
                    "height": 480,
                    "location": f"loc-{idx % num_locations}",
                }
            )
            annotations.append(
                {
                    "id": f"ann-{ann_id}",
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [1, 1, 10, 10],
                }
            )
            idx += 1
            ann_id += 1
    for _ in range(num_empty):
        image_id = f"img-{idx:03d}"
        images.append(
            {
                "id": image_id,
                "file_name": f"{image_id}.png",
                "width": 640,
                "height": 480,
                "location": f"loc-{idx % num_locations}",
            }
        )
        idx += 1
    return CocoDataset.model_validate(
        {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": cid, "name": name} for cid, name in categories.items()],
        }
    )


# ---------------------------------------------------------------------------
# 1. Determinism — same seed → byte-identical selection + split.
# ---------------------------------------------------------------------------
def test_sample_subset_is_deterministic_for_a_fixed_seed():
    dataset = _make_dataset(nonempty_per_class={1: 10, 2: 8}, num_empty=12)
    config = SamplingConfig(seed=123, min_per_class=3)
    first = sample_subset(dataset, config)
    second = sample_subset(dataset, config)
    assert first == second


def test_different_seeds_can_change_the_selection():
    # With more candidates than the floor, the seeded sample should differ across seeds.
    dataset = _make_dataset(nonempty_per_class={1: 30}, num_empty=0)
    a = sample_subset(dataset, SamplingConfig(seed=1, min_per_class=5, val_location_fraction=0.0))
    b = sample_subset(dataset, SamplingConfig(seed=2, min_per_class=5, val_location_fraction=0.0))
    assert a.image_ids != b.image_ids


# ---------------------------------------------------------------------------
# 2. Per-class floor — at least min_per_class per class, all of a rare class.
# ---------------------------------------------------------------------------
def test_per_class_floor_caps_common_classes_and_keeps_all_of_rare():
    # class 1 is common (5 imgs), class 2 is rare (1 img). Floor of 2.
    dataset = _make_dataset(
        nonempty_per_class={1: 5, 2: 1}, num_empty=0, categories={1: "common", 2: "rare"}
    )
    result = sample_subset(dataset, SamplingConfig(seed=7, min_per_class=2))
    # Common is capped at the floor; rare keeps all (fewer than the floor).
    assert result.per_class_counts["common"] == 2
    assert result.per_class_counts["rare"] == 1


# ---------------------------------------------------------------------------
# 3. Empty-ratio reduction — down-sample the empties toward the target.
# ---------------------------------------------------------------------------
def test_empty_ratio_is_reduced_toward_target():
    # 8 non-empty + 20 empty → source empty ratio ~0.71; target 0.25.
    dataset = _make_dataset(nonempty_per_class={1: 4, 2: 4}, num_empty=20)
    result = sample_subset(
        dataset, SamplingConfig(seed=3, min_per_class=100, target_empty_ratio=0.25)
    )

    source_ratio = 20 / 28
    result_ratio = result.num_empty_images / result.num_images
    assert result_ratio < source_ratio
    assert result_ratio == pytest.approx(0.25, abs=0.05)
    # e = round(0.25 * 8 / 0.75) = 3
    assert result.num_empty_images == 3


# ---------------------------------------------------------------------------
# 4. Location-disjoint split — train/val cameras never overlap, both non-empty.
# ---------------------------------------------------------------------------
def test_train_and_val_locations_are_disjoint_and_both_populated():
    dataset = _make_dataset(nonempty_per_class={1: 12, 2: 9}, num_empty=9, num_locations=5)
    result = sample_subset(dataset, SamplingConfig(seed=42))

    train_locs, val_locs = set(result.train_locations), set(result.val_locations)
    assert train_locs and val_locs  # both non-empty
    assert train_locs.isdisjoint(val_locs)

    # Every image's split agrees with its location's side — the actual disjointness guarantee.
    images_by_id = {img.id: img for img in dataset.images}
    for image_id, split in result.image_splits.items():
        location = images_by_id[image_id].location
        assert location in (val_locs if split == "val" else train_locs)
    assert set(result.image_splits.values()) == {"train", "val"}


def test_missing_location_on_a_selected_image_is_rejected():
    dataset = CocoDataset.model_validate(
        {
            "images": [
                {"id": "i1", "file_name": "i1.png", "width": 100, "height": 100},  # no location
            ],
            "annotations": [{"id": "a1", "image_id": "i1", "category_id": 1, "bbox": [1, 1, 5, 5]}],
            "categories": [{"id": 1, "name": "x"}],
        }
    )
    with pytest.raises(ValueError, match="no camera location"):
        sample_subset(dataset, SamplingConfig(min_per_class=10))


# ---------------------------------------------------------------------------
# 5. write_subset_coco — faithful, re-parseable, only the selected images.
# ---------------------------------------------------------------------------
def test_write_subset_coco_keeps_only_selected_and_reparses(tmp_path: Path):
    fixture = generate_synthetic_dataset(tmp_path / "src", seed=9)
    dataset = CocoDataset.from_path(fixture.coco_path)
    result = sample_subset(dataset, SamplingConfig(seed=9))

    out = write_subset_coco(fixture.coco_path, list(result.image_ids), tmp_path / "subset.json")
    parsed = CocoDataset.from_path(out)

    selected = set(result.image_ids)
    assert {img.id for img in parsed.images} == selected
    assert all(ann.image_id in selected for ann in parsed.annotations)
    assert len(parsed.images) == result.num_images
    assert len(parsed.annotations) == result.num_annotations
    # The full label space is preserved, even classes that didn't make the subset.
    assert len(parsed.categories) == len(dataset.categories)
