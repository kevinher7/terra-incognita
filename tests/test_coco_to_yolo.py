"""COCO→YOLO converter tests — the riskiest pure logic, tested exhaustively (issue #2).

Three things must hold:
  1. The normalization math matches hand-computed boxes (``cx=(x+w/2)/W`` etc.).
  2. Round-trip: every COCO annotation becomes exactly one YOLO label line, with class
     indices contiguous from 0 and split into the right Ultralytics directories.
  3. The YOLO-index → COCO-``category_id`` map is emitted and reloadable (the serving
     contract artifact, .plans/contracts/serving-io.md).
"""

from __future__ import annotations

import math

import pytest

from terra_incognita.data.coco_to_yolo import (
    CategoryIndex,
    CocoCategory,
    CocoDataset,
    YoloLabel,
    convert_coco_to_yolo,
    load_category_index,
    normalize_bbox,
)
from tests.fixtures.synthetic import generate_synthetic_dataset

TOL = 1e-9


# ---------------------------------------------------------------------------
# 1. Normalization math — asserted against boxes computed by hand.
# ---------------------------------------------------------------------------
def test_normalize_bbox_hand_computed():
    # W=800, H=600, box=[100,50,200,100]:
    #   cx=(100+100)/800=0.25  cy=(50+50)/600=1/6  w=200/800=0.25  h=100/600=1/6
    cx, cy, w, h = normalize_bbox([100, 50, 200, 100], 800, 600)
    assert cx == pytest.approx(0.25, abs=TOL)
    assert cy == pytest.approx(1 / 6, abs=TOL)
    assert w == pytest.approx(0.25, abs=TOL)
    assert h == pytest.approx(1 / 6, abs=TOL)


def test_normalize_bbox_full_image_is_centered_unit_box():
    # A box covering the whole image → center (0.5, 0.5), size (1, 1).
    assert normalize_bbox([0, 0, 640, 480], 640, 480) == pytest.approx((0.5, 0.5, 1.0, 1.0))


def test_normalize_bbox_corner_box():
    # Top-left quarter of a 1000x1000 image → center (0.125, 0.125), size (0.25, 0.25).
    assert normalize_bbox([0, 0, 250, 250], 1000, 1000) == pytest.approx((0.125, 0.125, 0.25, 0.25))


def test_normalize_bbox_rejects_nonpositive_dims():
    with pytest.raises(ValueError, match="dimensions must be positive"):
        normalize_bbox([0, 0, 10, 10], 0, 100)


def test_yolo_label_format():
    line = YoloLabel(2, 0.25, 1 / 6, 0.25, 1 / 6).format()
    assert line == "2 0.250000 0.166667 0.250000 0.166667"


# ---------------------------------------------------------------------------
# 2. Category index — contiguous 0..N-1, sorted by category_id, reversible.
# ---------------------------------------------------------------------------
def test_category_index_is_contiguous_and_sorted_by_category_id():
    # Intentionally out-of-order, sparse category ids.
    categories = [
        CocoCategory(id=12, name="bobcat"),
        CocoCategory(id=3, name="raccoon"),
        CocoCategory(id=7, name="coyote"),
    ]
    index = CategoryIndex.from_categories(categories)

    # Contiguous from 0, assigned by ascending category_id.
    assert [e.index for e in index.classes] == [0, 1, 2]
    assert index.index_to_category_id == {0: 3, 1: 7, 2: 12}
    assert index.category_id_to_index == {3: 0, 7: 1, 12: 2}
    assert index.names == {0: "raccoon", 1: "coyote", 2: "bobcat"}


def test_category_index_round_trips_through_dict():
    categories = [CocoCategory(id=7, name="coyote"), CocoCategory(id=3, name="raccoon")]
    index = CategoryIndex.from_categories(categories)
    assert CategoryIndex.from_dict(index.to_dict()) == index


def test_load_category_index_rejects_incompatible_schema():
    with pytest.raises(ValueError, match="schema_version"):
        CategoryIndex.from_dict({"schema_version": 999, "classes": []})


# ---------------------------------------------------------------------------
# 3. End-to-end conversion on synthetic fixtures — round-trip + layout + map artifact.
# ---------------------------------------------------------------------------
def test_convert_produces_ultralytics_layout(tmp_path):
    fixture = generate_synthetic_dataset(tmp_path / "src", seed=7)
    out = tmp_path / "yolo"
    result = convert_coco_to_yolo(fixture.coco_path, fixture.images_dir, out, fixture.image_splits)

    # The four Ultralytics directories + data.yaml + the category map all exist.
    for split in ("train", "val"):
        assert (out / "images" / split).is_dir()
        assert (out / "labels" / split).is_dir()
    assert result.data_yaml_path.exists()
    assert result.category_map_path.exists()

    # Every image landed in its assigned split, as both an image and a label file.
    for image_id, split in fixture.image_splits.items():
        assert (out / "images" / split / f"{image_id}.png").exists()
        assert (out / "labels" / split / f"{image_id}.txt").exists()


def test_round_trip_every_annotation_becomes_one_label_line(tmp_path):
    fixture = generate_synthetic_dataset(tmp_path / "src", seed=11)
    out = tmp_path / "yolo"
    result = convert_coco_to_yolo(fixture.coco_path, fixture.images_dir, out, fixture.image_splits)

    # Count label lines across every emitted .txt file.
    total_lines = 0
    seen_indices: set[int] = set()
    for label_file in out.glob("labels/*/*.txt"):
        for raw in label_file.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            total_lines += 1
            seen_indices.add(int(raw.split()[0]))

    # Exactly one YOLO line per COCO annotation — no drops, no duplicates.
    assert total_lines == fixture.num_annotations
    assert result.num_annotations == fixture.num_annotations
    assert sum(result.annotations_per_split.values()) == fixture.num_annotations

    # Class indices are contiguous from 0 (a subset of {0,1,2} for the 3-class fixture).
    assert seen_indices, "fixture must contain at least one annotation"
    assert seen_indices <= {0, 1, 2}
    assert min(seen_indices) == 0
    # No fixture annotation pokes outside its image, so nothing is flagged out-of-bounds.
    assert result.num_out_of_bounds == 0


def test_label_values_match_hand_computed_anchor(tmp_path):
    fixture = generate_synthetic_dataset(tmp_path / "src", seed=3)
    out = tmp_path / "yolo"
    convert_coco_to_yolo(fixture.coco_path, fixture.images_dir, out, fixture.image_splits)

    anchor = fixture.anchor
    split = fixture.image_splits[anchor.image_stem]
    label_text = (out / "labels" / split / f"{anchor.image_stem}.txt").read_text(encoding="utf-8")
    lines = [line for line in label_text.splitlines() if line.strip()]
    assert len(lines) == 1

    parts = lines[0].split()
    assert int(parts[0]) == anchor.yolo_index
    values = [float(p) for p in parts[1:]]
    for got, expected in zip(values, anchor.expected_norm, strict=True):
        assert math.isclose(got, expected, abs_tol=1e-6)


def test_category_map_is_emitted_and_reloadable(tmp_path):
    fixture = generate_synthetic_dataset(tmp_path / "src", seed=5)
    out = tmp_path / "yolo"
    result = convert_coco_to_yolo(fixture.coco_path, fixture.images_dir, out, fixture.image_splits)

    reloaded = load_category_index(result.category_map_path)
    assert reloaded == result.category_index
    # The serving contract: index → real COCO category_id (sparse 3/7/12 fixture ids).
    assert reloaded.index_to_category_id == {0: 3, 1: 7, 2: 12}


def test_data_yaml_has_paths_and_indexed_names(tmp_path):
    import yaml

    fixture = generate_synthetic_dataset(tmp_path / "src", seed=9)
    out = tmp_path / "yolo"
    result = convert_coco_to_yolo(fixture.coco_path, fixture.images_dir, out, fixture.image_splits)

    data = yaml.safe_load(result.data_yaml_path.read_text(encoding="utf-8"))
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["nc"] == 3
    assert data["names"] == {0: "raccoon", 1: "coyote", 2: "bobcat"}
    assert data["path"] == str(out.resolve())


# ---------------------------------------------------------------------------
# Input validation — fail loud before writing a half-built layout.
# ---------------------------------------------------------------------------
def test_missing_split_assignment_is_rejected(tmp_path):
    fixture = generate_synthetic_dataset(tmp_path / "src", seed=2)
    splits = dict(fixture.image_splits)
    splits.popitem()  # drop one image's split assignment
    with pytest.raises(ValueError, match="missing a train/val split"):
        convert_coco_to_yolo(fixture.coco_path, fixture.images_dir, tmp_path / "yolo", splits)


def test_unknown_category_is_rejected(tmp_path):
    # Build from a raw COCO dict (model_validate) — the converter's real input shape.
    dataset = CocoDataset.model_validate(
        {
            "images": [{"id": "i1", "file_name": "i1.png", "width": 100, "height": 100}],
            "annotations": [
                {"id": "a1", "image_id": "i1", "category_id": 99, "bbox": [1, 1, 5, 5]}
            ],
            "categories": [{"id": 3, "name": "raccoon"}],
        }
    )
    with pytest.raises(ValueError, match="unknown category ids"):
        convert_coco_to_yolo(
            dataset, tmp_path / "noimg", tmp_path / "yolo", {"i1": "train"}, copy_images=False
        )
