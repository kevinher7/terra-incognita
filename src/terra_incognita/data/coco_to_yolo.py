"""COCO → YOLO (Ultralytics) converter — the riskiest pure logic in the repo.

Two transforms live here, and they are deliberately kept apart from the rest of the
pipeline so they can be unit-tested exhaustively against synthetic fixtures (PLAN §10,
§13.3):

1. **Box transform.** COCO stores absolute top-left ``xywh`` (pixels). YOLO/Ultralytics
   wants *normalized center* ``cx cy w h`` in ``[0, 1]``::

       cx = (x + w/2) / W      w_norm = w / W
       cy = (y + h/2) / H      h_norm = h / H

   This is a **training-internal** transform — distinct from the ``xyxy`` box format
   stored/served end-to-end (see .plans/contracts/bbox-format.md).

2. **Class-index transform.** YOLO needs a *contiguous* ``0..N-1`` class index, but COCO
   ``category_id``s are arbitrary and sparse. We assign indices by sorting categories on
   ``category_id`` (deterministic, independent of the order categories happen to appear
   in the file) and **persist the index → ``category_id`` map as an artifact**. Serving
   reverses it (model emits an index → real ``category_id``), so this map MUST match what
   serving loads (see .plans/contracts/serving-io.md).

The converter is **policy-free**: the train/val split is an *input* (a mapping of image id
→ split). Deciding the split — by camera location, seeded — is the subset step's job
(PLAN §5.3), not the mechanical converter's. It returns a rich :class:`ConversionResult`
so the future ``convert`` CLI step can log counts as a wide event without re-deriving them.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CategoryIndex",
    "ClassEntry",
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

# The two Ultralytics split directories. A Literal (not a free string)
Split = Literal["train", "val"]
SPLITS: tuple[Split, ...] = ("train", "val")

# Persisted map schema version — bump if the on-disk shape of the category map changes,
# so serving can refuse a map it doesn't understand rather than silently mis-join classes.
CATEGORY_MAP_SCHEMA_VERSION = 1
CATEGORY_MAP_FILENAME = "index_to_category_id.json"
DATA_YAML_FILENAME = "data.yaml"

# Label coordinates are written with this many decimals: enough to round-trip a box on a
# multi-thousand-pixel image without visible drift, few enough to keep label files small.
_COORD_PRECISION = 6


# ---------------------------------------------------------------------------
# COCO input models. These describe ONLY the fields the converter reads; a real COCO file
# carries far more (licenses, info, segmentation, ...). `extra="ignore"` lets us parse it
# without enumerating everything, and `coerce_numbers_to_str` tolerates COCO files that
# use integer image ids (CCT uses string UUIDs; the spec allows ints) — ids are join keys,
# never arithmetic, so string is the safe normal form.
# ---------------------------------------------------------------------------
class _CocoModel(BaseModel):
    model_config = ConfigDict(extra="ignore", coerce_numbers_to_str=True)


class CocoImage(_CocoModel):
    id: str
    file_name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class CocoAnnotation(_CocoModel):
    image_id: str
    # category_id stays an int — it is the real COCO class id and the serving join key.
    category_id: int
    # COCO bbox is absolute top-left [x, y, w, h] in pixels.
    bbox: Annotated[list[float], Field(min_length=4, max_length=4)]


class CocoCategory(_CocoModel):
    id: int
    name: str


class CocoDataset(_CocoModel):
    images: list[CocoImage]
    annotations: list[CocoAnnotation]
    categories: list[CocoCategory]

    @classmethod
    def from_path(cls, path: Path) -> CocoDataset:
        """Parse a COCO JSON file into the typed subset the converter needs."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Box transform (the hand-checkable math).
# ---------------------------------------------------------------------------
def normalize_bbox(
    bbox_xywh: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Absolute top-left ``[x, y, w, h]`` → normalized center ``(cx, cy, w, h)`` in ``[0, 1]``.

    The single source of the COCO→YOLO box math. Kept pure (no I/O, no clamping) so it can
    be asserted against hand-computed boxes; out-of-image annotations are *surfaced* (a
    count in :class:`ConversionResult`), not silently clamped — cleaning bad boxes is a
    dataset concern, not a transform concern.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image dimensions must be positive, got {image_width}x{image_height}")
    x, y, w, h = bbox_xywh
    cx = (x + w / 2) / image_width
    cy = (y + h / 2) / image_height
    return (cx, cy, w / image_width, h / image_height)


def _is_out_of_bounds(bbox_xywh: list[float], image_width: int, image_height: int) -> bool:
    """True if the box pokes outside the image — recorded for debuggability, never fixed here."""
    x, y, w, h = bbox_xywh
    eps = 1e-6
    return x < -eps or y < -eps or (x + w) > image_width + eps or (y + h) > image_height + eps


@dataclass(frozen=True)
class YoloLabel:
    """One YOLO label line: a class index plus a normalized-center box."""

    class_index: int
    cx: float
    cy: float
    w: float
    h: float

    def format(self) -> str:
        """``"<class> <cx> <cy> <w> <h>"`` — the exact text of one line in a label file."""
        return (
            f"{self.class_index} "
            f"{self.cx:.{_COORD_PRECISION}f} {self.cy:.{_COORD_PRECISION}f} "
            f"{self.w:.{_COORD_PRECISION}f} {self.h:.{_COORD_PRECISION}f}"
        )


# ---------------------------------------------------------------------------
# Class-index transform (the serving-critical map).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassEntry:
    """One class: its contiguous YOLO ``index``, real COCO ``category_id``, and ``name``."""

    index: int
    category_id: int
    name: str


@dataclass(frozen=True)
class CategoryIndex:
    """The YOLO contiguous-index ↔ COCO ``category_id`` map (the serving artifact).

    Built by sorting COCO categories on ``category_id`` and numbering them ``0..N-1``, so
    the mapping is reproducible no matter what order categories appear in the COCO file.
    """

    classes: tuple[ClassEntry, ...]

    @classmethod
    def from_categories(cls, categories: list[CocoCategory]) -> CategoryIndex:
        ordered = sorted(categories, key=lambda c: c.id)
        return cls(
            classes=tuple(
                ClassEntry(index=i, category_id=c.id, name=c.name) for i, c in enumerate(ordered)
            )
        )

    @property
    def index_to_category_id(self) -> dict[int, int]:
        """YOLO index → real COCO ``category_id`` (what serving applies to model output)."""
        return {e.index: e.category_id for e in self.classes}

    @property
    def category_id_to_index(self) -> dict[int, int]:
        """Real COCO ``category_id`` → YOLO index (what the converter applies to labels)."""
        return {e.category_id: e.index for e in self.classes}

    @property
    def names(self) -> dict[int, str]:
        """YOLO index → class name, as Ultralytics' ``data.yaml`` ``names`` wants it."""
        return {e.index: e.name for e in self.classes}

    def to_dict(self) -> dict[str, Any]:
        """Json-able artifact body. The ``classes`` list is fully self-describing/reloadable."""
        return {
            "schema_version": CATEGORY_MAP_SCHEMA_VERSION,
            "classes": [
                {"index": e.index, "category_id": e.category_id, "name": e.name}
                for e in self.classes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoryIndex:
        version = data.get("schema_version")
        if version != CATEGORY_MAP_SCHEMA_VERSION:
            raise ValueError(
                f"category map schema_version {version!r} != expected "
                f"{CATEGORY_MAP_SCHEMA_VERSION} (artifact written by an incompatible version)"
            )
        return cls(
            classes=tuple(
                ClassEntry(index=e["index"], category_id=e["category_id"], name=e["name"])
                for e in data["classes"]
            )
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_category_index(path: Path) -> CategoryIndex:
    """Reload the persisted index → ``category_id`` map (serving + tests use this)."""
    return CategoryIndex.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# The conversion result — deliberately rich (PLAN observability ethos: capture anything
# that could help debug a bad dataset build later, even remotely).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConversionResult:
    """Everything the conversion produced and learned — paths plus aggregatable counts."""

    output_dir: Path
    data_yaml_path: Path
    category_map_path: Path
    category_index: CategoryIndex

    num_categories: int
    num_images: int
    num_annotations: int  # total COCO annotations converted == total YOLO label lines written
    images_per_split: dict[Split, int] = field(default_factory=dict)
    annotations_per_split: dict[Split, int] = field(default_factory=dict)
    num_empty_images: int = 0  # images with zero annotations → an empty label file
    num_out_of_bounds: int = 0  # boxes poking outside the image (kept, but flagged)


def split_by_fraction(image_ids: list[str], val_fraction: float, seed: int) -> dict[str, Split]:
    """Convenience split for manual/smoke use: deterministically assign ``val_fraction`` to val.

    A *placeholder* split policy. The real pipeline splits by camera **location** (PLAN
    §5.3) so train/val are location-disjoint; that lives in the subset step. This exists
    only so the ``convert`` CLI and the fixture smoke path have a reproducible split to
    hand the (policy-free) converter.
    """
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError(f"val_fraction must be in [0, 1], got {val_fraction}")
    # Hash by id+seed so the assignment is deterministic and independent of input order.
    ordered = sorted(image_ids)
    n_val = round(len(ordered) * val_fraction)
    rng_order = sorted(ordered, key=lambda i: _stable_hash(f"{seed}:{i}"))
    val = set(rng_order[:n_val])
    return {image_id: ("val" if image_id in val else "train") for image_id in ordered}


def _stable_hash(text: str) -> int:
    """A process-independent hash (Python's ``hash`` is salted per process)."""
    import hashlib

    return int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")


def convert_coco_to_yolo(
    coco: CocoDataset | Path,
    images_dir: Path,
    output_dir: Path,
    image_splits: dict[str, Split],
    *,
    copy_images: bool = True,
) -> ConversionResult:
    """Convert a COCO dataset into the Ultralytics on-disk layout.

    Emits ``images/{train,val}`` + ``labels/{train,val}`` + ``data.yaml`` +
    ``index_to_category_id.json`` under ``output_dir``. ``image_splits`` maps every image
    id to its split (the converter is policy-free; see module docstring). Returns a
    :class:`ConversionResult` with paths and counts for logging/assertions.
    """
    dataset = coco if isinstance(coco, CocoDataset) else CocoDataset.from_path(coco)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    category_index = CategoryIndex.from_categories(dataset.categories)
    cat_to_index = category_index.category_id_to_index

    # Fail loud on any structural mismatch BEFORE writing files — a half-written layout is
    # worse than a clear error. Every image must have a split, and every annotation must
    # reference a known image and a known category.
    images_by_id = {img.id: img for img in dataset.images}
    _validate_inputs(dataset, images_by_id, image_splits, cat_to_index)

    annotations_by_image: dict[str, list[CocoAnnotation]] = {img_id: [] for img_id in images_by_id}
    for ann in dataset.annotations:
        annotations_by_image[ann.image_id].append(ann)

    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    images_per_split: dict[Split, int] = {s: 0 for s in SPLITS}
    annotations_per_split: dict[Split, int] = {s: 0 for s in SPLITS}
    num_empty_images = 0
    num_out_of_bounds = 0

    for image in dataset.images:
        split = image_splits[image.id]
        stem = Path(image.file_name).stem
        dest_image_name = Path(image.file_name).name

        if copy_images:
            shutil.copyfile(
                images_dir / image.file_name, output_dir / "images" / split / dest_image_name
            )

        annotations = annotations_by_image[image.id]
        lines: list[str] = []
        for ann in annotations:
            if _is_out_of_bounds(ann.bbox, image.width, image.height):
                num_out_of_bounds += 1
            cx, cy, w, h = normalize_bbox(ann.bbox, image.width, image.height)
            label = YoloLabel(cat_to_index[ann.category_id], cx, cy, w, h)
            lines.append(label.format())

        # Empty label file for an image with no boxes — Ultralytics reads it as a pure
        # background/negative example (the camera-trap "empty" case), and it keeps the
        # round-trip count honest (one .txt per image).
        label_path = output_dir / "labels" / split / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        images_per_split[split] += 1
        annotations_per_split[split] += len(annotations)
        if not annotations:
            num_empty_images += 1

    data_yaml_path = _write_data_yaml(output_dir, category_index)
    category_map_path = output_dir / CATEGORY_MAP_FILENAME
    category_index.save(category_map_path)

    return ConversionResult(
        output_dir=output_dir,
        data_yaml_path=data_yaml_path,
        category_map_path=category_map_path,
        category_index=category_index,
        num_categories=len(category_index.classes),
        num_images=len(dataset.images),
        num_annotations=len(dataset.annotations),
        images_per_split=images_per_split,
        annotations_per_split=annotations_per_split,
        num_empty_images=num_empty_images,
        num_out_of_bounds=num_out_of_bounds,
    )


def _validate_inputs(
    dataset: CocoDataset,
    images_by_id: dict[str, CocoImage],
    image_splits: dict[str, Split],
    cat_to_index: dict[int, int],
) -> None:
    """Reject structurally broken inputs up front, with a message that names what's wrong."""
    missing_splits = sorted(img_id for img_id in images_by_id if img_id not in image_splits)
    if missing_splits:
        raise ValueError(f"images missing a train/val split assignment: {missing_splits}")

    bad_split_values = sorted({s for s in image_splits.values() if s not in SPLITS})
    if bad_split_values:
        raise ValueError(f"split values must be one of {SPLITS}, got {bad_split_values}")

    unknown_images = sorted(
        {a.image_id for a in dataset.annotations if a.image_id not in images_by_id}
    )
    if unknown_images:
        raise ValueError(f"annotations reference unknown image ids: {unknown_images}")

    unknown_categories = sorted(
        {a.category_id for a in dataset.annotations if a.category_id not in cat_to_index}
    )
    if unknown_categories:
        raise ValueError(f"annotations reference unknown category ids: {unknown_categories}")


def _write_data_yaml(output_dir: Path, category_index: CategoryIndex) -> Path:
    """Write the Ultralytics ``data.yaml`` (paths relative to ``path``; index→name names)."""
    content = {
        # Absolute root so a run launched from anywhere resolves the layout (the GPU box
        # materializes this from S3 into an arbitrary dir — never assume the CWD).
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(category_index.classes),
        "names": category_index.names,
    }
    data_yaml_path = output_dir / DATA_YAML_FILENAME
    data_yaml_path.write_text(
        yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return data_yaml_path
