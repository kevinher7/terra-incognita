"""Seeded synthetic fixture generator — shape-on-noise images + an exactly-matching COCO.

PLAN §10/§13.3: rather than commit real camera-trap images to git (git-LFS is a noted
fallback, not the plan), every test materializes ~15 tiny shape-on-noise images and the
COCO JSON that *exactly* describes them into a tmp dir at test time. "Exactly matching"
is structural: each annotation's ``bbox`` is the very box we drew, so the COCO→YOLO
normalization has a known ground truth to be asserted against.

The generator is fully seeded (``random.Random(seed)``), so two runs produce byte-stable
boxes — which is what lets the round-trip and math assertions be deterministic.

Pillow is a *dev/test-only* dependency (see pyproject ``dependency-groups.dev``); the
production converter in ``src/`` never imports it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from terra_incognita.data.coco_to_yolo import Split

# Deliberately sparse / non-contiguous COCO category ids, so any fixture-driven test
# exercises the "arbitrary category_id → contiguous 0..N-1 YOLO index" remap (sorted by
# id → raccoon=0, coyote=1, bobcat=2) rather than an identity mapping that would hide bugs.
CATEGORIES: tuple[tuple[int, str], ...] = ((3, "raccoon"), (7, "coyote"), (12, "bobcat"))

# A fixed, round-numbered "anchor" annotation in image 0 so a test can assert the
# normalized box against numbers computed by hand (W=800, H=600, box=[100,50,200,100]):
#   cx=(100+100)/800=0.25  cy=(50+50)/600=0.166667  w=200/800=0.25  h=100/600=0.166667
ANCHOR_WIDTH = 800
ANCHOR_HEIGHT = 600
ANCHOR_BBOX: tuple[float, float, float, float] = (100.0, 50.0, 200.0, 100.0)
ANCHOR_CATEGORY_ID = 3  # raccoon → YOLO index 0


@dataclass(frozen=True)
class Anchor:
    """The hand-computable annotation, surfaced so a test can assert exact normalized values."""

    image_stem: str
    category_id: int
    yolo_index: int
    bbox: tuple[float, float, float, float]
    image_width: int
    image_height: int
    expected_norm: tuple[float, float, float, float]


@dataclass(frozen=True)
class SyntheticDataset:
    """Handles to a generated fixture: the COCO file, the image dir, and the chosen split."""

    root: Path
    images_dir: Path
    coco_path: Path
    image_splits: dict[str, Split]
    num_images: int
    num_annotations: int
    anchor: Anchor


def generate_synthetic_dataset(
    out_dir: Path,
    *,
    seed: int = 1234,
    num_images: int = 15,
) -> SyntheticDataset:
    """Draw ``num_images`` shape-on-noise images + write the COCO JSON that describes them.

    Layout under ``out_dir``: ``images/<id>.png`` and ``annotations.json``. Returns the
    paths plus a deterministic train/val split and the :class:`Anchor` for math assertions.
    """
    rng = random.Random(seed)
    images_dir = Path(out_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    annotations: list[dict] = []
    image_splits: dict[str, Split] = {}
    ann_id = 0

    for i in range(num_images):
        image_id = f"img-{i:03d}"
        # Image 0 carries the fixed anchor box on the fixed anchor canvas; the rest get
        # varied dimensions so normalization is exercised across many W/H denominators.
        if i == 0:
            width, height = ANCHOR_WIDTH, ANCHOR_HEIGHT
        else:
            width = rng.choice([512, 640, 800, 1024])
            height = rng.choice([384, 480, 600, 768])

        canvas = Image.effect_noise((width, height), 48).convert("RGB")
        draw = ImageDraw.Draw(canvas)

        boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        if i == 0:
            boxes.append((ANCHOR_CATEGORY_ID, ANCHOR_BBOX))
        # Every 4th image is a deliberate empty/background frame (no animal, no box) — the
        # camera-trap "empty" case the model must also see (PLAN §5.3 empty ratio).
        elif i % 4 == 0:
            boxes = []
        else:
            for _ in range(rng.randint(1, 3)):
                boxes.append(_random_box(rng, width, height))

        for category_id, (x, y, w, h) in boxes:
            _draw_shape(rng, draw, x, y, w, h)
            annotations.append(
                {
                    "id": f"ann-{ann_id:04d}",
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        file_name = f"{image_id}.png"
        canvas.save(images_dir / file_name)
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        # Deterministic, both-splits-non-empty: every 3rd image to val.
        image_splits[image_id] = "val" if i % 3 == 0 else "train"

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid, "name": name} for cid, name in CATEGORIES],
    }
    coco_path = Path(out_dir) / "annotations.json"
    coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    cx = (ANCHOR_BBOX[0] + ANCHOR_BBOX[2] / 2) / ANCHOR_WIDTH
    cy = (ANCHOR_BBOX[1] + ANCHOR_BBOX[3] / 2) / ANCHOR_HEIGHT
    anchor = Anchor(
        image_stem="img-000",
        category_id=ANCHOR_CATEGORY_ID,
        yolo_index=0,
        bbox=ANCHOR_BBOX,
        image_width=ANCHOR_WIDTH,
        image_height=ANCHOR_HEIGHT,
        expected_norm=(cx, cy, ANCHOR_BBOX[2] / ANCHOR_WIDTH, ANCHOR_BBOX[3] / ANCHOR_HEIGHT),
    )
    return SyntheticDataset(
        root=Path(out_dir),
        images_dir=images_dir,
        coco_path=coco_path,
        image_splits=image_splits,
        num_images=len(images),
        num_annotations=len(annotations),
        anchor=anchor,
    )


def _random_box(
    rng: random.Random, width: int, height: int
) -> tuple[int, tuple[float, float, float, float]]:
    """A category id + an in-bounds ``[x, y, w, h]`` box (margins keep it off the edges)."""
    category_id = rng.choice([cid for cid, _ in CATEGORIES])
    w = rng.randint(width // 8, width // 3)
    h = rng.randint(height // 8, height // 3)
    x = rng.randint(0, width - w)
    y = rng.randint(0, height - h)
    return category_id, (float(x), float(y), float(w), float(h))


def _draw_shape(
    rng: random.Random, draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float
) -> None:
    """Paint one filled rectangle or ellipse whose bounding box is exactly ``[x, y, w, h]``."""
    box = [x, y, x + w, y + h]
    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    if rng.random() < 0.5:
        draw.rectangle(box, fill=color)
    else:
        draw.ellipse(box, fill=color)
