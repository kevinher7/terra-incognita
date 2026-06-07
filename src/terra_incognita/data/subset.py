"""Stratified subset sampler — the one-time, seeded dataset-construction policy (PLAN §5.3).

The real Caltech Camera Traps set is ~66K annotated images, ~70% empty, with an extreme
long tail. Training on all of it is both slow and pointless (the model is *deliberately*
weak — accuracy is a non-goal). This step distills a small, reproducible subset:

  1. **Per-class floor.** Keep at least ``min_per_class`` images for every class — or *all*
     of them for a rare class (badger/fox have a handful). This is what stops the tail from
     vanishing entirely.
  2. **Empty-ratio reduction.** The source is ~70% empty; we down-sample empties to
     ``target_empty_ratio`` (~20-30%) so the model still sees the camera-trap "empty" case
     without drowning in it.
  3. **Location-disjoint split.** Train/val are split by **camera location**, so no camera
     appears in both — the canonical CCT split that actually tests generalization
     (CONTEXT §9). This split is a *training* input (it feeds the policy-free COCO→YOLO
     converter); it is **not** part of dataset registration (dataset-conventions.md).

Everything is driven by a seeded :class:`random.Random` over **sorted** inputs, so the
subset and the split are byte-stable for a fixed seed — the reproducibility the
``datasets`` convention promises. The whole :class:`SamplingConfig` serializes straight
into the ``sampling_config_json`` registration tag, and the rich :class:`SubsetResult`
carries every count worth aggregating when debugging a bad dataset build later.

This module is **pure** (no S3, no MLflow) so it runs in the lean CI sync and is unit-
tested against the synthetic fixtures; the I/O that uploads + registers the subset lives in
``scripts/dataset_smoke.py`` (the heavy ``ml`` extra, not in CI).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from terra_incognita.data.coco_to_yolo import CocoDataset, Split

__all__ = [
    "SamplingConfig",
    "SubsetResult",
    "sample_subset",
    "write_subset_coco",
]


class SamplingConfig(BaseModel):
    """The stratified-subset policy — the thing that makes a dataset version reproducible.

    ``extra="forbid"`` so a typo'd key fails loudly (same rationale as ``ExperimentConfig``).
    The defaults encode CONTEXT §4: floor ~20 images/class (all for rare), empties reduced
    from ~70% to ~25%, ~20% of camera locations held out for val. Serialized verbatim into
    the ``sampling_config_json`` tag, so this object *is* the on-the-record sampling recipe.
    """

    model_config = ConfigDict(extra="forbid")

    # A human-readable name for the strategy, logged so a reader of the tag knows what the
    # numbers below mean without consulting code.
    strategy: str = "stratified_location_split"
    min_per_class: int = Field(default=20, ge=0)
    target_empty_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    val_location_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    seed: int = 42


@dataclass(frozen=True)
class SubsetResult:
    """Everything the sampler selected and learned — the fixed image list plus aggregatable stats.

    Deliberately rich (PLAN observability ethos): the per-class and per-split counts are
    exactly what you want to pivot on when a dataset build looks wrong.
    """

    image_ids: tuple[str, ...]  # the fixed, sorted subset image list
    image_splits: dict[str, Split]  # selected image id → train/val (location-disjoint)
    num_images: int
    num_empty_images: int
    num_annotations: int
    per_class_counts: dict[str, int]  # class name → # selected images containing that class
    train_locations: tuple[str, ...]
    val_locations: tuple[str, ...]
    config: SamplingConfig = field(default_factory=SamplingConfig)


def sample_subset(dataset: CocoDataset, config: SamplingConfig | None = None) -> SubsetResult:
    """Select a seeded, stratified, location-split subset of ``dataset`` (pure, deterministic).

    Returns a :class:`SubsetResult` carrying the fixed image list, the location-disjoint
    train/val split, and the counts. Determinism comes from one seeded RNG consumed in a
    fixed order over **sorted** inputs — re-running with the same seed reproduces the subset
    byte for byte.
    """
    config = config or SamplingConfig()
    rng = random.Random(config.seed)

    images_by_id = {img.id: img for img in dataset.images}
    annotations_by_image: dict[str, list[Any]] = defaultdict(list)
    for ann in dataset.annotations:
        annotations_by_image[ann.image_id].append(ann)

    category_name = {cat.id: cat.name for cat in dataset.categories}

    # Images (by id) that contain at least one annotation of each category.
    images_by_category: dict[int, set[str]] = defaultdict(set)
    for ann in dataset.annotations:
        images_by_category[ann.category_id].add(ann.image_id)

    empty_ids = sorted(img_id for img_id in images_by_id if not annotations_by_image[img_id])

    # --- 1. per-class floor: keep up to min_per_class images per class (all if fewer) ------
    selected: set[str] = set()
    for category_id in sorted(images_by_category):
        candidates = sorted(images_by_category[category_id])
        if len(candidates) <= config.min_per_class:
            chosen = candidates
        else:
            chosen = rng.sample(candidates, config.min_per_class)
        selected.update(chosen)

    # --- 2. empty-ratio reduction: add empties up to the target ratio ---------------------
    n_nonempty = len(selected)
    target_empties = _target_empty_count(n_nonempty, config.target_empty_ratio, len(empty_ids))
    if target_empties >= len(empty_ids):
        selected.update(empty_ids)  # keep all (nothing to sample away)
    elif target_empties > 0:
        selected.update(rng.sample(empty_ids, target_empties))

    selected_ids = sorted(selected)

    # --- 3. location-disjoint train/val split ---------------------------------------------
    image_splits, train_locations, val_locations = _split_by_location(
        selected_ids, images_by_id, config.val_location_fraction, rng
    )

    # --- counts (rich, for debugging a bad build) -----------------------------------------
    per_class_counts: dict[str, int] = dict.fromkeys(sorted(category_name.values()), 0)
    for category_id, image_ids in images_by_category.items():
        name = category_name[category_id]
        per_class_counts[name] = len(image_ids & selected)

    num_annotations = sum(len(annotations_by_image[img_id]) for img_id in selected_ids)
    num_empty_images = sum(1 for img_id in selected_ids if not annotations_by_image[img_id])

    return SubsetResult(
        image_ids=tuple(selected_ids),
        image_splits=image_splits,
        num_images=len(selected_ids),
        num_empty_images=num_empty_images,
        num_annotations=num_annotations,
        per_class_counts=per_class_counts,
        train_locations=train_locations,
        val_locations=val_locations,
        config=config,
    )


def _target_empty_count(n_nonempty: int, target_ratio: float, available: int) -> int:
    """How many empties to keep so empties/(empties+nonempty) ≈ ``target_ratio``.

    Solving ``e / (e + n) = r`` for ``e`` gives ``e = r·n / (1 - r)``. ``r == 1`` (all
    empty) is the degenerate case — keep every empty available.
    """
    if target_ratio >= 1.0:
        return available
    return min(available, round(target_ratio * n_nonempty / (1.0 - target_ratio)))


def _split_by_location(
    selected_ids: list[str],
    images_by_id: dict[str, Any],
    val_location_fraction: float,
    rng: random.Random,
) -> tuple[dict[str, Split], tuple[str, ...], tuple[str, ...]]:
    """Assign each selected image to train/val by its camera location (locations disjoint).

    Fails loud if any selected image has no location — you cannot do a location-disjoint
    split without one. When ≥2 locations exist, both sides are forced non-empty (Ultralytics
    needs a populated train *and* val).
    """
    location_of: dict[str, str] = {}
    missing: list[str] = []
    for image_id in selected_ids:
        location = images_by_id[image_id].location
        if location is None:
            missing.append(image_id)
        else:
            location_of[image_id] = location
    if missing:
        raise ValueError(
            f"selected images have no camera location, cannot split by location: {missing}"
        )

    locations = sorted(set(location_of.values()))
    shuffled = list(locations)
    rng.shuffle(shuffled)

    n_val = round(len(locations) * val_location_fraction)
    if len(locations) >= 2:
        n_val = max(1, min(n_val, len(locations) - 1))  # both splits non-empty
    val_locations = set(shuffled[:n_val])

    image_splits: dict[str, Split] = {
        image_id: ("val" if location_of[image_id] in val_locations else "train")
        for image_id in selected_ids
    }
    train_locations = tuple(loc for loc in locations if loc not in val_locations)
    return image_splits, train_locations, tuple(sorted(val_locations))


def write_subset_coco(source_coco_path: Path, image_ids: list[str], out_path: Path) -> Path:
    """Write a COCO file holding only ``image_ids`` (+ their annotations), faithfully.

    Filters the **raw** source JSON rather than re-serializing the typed model, so every
    original field (``area``, ``iscrowd``, ``datetime`` …) survives — the registered COCO is
    the dashboard's source of truth, so it must not be silently lossy. The full category
    list is kept as-is: the dataset's label space is defined by its categories, not by which
    classes happen to appear in the sampled images.
    """
    raw = json.loads(Path(source_coco_path).read_text(encoding="utf-8"))
    keep = set(image_ids)
    # ids are coerced to str on the typed side (CocoImage); match that here since the raw
    # file may carry int ids.
    raw["images"] = [img for img in raw["images"] if str(img["id"]) in keep]
    raw["annotations"] = [ann for ann in raw["annotations"] if str(ann["image_id"]) in keep]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return out_path
