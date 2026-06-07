set dotenv-load := true

# Show available recipes.
default:
    @just --list

# --- environment ------------------------------------------------------------
# Sync core deps + dev group (lean: no torch/ultralytics/mlflow). This is what CI runs.
sync:
    uv sync

# Sync with the heavy ML stack for training/serving slices (torch wheel parity applies).
sync-ml:
    uv sync --extra ml

# --- quality gates (mirror CI) ----------------------------------------------
lint:
    uv run ruff check .

fmt:
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

typecheck:
    uv run ty check

test:
    uv run pytest

# Everything CI checks, locally.
check: lint fmt-check typecheck test

# --- pipeline steps (stubs in slice 1) --------------------------------------
download:
    uv run terra-incognita download

subset:
    uv run terra-incognita subset

# Convert a COCO file + images into the Ultralytics YOLO layout (placeholder fraction split).
convert coco images out:
    uv run terra-incognita convert --coco {{coco}} --images {{images}} --out {{out}}

upload:
    uv run terra-incognita upload

register-dataset:
    uv run terra-incognita register-dataset

# Train from a versioned experiment config (an experiment is a file, not env state).
train config="configs/baseline.yaml":
    uv run terra-incognita train --config {{config}}

package:
    uv run terra-incognita package

serve:
    uv run terra-incognita serve

smoke:
    uv run terra-incognita smoke

# Emit a registry-validated training.run wide event (observability smoke).
demo-event:
    uv run terra-incognita demo-event

# --- local stack (docker-compose lands in slice 2) --------------------------
up:
    @echo "[stub] local stack (mlflow + floci + SigNoz) lands in slice 2."

down:
    @echo "[stub] local stack teardown lands in slice 2."
