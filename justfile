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

# --- local stack (docker-compose) -------------------------------------------
# SigNoz runs from its own pinned, unmodified official compose (true prod parity with the
# Hetzner deploy). We fetch it once into a gitignored dir rather than hand-vendoring its
# ClickHouse/collector configs — that would only drift from upstream.
SIGNOZ_VERSION := "v0.127.0"
SIGNOZ_DIR := ".signoz"
SIGNOZ_COMPOSE := SIGNOZ_DIR / "deploy/docker/docker-compose.yaml"

# Bring up the whole local stack: mlflow + floci (this repo) AND always-on SigNoz.
up: _signoz-fetch
    docker compose up -d --build --wait
    docker compose -f {{SIGNOZ_COMPOSE}} up -d
    @echo "stack up — mlflow http://localhost:5000 · floci S3 :4566 · SigNoz http://localhost:8080 (OTLP :4318)"
    @echo "NOTE: first run only — create the SigNoz admin account once at http://localhost:8080"
    @echo "      (until an org exists, the collector can't register and OTLP ingestion is refused)."

# Clone the pinned official SigNoz stack on first use (gitignored; real, unmodified upstream).
_signoz-fetch:
    #!/usr/bin/env sh
    set -eu
    if [ ! -f "{{SIGNOZ_COMPOSE}}" ]; then
        echo "fetching SigNoz {{SIGNOZ_VERSION}} -> {{SIGNOZ_DIR}}/ (one-time)"
        git clone --depth 1 --branch {{SIGNOZ_VERSION}} https://github.com/SigNoz/signoz {{SIGNOZ_DIR}}
    fi

# Stop the stack, keeping volumes (MLflow registry + floci objects survive). Add `-v` to wipe.
down:
    -[ -f "{{SIGNOZ_COMPOSE}}" ] && docker compose -f {{SIGNOZ_COMPOSE}} down
    docker compose down

# Local-stack acceptance smoke (needs `just up` + `just sync-ml`): log an MLflow run and
# round-trip an artifact through floci S3. dotenv-load gives it the MLFLOW_*/AWS_* env.
stack-smoke:
    uv run --extra ml python scripts/stack_smoke.py
