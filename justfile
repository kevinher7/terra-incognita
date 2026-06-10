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

# Build a seeded, location-split stratified subset (writes subset COCO + split.json to `out`).
subset coco out:
    uv run terra-incognita subset --coco {{coco}} --out {{out}}

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

package image="cct-detector-serving":
    uv run --extra ml mlflow models build-docker -m "models:/cct-detector@champion" -n {{image}}

# Serve the @champion pyfunc over REST (base64 image in / xyxy + COCO id out). Locally we run
# in the current env (`--env-manager local`) for a fast dev loop; the byte-identical deployed
# path is `just package` (build-docker). POST base64 records to http://localhost:<port>/invocations.
serve port="5001":
    uv run --extra ml mlflow models serve -m "models:/cct-detector@champion" -p {{port}} --env-manager local

# Connected end-to-end smoke (slice 7): the regression backbone that chains every stage on
# synthetic fixtures — COCO→YOLO (assert math) → 1-epoch train + MLflow logging → pyfunc package
# → serving round-trip → training.run wide-event schema. Server-free (tmp SQLite + tmp local
# artifact dir, in-process load_model, in-memory OTel exporter): no `just up` stack needed, only
# the `ml` extra (on macOS `--extra ml` resolves the CPU/MPS wheels — fast). The CI `smoke` job
# runs the same test with CPU torch on Linux (see .github/workflows/ci.yml).
smoke:
    uv run --extra ml pytest tests/smoke

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

# Dataset-pipeline acceptance smoke (needs `just up` + `just sync-ml`): sample a subset,
# upload it to floci S3, register it in the `datasets` experiment, and verify the round-trip.
dataset-smoke:
    uv run --extra ml python scripts/dataset_smoke.py

# Training acceptance smoke (needs `just up` + `just sync-ml`): materialize a dataset from its
# s3_uri, 1-epoch device-agnostic train with MLflow autolog, register a @champion model with
# provenance + signature + architecture, and emit the training.run wide event (slice 5).
train-smoke:
    uv run --extra ml python scripts/train_smoke.py

# Serving round-trip acceptance smoke (needs `just up` + `just sync-ml`): produce a @champion,
# load it via the registry alias, and round-trip a base64 image through the real CCTDetector —
# asserting xyxy absolute-pixel boxes, the real COCO category_id join key, and conf honored (slice 6).
serve-smoke:
    uv run --extra ml python scripts/serve_smoke.py

# --- real Caltech data run (one-time, NOT in CI) ----------------------------
# Slice 8 step 1 (needs `just up` + `just sync-ml`): download the real LILA Caltech Camera Traps
# bbox annotations + the ~5K selected images, sample the seeded stratified location-split subset,
# upload it to S3 (floci), and register it in the `datasets` experiment. Heavy (a few GB of
# images, cached in data/); prints the version to pin into configs/cct_real.yaml.
real-dataset:
    uv run --extra ml python scripts/real_dataset.py

# Slice 8 step 2 (needs the registered dataset + `just up` + `just sync-ml`): materialize
# cct-subset from its s3_uri, train configs/cct_real.yaml on MPS, register a real @champion with
# provenance + metrics, emit the training.run wide event, and serve-check it on a held-out image.
real-train config="configs/cct_real.yaml":
    uv run --extra ml python scripts/real_train.py --config {{config}}
