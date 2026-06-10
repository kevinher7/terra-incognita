# terra-incognita

MLflow training + serving side of the camera-trap failure-mode CV project. The deeper
goal is **learning ML-Engineering best practices** (MLflow tracking, model registry +
aliases, reproducible runs, a clean serving contract, local/prod parity) and making the
whole system **observable** via OpenTelemetry wide events.

> Cross-repo contracts and design live in [`.plans/`](./.plans) — a read-only,
> `git subtree`-vendored mirror of [`terra-carta`](https://github.com/kevinher7/terra-carta).
> The single source of truth for shared interfaces. Do not edit it here.

## Slice 1 — scaffold (this slice)

A thin testable vertical: a Typer CLI command emits a registry-validated `training.run`
wide event captured by an in-memory OTel exporter and asserted in a unit test, all
behind a green CI gate.

| Piece | Where |
|---|---|
| Environment/parity config (env-driven: tracking URI, S3, OTLP, device) | `src/terra_incognita/config.py` |
| Experiment config (versioned hyperparameters, one file per experiment) | `src/terra_incognita/experiment.py` + `configs/*.yaml` |
| Typed OTel wide-event helper (Pydantic-backed) | `src/terra_incognita/obs/` |
| CLI (one stub per pipeline step + `demo-event`) | `src/terra_incognita/cli.py` |
| Reproducible step wrapper | `MLproject` + `python_env.yaml` |
| Command surface | `justfile` |
| CI gate (uv sync → ruff → ty → pytest) | `.github/workflows/ci.yml` |

The wide-event field names come from `.plans/contracts/observability.attributes.yaml`;
a CI guard test fails if the typed models drift from that registry.

## Quickstart

```sh
just sync          # core + dev deps (lean; no torch). Mirrors CI.
just check         # ruff lint + format-check + ty type-check + pytest
just demo-event    # emit a training.run wide event (set OTEL_EXPORTER_OTLP_ENDPOINT to ship it)
just sync-ml       # add the heavy ML stack (torch/ultralytics/mlflow) for later slices
```

Copy [`.env.example`](./.env.example) to `.env` for local config. Torch wheels are
platform-routed (macOS → MPS-capable CPU wheels, Linux → CUDA cu124) in
`pyproject.toml` so the same project installs on a laptop and the GPU box unchanged.

## Local stack (docker-compose)

The dataset/training slices need a tracking server, an S3 API, and a telemetry sink. The
local stack stands these up so **local and deployed differ only by env values** (PLAN §2):

```sh
just up      # mlflow + floci (this repo's compose) + always-on SigNoz (pinned official stack)
just down    # stop everything (volumes kept — add `-v` to wipe the registry + objects)
just stack-smoke   # acceptance: log an MLflow run + round-trip an artifact through floci S3
```

> **First `just up` only:** create the SigNoz admin account once at <http://localhost:8080>.
> SigNoz won't let its OTLP collector register until an org exists, so wide-event ingestion
> is refused (connection reset) until you do — a one-time, upstream SigNoz requirement, not a
> stack bug. After that, `just demo-event` lands in SigNoz's trace view.

| Service | Local | Prod (parity by env only) | Endpoint |
|---|---|---|---|
| **mlflow** (`docker/mlflow.Dockerfile` = official image + boto3) | docker | dedicated EC2 | `MLFLOW_TRACKING_URI` → `:5000` |
| **floci** (S3 emulator) | docker | real AWS S3 | `MLFLOW_S3_ENDPOINT_URL` → `:4566` |
| **SigNoz** (wide-event sink) | docker (always-on) | self-hosted on Hetzner | `OTEL_EXPORTER_OTLP_ENDPOINT` → `:4318` · UI `:8080` |

Design notes (the *why*):

- **MLflow:** SQLite backend (the registry; a named volume), S3 artifact root, and
  **`--no-serve-artifacts`** so clients read/write artifacts **directly** to S3 — matching
  the dashboard's "read from S3" assumption ([`contracts/mlflow-topology.md`](./.plans/contracts/mlflow-topology.md)).
- **floci** is the only delta vs prod S3: it needs **path-style addressing**
  (`MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE=path`). Buckets/prefixes are bootstrapped on `up`
  by a floci `ready.d` init hook (`docker/floci-init/`), not a side container.
- **SigNoz** runs from its **own pinned, unmodified official compose** (fetched once into a
  gitignored `.signoz/` on first `up`) so the local sink is the *same* stack the Hetzner
  deploy runs (infra I8) — real local/prod parity, no hand-vendored ClickHouse configs to
  drift. It's the sink + query UI for every wide event `obs/` emits (operational telemetry —
  distinct from MLflow's ML metrics; [`contracts/observability.md`](./.plans/contracts/observability.md)).
- **CI uses none of this** — it asserts the event path with the in-memory OTel exporter
  (`tests/`), so there's no docker/SigNoz container in CI.

### Config split: environment vs experiment

Two config surfaces, separated by one question — *"does this differ between my laptop and
the GPU box for the **same** experiment?"*

- **Yes → environment/parity.** Tracking URI, S3 endpoint, OTLP endpoint, credentials,
  `device`, `instance_type`. These come from env vars / `.env` (`Settings`, `config.py`).
- **No → it defines the experiment.** `epochs`, `imgsz`, `batch`, `seed`, `model_arch`,
  `dataset_version`. These live in a **committed** `configs/*.yaml` (`ExperimentConfig`),
  so an experiment is a diffable file you point at — never an edit to ambient `.env` state:

  ```sh
  mlflow run . -e train -P config=configs/baseline.yaml   # reproducible surface
  uv run terra-incognita train --config configs/baseline.yaml --epochs 3   # ad-hoc override
  ```

  Define a new experiment by **copying** `configs/baseline.yaml`, not by editing `.env`.
  Unknown/typo'd keys in a config fail loudly.

## Real data run (one-time) — produce a genuine `@champion`

Slices 1–7 prove the whole pipeline on synthetic fixtures in CI. Slice 8 points the *same*
machine at the real [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps)
set (LILA, public, no auth) to register a real dataset version and train the first real
champion. It is **heavy and not in CI** (a ~38 MB annotation download + a few GB of images,
cached in the gitignored `data/`, then a ~1–2 h MPS train). Run it against the local stack:

```sh
just up && just sync-ml          # stack + heavy ML deps
just real-dataset                # download → clean → sample ~5K subset → upload to S3 → register
#   → prints the registered version (e.g. "v1"); pin it:
#   edit configs/cct_real.yaml → dataset_version: "v1"
just real-train                  # materialize from S3 → MPS train → register @champion → wide event
just serve                       # serve the @champion; POST a base64 real image to /invocations
```

What each step honors:

- **`just real-dataset`** (`scripts/real_dataset.py`) downloads `caltech_bboxes_20200316.json`,
  drops the bbox-less "empty" markers so empty images become background frames
  (`clean_bbox_coco`), runs the seeded stratified **location-disjoint** sampler (~5K images,
  ~25% empty), pulls **only the selected images** per-image from LILA's unzipped folder, uploads
  the subset + COCO to S3 (floci), and registers one run in the `datasets` experiment
  ([`dataset-conventions.md`](./.plans/contracts/dataset-conventions.md)). `min_per_class` in the
  script is the subset-size lever.
- **`just real-train`** (`scripts/real_train.py`) discovers that dataset the dashboard's way
  (`search_runs(experiment_names=["datasets"])`), materializes it from `s3_uri`, re-derives the
  location split, trains `configs/cct_real.yaml`, registers a real `cct-detector@champion` with
  provenance + mAP/precision/recall, emits the `training.run` wide event (queryable in SigNoz),
  and serve-checks the champion on a held-out **real** image.

> **Staging/prod is the deferred infra step.** This run targets the *local* stack; reaching a
> staging S3 + an EC2-served champion needs only env changes (`MLFLOW_TRACKING_URI`,
> `MLFLOW_S3_ENDPOINT_URL`) then `just package` (build-docker) → ECR → container — **no code
> changes** (PLAN §7b/§9; infra PLAN I1–I4/I7).
