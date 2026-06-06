# Contract: MLflow topology (placement, backend store, serving)

**Governs:** where MLflow runs, what backs its registry, how the served model is
hosted, and which piece of state is critical. This is the interface between the
infra/Terraform repo and the two MLflow consumers (dashboard + training).

## Spec

### Placement & backend store

- **Placement:** a **dedicated small EC2** for the MLflow server (tracking +
  registry + serving container). Clean separation from the dashboard, still
  cheap. Must be reachable from the dashboard EC2 and from training environments
  (same VPC / peering).
- **Backend store: SQLite** (`--backend-store-uri sqlite:///...`), not Postgres.
  Behind a store URI so it can be swapped to a managed DB later without code
  changes. (The deprecated thing is the *FileStore* `./mlruns`, **not** SQLite.
  Fine for single-writer scale.)
- **Artifact store: S3** (`--default-artifact-root s3://...`); floci (emulated
  S3) locally, real AWS S3 deployed, via `MLFLOW_S3_ENDPOINT_URL`. Artifact
  access is **direct client → S3** (no `--serve-artifacts` proxy).
- **Tracking server:** `mlflow server`, HTTP API on **`:5000`**. Must be
  network-reachable from training + dashboard.
- **Version: MLflow 3.x** (stages removed → aliases; see
  [model-registry.md](./model-registry.md)).

### The MLflow backend SQLite is Tier-0 critical state

The MLflow backend SQLite **is the registry** — it holds every experiment, run,
registered model + version + alias, and the `datasets`-experiment records (see
[dataset-conventions.md](./dataset-conventions.md) and
[model-registry.md](./model-registry.md)). Losing it loses **all** model and
dataset metadata the dashboard depends on; the S3 artifacts alone cannot
reconstruct it. It gets the **identical** survival treatment as the dashboard
SQLite: dedicated EBS volume with `delete_on_termination = false`, scheduled
backup to S3, and restore-on-launch via user-data. Treat it as **Tier-0**.

### Model serving

`mlflow models serve` (or the built serving image) is a **long-running container
on its own port**, not a serverless call. Run it as a **separate container on the
MLflow EC2** (cheapest option), reachable from the dashboard EC2. Its runtime IAM
is **~none** (no S3 / image-bucket access) because images arrive base64 and the
model is baked into the image at build time — S3 read for the model artifact
happens at **build time (CI)**, not at runtime (see [serving-io.md](./serving-io.md)).

A promoted `@champion` reaches live serving only by rebuilding the baked-in
serving image and redeploying the container — **OSS MLflow has no native registry
webhooks**, so this is triggered by a manual `workflow_dispatch` or a
`repository_dispatch` from the training job, never by MLflow itself.

### Boundary with operational telemetry

MLflow owns **ML metrics** (mAP, per-class AP, loss curves, run params). It does **not**
own operational telemetry (request/run/inference latency, exit reasons, counts) — those are
**wide events** under [observability.md](./observability.md). The two share keys
(`dataset_version`, `model_version`, `git_sha`) and join across systems, but never duplicate
payload. Do not log operational latency into MLflow; do not log ML metrics into wide events.

## Depended on by

- **infra** (`Terraform` repo) — provisions the dedicated MLflow EC2 + EBS,
  protects the backend SQLite as Tier-0, allocates the serving container, and
  scopes IAM. Actionable deltas live in `../infra/PLAN.md`.
- **training** (`terra-incognita`) — logs to the tracking server + S3 artifact
  store via env (`MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`); local/prod
  parity comes from env values only.
- **dashboard** — reaches the tracking server (registry/dataset discovery) + the
  serving endpoint; reads artifacts/datasets directly from S3.

## Rule

Do not redefine this elsewhere. Reference this file. To change MLflow placement,
backend store, or serving topology, change it here.
