# Training Pipeline — Planning & Decisions Document

> **What this is.** The consolidated output of the planning/grilling session for the
> `terra-incognita` repo (the MLflow training + serving side of the camera-trap
> failure-analysis project). It captures the goal, the decisions we made and *why*, the
> contracts with the dashboard repo, and the infra handoffs — enough to write a concrete
> implementation plan from. It is **context for planning, not an implementation plan**.
>
> **Companion docs:**
> - [`CONTEXT.md`](./CONTEXT.md) — original aspirational context (dataset, model, MLflow concepts).
> - [`../contracts/`](../contracts/) — the **authoritative cross-repo contracts**
>   (serving I/O, dataset conventions, model registry, bbox format, MLflow topology).
>   Serving / dataset / promotion / bbox specifics live there; this doc links to them.
> - [`../dashboard/DESIGN.md`](../dashboard/DESIGN.md) — the dashboard (consumer) design.
> - [`../infra/CONTEXT.md`](../infra/CONTEXT.md) — the infra/Terraform planning context.

---

## 1. Goal & non-goals

**Goal.** Use **MLflow** to train, track, register, and **serve** a wildlife object
detection model so the dashboard has a *usable served model + a discoverable registered
dataset* to consume. The deeper goal is **learning ML-Engineering best practices**
around MLflow: experiment tracking, model registry + aliases, reproducible runs, a clean
serving contract, and local/prod parity.

**Non-goals.**
- **Model accuracy.** The model is *deliberately weak* (nano architecture, small subset)
  so the dashboard has failure modes to analyze. We never optimize mAP.
- **Terraform implementation.** Infra-as-code is owned by the infra repo (the user).
  This doc only *flags* what infra is needed (§11).
- **Dashboard implementation.** Separate repo; we only honor/define its contracts.

---

## 2. Architecture topology

```
LOCAL ITERATION                          DEPLOYED (AWS)
─────────────────                        ──────────────
docker-compose:                          EC2 (MLflow) — placement TBD by infra repo:
 ├─ mlflow server                          ├─ mlflow server (docker)
 │   backend-store: sqlite (volume)        │   backend-store: sqlite (EBS volume)
 │   artifact-store: s3 (→ floci)          │   artifact-store: s3 (→ real AWS S3)
 ├─ floci  (AWS/S3 emulator @ :4566)      AWS S3 bucket(s): datasets, model weights
 └─ mlflow models serve (pyfunc)         EC2/container: mlflow models serve (@champion)
                                          EC2 spot (GPU): final training runs (deferred)
training CLI (local, MPS) ──logs──►MLflow      training (GPU) ──logs──► MLflow
                                          EC2 (dashboard, separate repo) ─consumes─►
                                              registry API + S3 + serving endpoint
```

**Key parity principle.** Local and deployed differ **only by environment values**, never
by code. The S3 endpoint (`MLFLOW_S3_ENDPOINT_URL`) points at **floci** locally and
**real S3** deployed; the tracking URI points at local docker vs EC2. floci
([floci.io](https://floci.io)) is a free local AWS emulator (LocalStack-Community
replacement) — it gives us a real S3 API locally so the same artifact/dataset code runs
in both places. floci has nothing to do with ML; it's purely the local S3.

---

## 3. MLflow setup (decided)

| Concern | Decision | Notes |
|---|---|---|
| Version | **MLflow 3.x** | Stages removed → we use aliases. Modern API. |
| Backend store | **SQLite** (`--backend-store-uri sqlite:///...`) | Registry needs a DB; the deprecated thing is the *FileStore* (`./mlruns`), **not** SQLite. Fine for single-writer scale. Behind a URI → swap to managed DB later with one line. |
| Artifact store | **S3** (`--default-artifact-root s3://...`) | floci locally, AWS S3 deployed, via `MLFLOW_S3_ENDPOINT_URL`. |
| Artifact access | **Direct client → S3** (no `--serve-artifacts` proxy) | Matches the dashboard's existing "read from S3 directly" assumption. |
| Tracking server | `mlflow server` in **docker-compose** | HTTP API on `:5000`. Must be network-reachable from training + dashboard. |
| Server placement (deployed) | EC2 — dedicated | Decided; see [../contracts/mlflow-topology.md](../contracts/mlflow-topology.md). This repo only needs the URI + S3 creds. |

---

## 4. Tech stack & repo conventions (decided)

- **`uv`** — env, deps, lockfile. Plan torch-wheel selection so a CUDA build installs on
  the GPU box and an MPS/CPU build installs locally (extras / platform markers / index
  URL). This is the one dependency subtlety to design up front (see §9).
- **`src/` layout**, **`ruff`** (lint + format), type hints throughout.
- **`Typer`** for CLI entrypoints (one per pipeline step).
- **`justfile`** as the canonical command surface (`just train`, `just serve`, `just smoke`, …).
- **MLflow `MLproject`** wrapping the steps so runs are reproducible via `mlflow run .`
  with pinned env + params.
- **Config** via env + a typed config object (paths, S3, tracking URI, hyperparams);
  **nothing hardcoded** — this is what makes local/GPU parity free.

---

## 5. Data pipeline (steps & decisions)

Source: **Caltech Camera Traps** (LILA), COCO format. Per [`CONTEXT.md`](./CONTEXT.md) §3–§5.

1. **Download** annotation JSON + images from LILA (on-demand script; not in CI).
2. **Filter** to images with bbox annotations (~66K of 243K).
3. **Stratified subset** ~5–10K images: min floor per class (~20, or all for rare like
   badger/fox), empty ratio reduced 70% → ~20–30%, **split by camera location**. Seeded,
   one-time, produces a fixed image list.
4. **COCO → YOLO conversion** — absolute top-left `xywh` → normalized center
   `class cx cy w h`; produce Ultralytics `images/{train,val}` + `labels/{train,val}` +
   `data.yaml`. **Preserve the YOLO-index → COCO-`category_id` map** (needed at serving —
   see [../contracts/serving-io.md](../contracts/serving-io.md)).
5. **Upload** subset (images + COCO file) to S3 (floci/real).
6. **Register the dataset** in MLflow per the `datasets`-experiment convention
   ([../contracts/dataset-conventions.md](../contracts/dataset-conventions.md)): one run
   per dataset version, tags carry `s3_uri`, `coco_annotation_key`, `sampling_config_json`,
   `seed`, `class_map_json`, stats; the COCO file is also logged as a run artifact.

**Gotchas to honor** ([`CONTEXT.md`](./CONTEXT.md) §9): ~5% annotation error rate, mixed
flash types, categories at different taxonomy levels, location-disjoint train/val (tests
generalization). None block us; the model is meant to be imperfect.

---

## 6. Training (decisions)

- **Architecture:** YOLOv8n / YOLOv11n via **Ultralytics**.
- **Device-agnostic:** `device: auto|mps|cuda|cpu` config — never hardcode. Local = MPS,
  GPU box = CUDA, same code.
- **MLflow logging — hybrid:** use **Ultralytics' built-in MLflow autologging** for
  params/metrics/mAP/per-class AP; write **custom code** only for what autolog can't do —
  pyfunc packaging, model signature, registry + `@champion` alias, dataset logging, and
  provenance tags (`device`, instance type, git SHA, dataset version).
- **Reproducibility:** fixed seeds; full config logged as params. Note MPS↔CUDA results
  are not bit-identical — acceptable (accuracy is a non-goal).
- **Data input:** training **materializes the dataset from its `s3_uri`** into the local
  YOLO layout (not a hardcoded path), so local and GPU runs are identical.

---

## 7. Model packaging & serving (decisions — full contract in contracts/)

Served as a **custom `mlflow.pyfunc`** via `mlflow models serve` (REST). This wrapper is
the single most important artifact in the repo.

**The authoritative wire contract is [../contracts/serving-io.md](../contracts/serving-io.md)**
— base64 image in, detections out, with the YOLO-index → COCO id translation and the
inference-param defaults specified there. Do not restate it here.

**Packaging:** runnable image via **`mlflow models build-docker`** so local and deployed
serving are byte-identical.

**Promotion** is alias-based per [../contracts/model-registry.md](../contracts/model-registry.md)
— `@champion` (served/consumed), `@challenger` (candidate); serving targets
`models:/<name>@champion`. No stages (removed in MLflow 3.x).

---

## 7b. Runtime model: training vs serving (and how a champion reaches serving)

These are **two different runtimes** with different lifecycles. They never call each
other; they meet only at the **MLflow tracking server + registry + S3** (topology:
[../contracts/mlflow-topology.md](../contracts/mlflow-topology.md)).

| | **Training runtime** | **Serving runtime** |
|---|---|---|
| Lifecycle | Ephemeral / batch — runs to completion, exits | Long-running — stays up |
| Traffic | Outbound only (writes runs/artifacts to MLflow + S3) | Inbound (answers REST inference from the dashboard) |
| Triggered by | A launch action (you / script / Terraform / CI) | Deployed once; restarts on a new model |
| Needs | S3 (r/w), tracking server, compute (MPS/GPU) | A port + reachability (no S3 at runtime — base64 + baked-in model) |
| Produces | A registered model + alias | Predictions |
| Where (now) | Laptop (MPS); GPU spot EC2 later | Container on the MLflow EC2 |

**Two clarifications that are easy to get wrong:**

- **Nothing triggers training through the serving runtime, and the MLflow tracking server
  is not a job scheduler.** It logs and stores; it cannot launch training. Training must
  be *started* by something that also provides compute — locally that's your shell; on AWS
  it's a provisioning action (Terraform / a boto3 launch script / a GitHub Actions
  `workflow_dispatch`) that boots a **transient GPU spot instance** running the thin
  "pull data → train → log → register + alias → self-terminate" entrypoint (§9). You watch
  progress in the MLflow UI, not by SSHing in.

- **Training is location-independent of where MLflow lives.** The same run can log to the
  local docker MLflow (floci S3) or to the real EC2 MLflow (real S3) purely via env vars —
  so a **local MPS run can produce a real `@champion` for the deployed dashboard**; GPU is
  only a speed/scale upgrade, never a requirement.

**How a newly promoted champion actually reaches serving.** Because the model is baked
into the serving image at build time (`build-docker`, §7), promoting the alias does **not**
change live serving on its own. A **GitHub Actions deploy pipeline** must rebuild the
serving image with the new champion → push to ECR → redeploy the container (see
[../infra/PLAN.md](../infra/PLAN.md) I4 and [../contracts/mlflow-topology.md](../contracts/mlflow-topology.md)).
Trigger options:
- **Manual `workflow_dispatch`** (recommended start — promotion is a deliberate act), or
- **Programmatic** — the training job's final step fires a GitHub `repository_dispatch`
  after setting the alias.

> Note: **OSS MLflow has no native registry webhooks** (a Databricks-managed feature), so
> there is no automatic "alias changed → redeploy" — the redeploy is always triggered by
> you or by the training job, never by MLflow itself. The alternative
> (`mlflow models serve -m models:/<name>@champion`, fetch-at-startup) trades immutability
> for a runtime S3+tracking dependency and lets a plain container restart pick up the new
> champion; we chose the immutable baked-in image.

---

## 8. Cross-repo contract (what the dashboard relies on)

The dashboard is a **pure consumer** (never triggers training, never writes). The
interfaces it relies on are the [`../contracts/`](../contracts/) docs — referenced here,
not restated:

1. **Datasets:** discover via the `datasets`-experiment convention —
   [../contracts/dataset-conventions.md](../contracts/dataset-conventions.md).
2. **Models:** resolve the served model via the registry —
   [../contracts/model-registry.md](../contracts/model-registry.md).
3. **Inference:** the pyfunc serving wire format —
   [../contracts/serving-io.md](../contracts/serving-io.md); store boxes per
   [../contracts/bbox-format.md](../contracts/bbox-format.md).
4. **Required model metadata** the training repo must log: the `architecture` string, plus
   metrics (mAP/precision/recall/per-class AP) for MLflow-UI comparison (see the
   model-registry contract).

---

## 9. AWS GPU activation — bake in now, flip on later

The repo is written so enabling GPU training requires **zero code changes** — only infra
(yours) + env values. Requirements designed in from day one:

- **Device-agnostic config** (§6) — `device=cuda` on the box.
- **S3-first data access** (§5/§6) — GPU box pulls dataset from `s3_uri`; no local data
  assumption.
- **Env/config-driven endpoints** — tracking URI, S3 endpoint/bucket, dataset version all
  from env. No `localhost`.
- **Torch wheel parity** (§4) — CUDA build resolvable on the GPU box.
- **GPU entrypoint** = thin "pull data → train → log to MLflow → shutdown" wrapper around
  the same `MLproject` ([`CONTEXT.md`](./CONTEXT.md) §8).
- **Spot interruption (optional):** push Ultralytics `last.pt` checkpoints to S3;
  `resume=True` on relaunch. Cheap insurance for ~2–4h runs.
- **Provenance tags** — `device`, instance type, git SHA, so GPU vs local runs are
  distinguishable.

**What changes when you flip it on (infra repo / Terraform — yours):** provision a GPU
**spot** instance (g4dn.xlarge or p3.2xlarge) with an IAM role granting **S3 read (data)
+ S3 write (artifacts)** and **network reach to the MLflow tracking server** (same
VPC / peering). See [../infra/PLAN.md](../infra/PLAN.md) I7. Terminate after the run.
Est. ~$2–6/run.

---

## 10. Testing & CI (decided)

- **Synthetic fixtures (no images in git).** A seeded generator (`tests/`) draws ~15
  shape-on-noise images and emits an *exactly matching* COCO JSON at test time into a tmp
  dir. (git-LFS for real images is a noted fallback, not the plan.)
- **Smoke pipeline** on the fixtures: COCO→YOLO (assert normalization math vs known
  boxes) → Ultralytics layout → **1-epoch smoke-train** (verifies training + MLflow
  logging fire) → pyfunc package → **serving round-trip** (base64 in → detections out).
- **GitHub Actions on PR:** `uv sync` → `ruff` (lint/format) → `pytest` (unit + the
  fixture smoke pipeline). MLflow in CI uses a **tmp SQLite backend + tmp local artifact
  dir** (no server needed); an **optional floci service container** can exercise the real
  S3 artifact path. Mirrors the dashboard's CI style (see [../infra/CONTEXT.md](../infra/CONTEXT.md) §6).

---

## 11. Infra handoffs (owned by the infra/Terraform repo — the user)

> All of these are recorded on the infra side; see [../infra/PLAN.md](../infra/PLAN.md)
> (actionable deltas I1–I7) and [../infra/CONTEXT.md](../infra/CONTEXT.md).

- **S3 bucket(s)** for datasets (COCO + images), model-weight artifacts. Private. Key
  layout (one bucket w/ prefixes vs many) TBD with infra. The dataset `s3_uri` the
  training repo registers must let the dashboard's IAM role download the COCO JSON +
  resolve image keys.
- **IAM:** training (local uses floci dummy creds; GPU box needs S3 read+write role);
  dashboard EC2 needs S3 read; serving process does **not** need S3 (base64 design — see
  [../contracts/mlflow-topology.md](../contracts/mlflow-topology.md)).
- **EC2 for MLflow server** (dedicated — decided) reachable from training + dashboard
  (VPC/peering). Backend SQLite on an EBS volume, protected as Tier-0 state.
- **Serving container** allocated compute on the MLflow EC2 and a **second ECR repo +
  deploy path** for the serving image.
- **GPU spot EC2** (deferred) per §9.

---

## 12. Open items

- ⏳ S3 bucket/prefix scheme (with infra; see [../infra/CONTEXT.md](../infra/CONTEXT.md) §3/§4).
- ⏳ Concrete tracking-server + serving-endpoint URLs (config-supplied).
- ⏳ Whether spot checkpoint-resume is worth implementing for the short nano runs.

---

## 13. Suggested implementation phases (for the future plan)

1. **Scaffold:** uv project, src layout, ruff, justfile, Typer skeleton, MLproject, config.
2. **Local MLflow stack:** docker-compose (mlflow server + floci), bucket bootstrap, env wiring.
3. **Fixtures + COCO→YOLO converter** + unit tests (the math is the riskiest pure logic).
4. **Dataset pipeline:** subset sampler, S3 upload, `datasets`-experiment registration.
5. **Training:** device-agnostic Ultralytics + autolog hybrid + provenance tags.
6. **Packaging + serving:** pyfunc wrapper (base64 in / xyxy + category_id out), signature,
   index→category_id map, `build-docker`, `@champion` alias.
7. **Smoke pipeline end-to-end** on fixtures; wire GitHub Actions.
8. **Real Caltech run** (local MPS) to produce a genuine `@champion`.
9. **(Deferred) GPU path** — thin entrypoint + infra handoff.
