# Infra Plan — deltas the Terraform repo must absorb

> The actionable changes/decisions for the infra/Terraform repo, derived from the
> MLflow training + serving design. Background and the folded-in context live in
> [`CONTEXT.md`](./CONTEXT.md); the canonical MLflow topology is the contract
> [`../contracts/mlflow-topology.md`](../contracts/mlflow-topology.md).
>
> **Status legend:** 🔧 required change · ✅ decided / record.

---

## I1 — Protect the MLflow backend SQLite as Tier-0 state 🔧

The MLflow backend SQLite **is the registry** — it holds every experiment, run,
registered model + version + alias, and the `datasets`-experiment records. Losing it
loses **all** model and dataset metadata the dashboard depends on; the S3 artifacts alone
cannot reconstruct it.

**Required.** Give the MLflow server's SQLite the **identical** treatment as the dashboard
SQLite: dedicated EBS volume with `delete_on_termination = false`, scheduled backup to S3,
and restore-on-launch via user-data. Treat it as Tier-0. (See
[../contracts/mlflow-topology.md](../contracts/mlflow-topology.md) and
[`CONTEXT.md`](./CONTEXT.md) §4.)

---

## I2 — MLflow placement & backend store ✅

**Decided.**
- **Placement: a dedicated small EC2** for the MLflow server (tracking + registry +
  serving container). Reachable from the dashboard EC2 and from training environments
  (same VPC / peering).
- **Backend store: SQLite** (not Postgres), behind a store URI so it can be swapped later
  without code changes.

**Cost impact to record.** The v1 ~$10–15/mo estimate **omitted MLflow compute entirely**.
Add the dedicated EC2 + its EBS (~$8–9/mo) to the estimate. (SQLite still avoids RDS cost.)

---

## I3 — Provision the model-serving container 🔧

`mlflow models serve` (or the built serving image) is a **long-running container on its own
port**, not a serverless call. It must be running and reachable for the dashboard to do
inference at all.

**Required.** Run the served model as a **separate container on the MLflow EC2** (cheapest
option), on its own port, reachable from the dashboard EC2. Account for its memory/CPU
headroom on that instance. (Can move to its own instance later if load warrants.)

---

## I4 — Second ECR repo + deploy path for the served-model image 🔧

The model is packaged via **`mlflow models build-docker`** — a *separate* Docker image
that also needs to be built, stored, and deployed.

**Required.** Add a **second ECR repository** (e.g. `cv-model-serving`) and a
build → push → deploy path analogous to the dashboard backend. The **build step** resolves
`models:/<name>@champion` and pulls the model artifact from S3, so the **CI identity needs
S3 read** (and ECR push). Deploy = the MLflow EC2 pulls the new image and restarts the
serving container. (Promotion → deploy trigger is manual `workflow_dispatch` or
`repository_dispatch`; OSS MLflow has no registry webhooks — see
[../training/PLAN.md](../training/PLAN.md) §7b.)

---

## I5 — Serving runtime IAM is minimal ✅

Images arrive **base64 in the request body**, and the model is **baked into the serving
image at build time** (see [../contracts/serving-io.md](../contracts/serving-io.md)).

**Decided / record.** The **running serving container needs no S3 access and no
image-bucket access** — only network reachability from the dashboard. S3 read for the
model artifact happens at **build time (CI)**, not at runtime. Scope the serving
container's IAM accordingly (effectively none for AWS data services). Deliberate
least-privilege simplification, not an oversight.

---

## I6 — Enumerate the full S3 inventory 🔧

Beyond camera-trap images, S3 must also host:
1. **Dataset COCO annotation files** — the dashboard downloads these to build its SQLite
   (see [../contracts/dataset-conventions.md](../contracts/dataset-conventions.md)).
2. **The MLflow artifact store** — model weights + logged artifacts.

**Required.** Enumerate these in the inventory (as prefixes in a shared bucket or separate
buckets — the "one bucket w/ prefixes vs many" question is still open). Access pattern to
provision:
- **Dashboard EC2:** read — dataset images + COCO files.
- **Training (GPU box) + CI build:** write — the MLflow artifact prefix; read — datasets.
- **Local dev:** uses **floci** (emulated S3), no real-infra dependency.
All buckets private.

---

## I8 — Self-hosted SigNoz observability backend (Hetzner, Terraform) 🔧

Every runtime exports OTLP wide events to a self-hosted **SigNoz** backend. Local is SigNoz
in docker-compose; **deployed is SigNoz on Hetzner** — off-AWS, a deliberate multicloud
choice (cost rationale + cross-cloud analysis: [`CONTEXT.md`](./CONTEXT.md) §10). Canonical
telemetry contract: [../contracts/observability.md](../contracts/observability.md).

**Required.** Provision the deployed SigNoz via the **`hcloud` Terraform provider** in the
same repo (multi-provider config, or split state):
- **`hcloud_server`** — a **CAX21** (ARM, 4 vCPU / 8GB, NVMe) with **cloud-init** to install
  Docker and bring up the SigNoz compose stack on first boot.
- **`hcloud_volume`** — NVMe data volume for ClickHouse (retention-capped; a few GB at this
  volume).
- **`hcloud_firewall`** — open OTLP ports (4317/4318) **only to the AWS egress IP**, and the
  SigNoz UI port only to the admin IP. **TLS + bearer-token** on the OTLP exporter (token in
  Secrets Manager / env, never committed) — see [`CONTEXT.md`](./CONTEXT.md) §4 Security.
- **Config handoff:** supply `OTEL_EXPORTER_OTLP_ENDPOINT` + auth token to the dashboard
  EC2, the serving container, and training, exactly as the tracking URI is supplied. Local
  vs deployed differ by **this env value only** — same emit code (contract parity rule).

**Cost impact to record.** +~€6.49/mo (Hetzner CAX21). Cheaper *and* more reliable than any
AWS-resident option (a second AWS box is $24–30/mo); GCP/Azure free tiers OOM. Cross-cloud
egress ≈ $0 at this volume. **Fallback:** OTLP is vendor-neutral — the deployed sink can be
swapped to the **Honeycomb free tier** with one env-var change, no code change.

---

## Open / deferred

### I7 — Transient GPU training compute ⏳

**Deferred — activate when GPU training is turned on.** Until then, training is local-MPS
only and needs no AWS compute (a local MPS run can still produce a real champion for the
deployed dashboard).

When activated, add a **transient GPU spot instance** path (e.g. g4dn.xlarge /
p3.2xlarge), provisioned for a run and terminated after. Requirements:
- **S3 read** — datasets; **S3 write** — the MLflow artifact prefix.
- **Network reach** to the MLflow tracking server (same VPC / peering).
- Runs the thin "pull data → train → log to MLflow → shutdown" entrypoint. Est. ~$2–6/run.

### Still-open infra questions

- ⏳ **S3 key layout** — one bucket with prefixes vs. multiple buckets (owned with infra).
  The training repo registers a concrete `s3_uri` per dataset; the exact bucket/prefix
  scheme is TBD.
- ⏳ **Endpoint URLs** — tracking server (`:5000`) and per-model serving endpoint URLs are
  config-supplied, not pinned here; must be reachable from the dashboard.

---

## Summary of infra deltas

| # | Type | One-liner |
|---|---|---|
| I1 | 🔧 gap | Protect MLflow SQLite as Tier-0 state (EBS + S3 backup + restore). |
| I2 | ✅ decision | Dedicated MLflow EC2; SQLite backend; +~$8–9/mo to estimates. |
| I3 | 🔧 gap | Provision the serving container (own port, on the MLflow EC2). |
| I4 | 🔧 gap | Second ECR repo + build/push/deploy path for the serving image. |
| I5 | ✅ decision | Serving runtime IAM ≈ none (base64 + baked-in model). |
| I6 | 🔧 gap | Enumerate S3: images + COCO datasets + MLflow artifact store. |
| I8 | 🔧 gap | Self-hosted SigNoz on Hetzner (`hcloud`); firewall OTLP to AWS IP; +€6.49/mo. |
| I7 | ⏳ deferred | Transient GPU spot module (S3 r/w + tracking reach). |
