# Infrastructure Planning Context

> Background/context document for the Terraform infrastructure (a separate
> repo). The MLflow-side corrections (I1–I7) are **folded into the relevant
> sections** below; the actionable deltas the Terraform repo must absorb are
> tracked in [`PLAN.md`](./PLAN.md). The canonical MLflow placement / backend /
> serving topology is the contract
> [`../contracts/mlflow-topology.md`](../contracts/mlflow-topology.md) — this doc
> covers the infra implementation around it.

---

## 1. What This Infrastructure Supports

A camera trap failure-analysis dashboard:
- **Frontend:** Static React build (Vite) served to browsers
- **Backend:** FastAPI Python application in a Docker container
- **Database:** SQLite file on the backend server's disk (EBS-backed)
- **Images:** Stored in S3, served via presigned URLs
- **MLflow:** Model registry + tracking server + a long-running model-serving container,
  on a **dedicated EC2** (decided — §7, [../contracts/mlflow-topology.md](../contracts/mlflow-topology.md))

---

## 2. Target Architecture

```
Internet
    |
    |---> CloudFront --> S3 (frontend static files)
    |
    |---> EC2 (FastAPI in Docker)  [dashboard]
    |        |-- SQLite on EBS volume
    |        |-- Cron: backup SQLite to S3
    |        |-- Pulls Docker image from ECR (terra-vigil-backend)
    |        +-- Generates presigned URLs for image S3 bucket
    |
    |---> EC2 (MLflow)  [dedicated — §7]
    |        |-- MLflow tracking + registry server (HTTP :5000)
    |        |-- Backend store: SQLite on its own EBS volume (Tier-0, §4)
    |        |-- Cron: backup MLflow SQLite to S3 + restore-on-launch
    |        |-- Serving container (own port), pulls cv-model-serving from ECR
    |        +-- Artifact store -> S3 (MLflow artifact prefix)
    |
    +---> S3 (images, COCO dataset files, MLflow artifacts, SQLite backups)

Observability (NOT AWS — separate cloud, §10)
    Both EC2s + training export OTLP wide events --> secured OTLP endpoint
        --> Hetzner CAX21 (ARM): self-hosted SigNoz (ClickHouse-backed)
            firewall: OTLP ports open only to the AWS egress IP + admin IP

ECR (Docker image registries)
    |-- terra-vigil-backend:latest
    +-- cv-model-serving:latest        [built by mlflow models build-docker]

GitHub Actions (CI/CD)
    |-- On merge to main (dashboard): build -> push -> deploy to dashboard EC2
    +-- On promotion (model):  resolve the champion alias -> build-docker -> push
                               -> deploy to MLflow EC2 serving container
```

---

## 3. AWS Services Required

| Service | Purpose | Cost Estimate |
|---|---|---|
| EC2 t3.micro (dashboard) | FastAPI backend + SQLite | Free tier / ~$8/mo |
| EC2 small (MLflow) | Tracking/registry server + serving container | ~$8/mo |
| EBS volume (gp3, 8GB) — dashboard | SQLite database storage | ~$1/mo |
| EBS volume (gp3, 8GB) — MLflow | MLflow backend SQLite (Tier-0) | ~$1/mo |
| S3 — images | Camera trap images (~5–10K subset) | ~$1–2/mo |
| S3 — datasets (COCO annotation files) | Dashboard downloads to build SQLite | ~$0.01/mo |
| S3 — MLflow artifacts | Registered model weights + logged artifacts | ~$0.10/mo |
| S3 — frontend | Static React build | ~$0.10/mo |
| S3 — backups | Dashboard + MLflow SQLite backups | ~$0.01/mo |
| CloudFront | CDN for frontend | ~$0–1/mo |
| ECR (×2) | `terra-vigil-backend` + `cv-model-serving` | ~$0.10–0.20/mo |
| Route 53 (optional) | DNS | $0.50/mo per zone |
| VPC + Security Groups | Network isolation | Free |
| IAM | Roles for EC2 (×2), GitHub Actions | Free |

**Estimated total: ~$18–25/mo** (less with free tier). The v1 estimate of ~$10–15/mo
omitted MLflow compute entirely; the dedicated MLflow EC2 + its EBS add ~$8–9/mo. SQLite
still avoids RDS cost (~$15–30/mo saved) for both databases.

> **Observability is off-AWS (§10).** The deployed SigNoz backend runs on a **Hetzner
> CAX21** (ARM, ~€6.49/mo), not AWS — a deliberate multicloud choice. It is **not** in the
> table above because it is not an AWS service; it is provisioned by the same Terraform via
> the `hcloud` provider ([PLAN.md](./PLAN.md) I8). Canonical telemetry contract:
> [../contracts/observability.md](../contracts/observability.md).

> **S3 layout note.** Whether the above S3 line items are **prefixes in one bucket** or
> **separate buckets** is still open (§4). The training repo registers a concrete `s3_uri`
> per dataset; the dashboard reads it. All buckets/prefixes private.

---

## 4. Key Infrastructure Concerns

### EBS Volumes and Terraform — both SQLite databases are Tier-0 state
Terraform may destroy and recreate EC2 instances on certain changes (AMI change,
instance type change, etc.). By default the root EBS volume is deleted with the instance.

There are **two** SQLite databases that must survive instance replacement:

1. **Dashboard SQLite** — the analysis DB (runs, predictions, error classifications).
2. **MLflow backend SQLite** — *this is the registry itself*. It holds every experiment,
   run, registered model + version + alias, and the `datasets`-experiment records. Losing
   it loses **all** model and dataset metadata the dashboard depends on — the S3 artifacts
   alone cannot reconstruct it. Treat it as **Tier-0** with the **identical** survival
   treatment as the dashboard SQLite, not as an afterthought. (Canonical statement:
   [../contracts/mlflow-topology.md](../contracts/mlflow-topology.md).)

Mitigations (applied to **both** databases):
- Set `delete_on_termination = false` on the EBS volume (dedicated volume per DB)
- Cron job backs up SQLite to S3 (e.g., every 6 hours)
- User data script on instance launch pulls latest `.db` from S3
- This makes each EC2 instance effectively stateless-recoverable

### Docker Deployment Flow
Two images, two flows (see §6):
1. **Dashboard backend** — GitHub Actions builds on merge to main, pushes to ECR, dashboard
   EC2 pulls latest image and restarts container.
2. **Model serving** — built via `mlflow models build-docker` on promotion, pushed to a
   second ECR repo, MLflow EC2 pulls and restarts the serving container.
Options for the pull/restart step: direct SSH/SSM command, or docker-compose on EC2.

### S3 Bucket Design
- Consider one bucket with prefixes vs. multiple buckets (open). Contents to host:
  **camera-trap images**, **dataset COCO annotation files**, the **MLflow artifact store**
  (model weights + logged artifacts), **frontend build**, **SQLite backups**.
- All buckets private (no public access)
- Presigned URLs for image access (generated by FastAPI backend)
- Frontend bucket serves as CloudFront origin

### Security
- EC2 in VPC with minimal security group (HTTPS + backend port inbound)
- IAM instance role for S3 access (no hardcoded access keys)
- GitHub Actions uses OIDC federation for AWS auth
- Application-level JWT auth (shared password in env var or Secrets Manager)
- **Observability OTLP endpoint (off-AWS):** the deployed SigNoz exposes a public OTLP
  endpoint (4317/4318) on the Hetzner box. Harden it: **TLS + bearer-token auth** on the
  exporter, and a **firewall locked to the AWS egress IP** (+ admin IP for the UI). Token
  in Secrets Manager / env, never committed. This is the one genuine new attack surface the
  multicloud choice introduces (§10 / [PLAN.md](./PLAN.md) I8).

### SSL/TLS
- CloudFront handles HTTPS for frontend automatically
- Backend options: ALB with ACM certificate, or Caddy/nginx reverse proxy on EC2
- ALB adds ~$16/mo; Caddy is free but more manual setup

---

## 5. Suggested Terraform Module Structure

```
infra-repo/
|-- main.tf
|-- variables.tf
|-- outputs.tf
|-- providers.tf
|-- modules/
|   |-- networking/     # VPC, subnets, security groups
|   |-- compute/        # dashboard EC2 + MLflow EC2, EBS, IAM roles, user data
|   |-- storage/        # S3 buckets/prefixes, CloudFront distribution
|   |-- ecr/            # Container registries (terra-vigil-backend + cv-model-serving)
|   |-- gpu-training/   # transient GPU spot (deferred — §9)
|   +-- dns/            # Route 53 (optional)
+-- environments/
    +-- prod/
        |-- main.tf
        +-- terraform.tfvars
```

---

## 6. CI/CD Pipeline Detail

### On PR (lint + test):
- ruff (Python linting)
- eslint (TypeScript linting)
- pytest (backend unit + integration tests)
- Vite build (verify frontend compiles)
- Docker build (verify image builds, don't push)

### On merge to main — dashboard deploy:
1. Build Docker image with commit SHA tag
2. Push to ECR `terra-vigil-backend` (both :latest and :sha-xxxxx)
3. SSM Run Command (or SSH) to dashboard EC2: pull new image, restart container
4. Build frontend with `npm run build`
5. Sync frontend build to S3 (`aws s3 sync`)
6. Invalidate CloudFront cache

### On model promotion — serving deploy (second flow):
1. Resolve `models:/<name>@champion` and pull the model artifact from S3
   (promotion mechanism: [../contracts/model-registry.md](../contracts/model-registry.md))
2. Build the serving image via `mlflow models build-docker`
3. Push to ECR `cv-model-serving` (:latest and :sha/version tag)
4. SSM/SSH to the **MLflow EC2**: pull new image, restart the serving container
> OSS MLflow has no native registry webhooks, so this flow is triggered by a manual
> `workflow_dispatch` (recommended start) or a `repository_dispatch` fired by the training
> job after it sets the alias — never by MLflow itself. See
> [../training/PLAN.md](../training/PLAN.md) §7b.

### GitHub Actions AWS Auth:
- OIDC identity provider in AWS
- IAM role with trust policy for GitHub Actions
- Permissions: `ecr:PushImage` (both repos), `s3:PutObject`, `ssm:SendCommand`, **plus
  `s3:GetObject` on the MLflow artifact prefix** — the serving-image build step resolves
  the champion alias and pulls the model artifact at build time (see §7 / `PLAN.md` I5).

---

## 7. MLflow Infrastructure (decided)

> Canonical placement / backend store / serving / IAM spec:
> [../contracts/mlflow-topology.md](../contracts/mlflow-topology.md). The infra
> implementation specifics are below.

### Placement & backend store
- **Placement: a dedicated small EC2** for the MLflow server (tracking + registry +
  serving container). Clean separation from the dashboard, still cheap. Must be reachable
  from the dashboard EC2 and from training environments (same VPC / peering — see network
  requirements below).
- **Backend store: SQLite** (not Postgres). Behind a store URI so it can be swapped to a
  managed DB later without code changes. Protected as Tier-0 state (§4).
- **Artifact store: S3** (the MLflow artifact prefix — §3). floci (emulated S3) locally.

### Model serving needs allocated compute
`mlflow models serve` (or the built serving image) is a **long-running container on its own
port**, not a serverless call. It must be running and reachable for the dashboard to do
inference at all. Run it as a **separate container on the MLflow EC2** (cheapest option),
on its own port, reachable from the dashboard EC2. Account for its memory/CPU headroom on
that instance. (If load ever warrants it, the serving container can move to its own
instance later — but one EC2 hosting both the tracking server and the serving container is
the right starting point for this project's scale.)

### Serving runtime IAM ≈ none (least-privilege win)
Images arrive **base64 in the request body**, and the model is **baked into the serving
image at build time** (see [../contracts/serving-io.md](../contracts/serving-io.md)).
Therefore the **running serving container needs no S3 access and no image-bucket access** —
only network reachability from the dashboard. S3 read for the model artifact happens at
**build time (CI)**, not at runtime (§6). Scope the serving container's IAM accordingly
(effectively none for AWS data services). This is a deliberate least-privilege
simplification.

### MLflow server requires:
- Tracking server (HTTP API on :5000)
- Backend store (SQLite — Tier-0, §4)
- Artifact store (S3 for model weights + datasets/logged artifacts)
- Model serving endpoint (long-running container — see above)

### Network requirements:
- Dashboard EC2 must reach MLflow tracking server + serving endpoint
- Training environment (local or GPU EC2) must reach MLflow tracking server
- Same VPC or VPC peering if separate instances

---

## 8. Cost Optimization Notes

- EC2 free tier: t3.micro, 750 hrs/mo for 12 months (covers one instance; the MLflow EC2
  is the second instance and is the source of the +~$8–9/mo over the v1 estimate)
- S3 Intelligent-Tiering for images
- CloudFront free tier: 1TB transfer/mo, 10M requests/mo
- ECR: first 500MB free, then $0.10/GB/mo (now two repos)
- Avoid ALB if possible (~$16/mo) — use Caddy on EC2 instead
- Spot instances for any GPU training workloads (see §9)
- SQLite eliminates RDS cost entirely (~$15–30/mo saved), for both the dashboard and the
  MLflow backend

---

## 9. GPU Training Compute (deferred)

> **Deferred — activate when GPU training is turned on.** Until then, training is
> local-MPS only and needs no AWS compute. A local MPS run can still produce a real
> champion for the deployed dashboard; GPU is only a speed/scale upgrade.

When activated, add a **transient GPU spot instance** path (e.g. g4dn.xlarge /
p3.2xlarge), provisioned for a run and terminated after (the `gpu-training` module, §5).
Requirements:
- **IAM:** S3 **read** (datasets) + S3 **write** (the MLflow artifact prefix).
- **Network:** reach to the MLflow tracking server (same VPC / peering, §7).
- **Entrypoint:** the thin "pull data → train → log to MLflow → shutdown" wrapper
  ([../training/PLAN.md](../training/PLAN.md) §9).
- **Cost:** est. ~$2–6/run.

---

## 10. Observability backend — multicloud (Hetzner), Terraform-provisioned

The deployed observability backend (self-hosted **SigNoz**, ClickHouse-backed) runs **off
AWS, on Hetzner** — a deliberate multicloud choice. The canonical telemetry contract is
[../contracts/observability.md](../contracts/observability.md); the actionable infra delta
is [PLAN.md](./PLAN.md) **I8**.

### Why off-AWS (the cost logic is counterintuitive)
SigNoz needs **4GB RAM minimum, 8GB comfortable** (ClickHouse OOMs at 2GB). Against that:
- **GCP/Azure free tiers are 1GB → they OOM**; their cheapest *viable* boxes cost
  **$24–60/mo** (GCP e2-medium ~$24, Azure B2ms ~$60) — the *priciest* options.
- A **second AWS box** (Lightsail 4GB $24 / EC2 t3.medium ~$30) **blows the whole project
  budget** by itself.
- **Hetzner CAX21** (ARM, 4 vCPU / 8GB, NVMe, 20TB traffic) = **~€6.49/mo**, reliable, and
  has an **official Terraform provider** (`hcloud`). *(Oracle Always Free A1 is $0 but has
  signup capacity-hunting + idle-reclaim risk on a demo box; rejected for reliability.)*

So the cheap, reliable, Terraform-native path is **off-AWS**.

### Cross-cloud is fine at this volume
- **Egress ≈ $0.** AWS gives 100GB/mo egress free, then $0.09/GB; a portfolio app's OTLP
  volume is far under that. The classic "cross-cloud egress will kill you" warning does
  **not** apply here.
- **Latency:** OTLP export is async/batched — cross-cloud RTT is irrelevant.
- **The real cost** is securing a public OTLP endpoint (§4 Security) — modest, and itself a
  good infra-as-code/security artifact.

### Terraform shape
Multi-provider config (`provider "aws"` + `provider "hcloud"`), or split state if you
prefer isolation. One `terraform apply` provisions the AWS ML system **and** its
self-hosted observability backend on a second cloud, wired over a firewalled OTLP endpoint.
