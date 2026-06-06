# terra-carta

**The single source of truth for plans and cross-repo contracts** for the
camera-trap failure-analysis project. It coordinates:

- **`terra-vigil`** — the failure-analysis dashboard (consumes this repo).
- **`terra-incognita`** — the MLflow training + serving pipeline (consumes this repo).
- the **infra / Terraform** repo — the AWS infrastructure. Its plans live here
  (`infra/`), but the infra repo does **not** vendor this mirror; its deltas are
  applied by hand.

Plans are plain markdown in git: greppable, diffable, commit-pinnable. There is
no Notion or auth-gated store. This repo replaces the old workflow of authoring
plans in `terra-vigil/.plans/` and hand-copying them into the other repos.

## Two content types

- **`contracts/`** — **shared interfaces that more than one repo depends on.**
  These are authoritative. Today:
  - [`serving-io.md`](contracts/serving-io.md) — the pyfunc inference wire format.
  - [`dataset-conventions.md`](contracts/dataset-conventions.md) — the `datasets`-experiment convention.
  - [`model-registry.md`](contracts/model-registry.md) — alias-based model promotion.
  - [`bbox-format.md`](contracts/bbox-format.md) — `xyxy` end-to-end bounding boxes.
  - [`mlflow-topology.md`](contracts/mlflow-topology.md) — MLflow placement, backend store, serving.
  - [`observability.md`](contracts/observability.md) — wide-event/OTel telemetry,
    trace propagation, and sinks (local vs deployed SigNoz). Its machine-readable
    field registry is [`observability.attributes.yaml`](contracts/observability.attributes.yaml).
- **`<repo>/`** — **implementation plans local to one repo:**
  - [`dashboard/`](dashboard/) — `terra-vigil` design.
  - [`training/`](training/) — `terra-incognita` context + plan.
  - [`infra/`](infra/) — infra/Terraform context + actionable deltas.

## The cardinal rule for consumers

**A repo NEVER redefines a contract locally** — it links to
`contracts/<name>.md`. To change a contract, change it **here**, in this repo.

## How a repo consumes this

The two code repos — `terra-vigil` and `terra-incognita` — vendor this repo into
their own `.plans/` as a **read-only mirror** via `git subtree`:

```bash
# one-time, run inside each consuming repo
git subtree add --prefix=.plans https://github.com/kevinher7/terra-carta main --squash

# to refresh later
git subtree pull --prefix=.plans https://github.com/kevinher7/terra-carta main --squash
```

The infra / Terraform repo is **not** a consumer: its plans (`infra/`) are
authored here and applied by hand, so it never vendors the mirror.

## Authoring rule

Edit here, push, then `git subtree pull` in each repo. **Never edit `.plans/`
inside a consuming repo** — those are read-only mirrors and your edits would be
lost on the next pull.
