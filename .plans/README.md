# terra-carta

**The single source of truth for plans and cross-repo contracts** across the
three repos of the camera-trap failure-analysis project:

- **`terra-vigil`** — the failure-analysis dashboard (consumer).
- **`terra-incognita`** — the MLflow training + serving pipeline (producer).
- the **infra / Terraform** repo — the AWS infrastructure.

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
- **`<repo>/`** — **implementation plans local to one repo:**
  - [`dashboard/`](dashboard/) — `terra-vigil` design.
  - [`training/`](training/) — `terra-incognita` context + plan.
  - [`infra/`](infra/) — infra/Terraform context + actionable deltas.

## The cardinal rule for consumers

**A repo NEVER redefines a contract locally** — it links to
`contracts/<name>.md`. To change a contract, change it **here**, in this repo.

## How a repo consumes this

Each consuming repo vendors this repo into its own `.plans/` as a **read-only
mirror** via `git subtree`:

```bash
# one-time, run inside each consuming repo
git subtree add --prefix=.plans https://github.com/kevinher7/terra-carta main --squash

# to refresh later
git subtree pull --prefix=.plans https://github.com/kevinher7/terra-carta main --squash
```

## Authoring rule

Edit here, push, then `git subtree pull` in each repo. **Never edit `.plans/`
inside a consuming repo** — those are read-only mirrors and your edits would be
lost on the next pull.
