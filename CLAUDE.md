# terra-incognita

The MLflow **training + serving** pipeline for the camera-trap project.

## Plans & cross-repo contracts — the single source of truth

All planning and cross-repo coordination lives in `.plans/`, a **read-only
mirror** of the [`terra-carta`](https://github.com/kevinher7/terra-carta) repo
(vendored via `git subtree`). Treat `.plans/` as the **single source of truth**
for how every repo in this project coordinates — the shared contracts *and* each
repo's design. When a question touches another repo's design or a shared
interface, read it from `.plans/` rather than guessing or restating it here.

**Never edit anything under `.plans/` in this repo.** It is a mirror; local edits
are overwritten on the next `subtree pull`. To change a plan or contract, edit it
in `terra-carta`, push, then refresh here:

```bash
git subtree pull --prefix=.plans https://github.com/kevinher7/terra-carta main --squash
```

### What's in `.plans/`

- `.plans/contracts/` — **authoritative** cross-repo interfaces. **Never redefine
  a contract locally**; link to the file.
  - `serving-io.md` — pyfunc inference wire format
  - `dataset-conventions.md` — the `datasets`-experiment convention
  - `model-registry.md` — alias-based model promotion
  - `bbox-format.md` — `xyxy` bounding boxes end-to-end
  - `mlflow-topology.md` — MLflow placement, backend store, serving
- `.plans/training/` — **this repo's** context + plan (`CONTEXT.md`, `PLAN.md`).
- `.plans/dashboard/` — the `terra-vigil` dashboard design.
- `.plans/infra/` — the infra/Terraform context + actionable deltas.
