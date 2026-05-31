# Contract: Model registry (alias-based promotion)

**Governs:** how models are promoted in the MLflow registry and how the
dashboard resolves the currently-served model. Uses **aliases**, not the removed
stage field.

## Spec

MLflow Model Registry **stages** (`Staging`/`Production`/`Archived`) are
**deprecated and removed in MLflow 3.x**, which this project uses. There is no
`current_stage` to read. Promotion is alias-based:

- `@champion` — the model currently served and consumed by the dashboard.
- `@challenger` — a candidate under evaluation (optional).

The dashboard resolves the alias:

```python
mv = MlflowClient().get_model_version_by_alias(name, "champion")
# read name, version, and the `architecture` tag/param
```

The model-selector dropdown may still list *all* versions for browsing;
`@champion` is the default/served one. Serving always targets
`models:/<name>@champion`.

For human-readable display, the training repo may **also** set a tag (e.g.
`validation_status=production`), but the **alias is the source of truth**, not
the tag.

**Required model metadata** the training repo must log: the `architecture`
string (e.g. `"yolov8n"`), plus training metrics (mAP / precision / recall /
per-class AP) for MLflow-UI comparison.

## Depended on by

- **dashboard** — model-sync code resolves `@champion` via
  `get_model_version_by_alias`; the selector dropdown may list all versions.
- **training** (`terra-incognita`) — registers the model and sets the
  `@champion` / `@challenger` aliases; logs `architecture` + metrics.
- **infra** — a promoted `@champion` reaches serving only via a rebuild/redeploy
  of the baked-in serving image (no native MLflow webhooks);
  see [mlflow-topology.md](./mlflow-topology.md) and [serving-io.md](./serving-io.md).

## Rule

Do not redefine this elsewhere. Reference this file. To change the promotion
mechanism, change it here.
