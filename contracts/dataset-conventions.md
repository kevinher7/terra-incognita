# Contract: Dataset conventions (the `datasets`-experiment convention)

**Governs:** how dataset versions are registered in MLflow and discovered by the
dashboard, including the authoritative source for the class list.

## Spec

MLflow's `mlflow.data` Dataset API is **not a queryable registry** — it only
attaches a dataset descriptor as an input to a specific run, with no "list all
datasets" / "get by name+version" API. So dataset versions are instead
registered by the training repo as **runs in a dedicated MLflow experiment named
`datasets`** (one run = one dataset version). The dashboard discovers them by
searching that experiment and reading tags:

```python
runs = mlflow.search_runs(experiment_names=["datasets"])  # lists all dataset versions
# pick by tags.dataset_name / tags.version
# read tags.s3_uri + tags.coco_annotation_key, then download + parse the COCO file
```

Relevant tags on each dataset run:

| Tag / param | Example | Notes |
|---|---|---|
| `dataset_name` | `cct-subset` | logical dataset name |
| `version` | `v3` | dataset version string |
| `s3_uri` | `s3://<bucket>/datasets/cct-subset/v3/` | prefix holding COCO file + images |
| `coco_annotation_key` | `datasets/cct-subset/v3/annotations.json` | exact key of the COCO JSON |
| `sampling_config_json` | `{...}` | strategy, per-class counts, empty ratio |
| `seed` | `42` | sampling seed for reproducibility |
| `class_map_json` | `{"0":"empty",...}` | **mirror only** — see note below |
| `num_images`, `num_annotations` | `5000`, `8123` | quick stats |

The actual COCO file is also logged as a run artifact for provenance. The
dashboard downloads the COCO JSON from `s3_uri` / `coco_annotation_key` and
parses it into SQLite (Image, Category, GroundTruth).

**Source of truth for classes.** The dashboard derives its `Category` table by
parsing the **COCO annotation file** it downloads from S3. `class_map_json` on
the run is a convenience mirror for quick listing, **not** the authoritative
class map. The COCO file wins.

## Depended on by

- **dashboard** — discovers datasets via `search_runs(experiment_names=["datasets"])`,
  reads tags, downloads + parses the COCO file into SQLite.
- **training** (`terra-incognita`) — registers one run per dataset version in the
  `datasets` experiment with the tags above; logs the COCO file as a run artifact.

## Rule

Do not redefine this elsewhere. Reference this file. To change the dataset
discovery convention, change it here.
