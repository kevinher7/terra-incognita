# Training Pipeline Planning Context

> Background/context document for the ML training pipeline and MLflow setup (the
> `terra-incognita` repo). The MLflow integration details are pinned as
> cross-repo contracts in [`../contracts/`](../contracts/) and linked below
> rather than restated here. The decisions + phases live in
> [`PLAN.md`](./PLAN.md).

---

## 1. What the Training Pipeline Does

Train a wildlife object detection model (YOLO) on camera trap data, register
models and datasets in MLflow, and serve models for the failure-analysis
dashboard to consume via API.

---

## 2. Model

- **Architecture:** YOLOv8n or YOLOv11n (nano variant)
- **Framework:** Ultralytics
- **Deliberately imperfect:** nano on a small subset produces the failures
  the dashboard is built to analyze
- **Training environments:**
  - Local (Apple Silicon / MPS): prototyping, ~2-4h for nano on subset
  - AWS GPU (Terraform-provisioned): final training runs

---

## 3. Dataset: Caltech Camera Traps

**Source:** https://lila.science/datasets/caltech-camera-traps

### Overview
- 243K total images, 140 camera locations, 22 categories
- COCO format annotations
- ~66K images have bounding box annotations (rest are image-level only)
- ~70% of images are empty (no animal)
- Extremely long-tailed class distribution

### All 22 Categories
| Category | Taxonomy Level | Approx. Train Count (CCT-20) |
|---|---|---|
| empty | -- | ~70% of total images |
| opossum | species | 2,470 |
| rabbit | family | 2,190 |
| coyote | species | 1,200 |
| cat | species | 1,164 |
| squirrel | genus | 1,024 |
| raccoon | species | 845 |
| bobcat | species | 673 |
| dog | species | 580 |
| bird | class | 353 |
| rodent | order | 260 |
| skunk | family | 212 |
| deer | genus | 38 |
| fox | species | 5 |
| badger | species | 3 |
| mountain_lion | species | rare |
| bat | order | rare |
| insect | class | rare |
| lizard | order | rare |
| cow | species | rare |
| pig | species | rare |
| car | -- | rare |

### Per-Image Metadata Fields
| Field | Type | Notes |
|---|---|---|
| id | string | Unique image ID |
| file_name | string | Relative path |
| width | int | Pixels |
| height | int | Pixels |
| datetime | string | "YYYY-MM-DD HH:MM:SS" |
| location | int | Camera/site ID (140 unique) |
| seq_id | string | Burst sequence ID |
| seq_num_frames | int | Total frames in burst |
| frame_num | int | 0-indexed position in burst |
| corrupt | bool | Failed to load |

### Per-Annotation Metadata Fields
| Field | Type | Notes |
|---|---|---|
| id | string | Unique annotation ID |
| image_id | string | References image |
| category_id | int | 0 = empty |
| bbox | [x,y,w,h] | Absolute pixels, top-left origin |
| area | float | Approx. (w*h)/2 |
| count | int | Number of individuals |

### Additional Available Data Files
- **GPS coordinates:** Separate `gps_locations.json`, obfuscated within 1km
- **MegaDetector results:** Pre-computed v4/v5 detections from LILA
- **Taxonomy mapping CSV:** Maps categories to iNaturalist taxonomy
- **Split file:** Train/val/test split by camera location

---

## 4. Subset Strategy

- Sample ~5-10K images from the ~66K with bounding boxes
- **Stratified sampling** with minimum floor per class (~20 images, or all
  available for rare classes like badger/fox)
- Reduce empty image ratio to ~20-30% (vs. 70% in source)
- Split by camera location (canonical approach for this dataset)
- One-time manual step: run script, produce fixed image list

---

## 5. Data Pipeline Steps

1. Download annotation JSON + images from LILA BC
2. Filter to images with bounding box annotations (~66K)
3. Stratified sampling to create ~5-10K subset
4. Convert COCO format to YOLO format
5. Upload subset images to S3
6. Register dataset in MLflow (name, version, S3 path, config)

### COCO to YOLO Conversion

COCO: `{"bbox": [x, y, width, height]}` -- absolute pixels, top-left origin

YOLO: `class_id center_x center_y width height` -- normalized [0,1], center

```
center_x = (x + width/2) / image_width
center_y = (y + height/2) / image_height
norm_width = width / image_width
norm_height = height / image_height
```

Ultralytics expected directory structure:
```
dataset/
|-- images/
|   |-- train/
|   +-- val/
+-- labels/
    |-- train/
    +-- val/
```
Plus a `data.yaml` with paths and class names.

> Note: this COCO→YOLO normalized-center conversion is a **training-internal**
> transform. It is distinct from the dashboard's stored/served box format, which
> is `xyxy` end-to-end — see [../contracts/bbox-format.md](../contracts/bbox-format.md).
> The YOLO-index → COCO-`category_id` map produced here must be preserved for
> serving (see [../contracts/serving-io.md](../contracts/serving-io.md)).

---

## 6. MLflow Integration

The MLflow interfaces are pinned as cross-repo contracts; this section links to
them rather than restating (the original draft here predated those contracts and
referred to deprecated MLflow *stages* and the `mlflow.data` Dataset API, both
since corrected):

- **Model Registry** — alias-based promotion; log the `architecture` string plus
  training metrics (mAP / precision / recall / per-class AP). Contract:
  [../contracts/model-registry.md](../contracts/model-registry.md).
- **Dataset Tracking** — the `datasets`-experiment convention (name, version, S3
  path to annotation file + images, sampling config). Contract:
  [../contracts/dataset-conventions.md](../contracts/dataset-conventions.md).
- **Model Serving** — the pyfunc wire format (image data in, detections out).
  Contract: [../contracts/serving-io.md](../contracts/serving-io.md).
- **MLflow placement / backend store / serving topology** — contract:
  [../contracts/mlflow-topology.md](../contracts/mlflow-topology.md).

**Experiment tracking** (not a cross-repo contract): log every training run with
full config and metrics; compare runs in the MLflow UI.

---

## 7. How the Dashboard Consumes This

The dashboard (separate repo) is a **pure consumer** of MLflow. The three
interfaces it relies on are the contracts linked in §6:
1. **Models** — resolve the served model via the registry
   ([model-registry](../contracts/model-registry.md)).
2. **Datasets** — discover via the `datasets`-experiment convention
   ([dataset-conventions](../contracts/dataset-conventions.md)).
3. **Inference** — send images to the serving endpoint, get predictions
   ([serving-io](../contracts/serving-io.md)).

When a dataset is first selected in the dashboard, the backend:
1. Fetches dataset info from MLflow (S3 path)
2. Downloads COCO annotation file from S3
3. Parses annotations and loads images/ground-truth into SQLite
4. Dashboard can now query and filter via SQLite

The dashboard does NOT:
- Trigger training runs
- Modify datasets
- Access training experiment history

---

## 8. Infrastructure Needs

### MLflow Server
- Tracking server (HTTP API on port 5000)
- Backend store: SQLite or Postgres for run/experiment metadata
- Artifact store: S3 bucket for model weights and dataset files
- Model serving: separate process/port per served model

> Placement and backend store are now **decided** (dedicated EC2, SQLite) — see
> [../contracts/mlflow-topology.md](../contracts/mlflow-topology.md) and
> [../infra/CONTEXT.md](../infra/CONTEXT.md).

### GPU Training (AWS)
- Spot instance (g4dn.xlarge or p3.2xlarge) for cost savings
- Terraform-provisioned, terminated after training completes
- Script: pull data from S3 -> train -> log to MLflow -> shutdown
- Estimated cost: $0.50-1.50/hr spot, ~$2-6 per training run

### Local Training (Apple Silicon)
- MPS backend for PyTorch/Ultralytics
- 10-24x slower than NVIDIA, fine for prototyping on subset
- Same MLflow tracking server (must be network-accessible)

---

## 9. Key Gotchas

- Only ~66K of 243K images have bboxes -- must sample from these
- 70% empty images -- downsample aggressively for training
- ~5% annotation error rate (noted by dataset authors)
- Different cameras use different flash types (infrared vs. white flash)
- Categories vary in taxonomy level (species vs. genus vs. order)
- YOLO expects normalized center-format bboxes, COCO uses absolute top-left
- MPS training is significantly slower than CUDA -- keep subset small for local
- Location-based splits mean train/val cameras are disjoint (tests generalization)
