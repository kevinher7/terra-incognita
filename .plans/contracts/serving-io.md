# Contract: Serving I/O (pyfunc inference wire format)

**Governs:** the request/response wire contract for model inference — how the
dashboard sends images to the served model and what shape of detections it gets
back.

## Spec

The model is served via `mlflow models serve` (REST) as a custom `pyfunc`. The
wire contract:

- **Input:** JSON, one record per image, batchable. Image bytes are
  **base64-encoded** in the request body. (Chosen over passing an S3 URI so the
  serving model stays a pure, stateless, offline-testable function with no
  S3/IAM coupling — the dashboard has S3 access, the serving process does not.)
- **Inference params** (MLflow signature `params`): `conf` (default `0.25`),
  `iou` (default `0.45`), `max_det` (default `300`).
- **Output:** per image, `width`, `height`, and a list of detections:
  ```json
  {
    "width": 2048, "height": 1536,
    "detections": [
      {"bbox_xyxy": [x1, y1, x2, y2], "category_id": 7, "class_name": "raccoon", "score": 0.83}
    ]
  }
  ```
  Boxes are **absolute-pixel `xyxy`** (see [bbox-format.md](./bbox-format.md)).
  The pyfunc **translates the YOLO contiguous index → real COCO `category_id`**
  before returning (the index→`category_id` map is stored with the model
  artifact, and must match the dataset the model was trained on), so the
  dashboard can join predictions to its COCO `Category` table directly.
  `class_name` is included for convenience but **`category_id` is the join key**.

**Packaging.** The serving image is built via `mlflow models build-docker` so
local and deployed serving are byte-identical, with the model baked into the
image at build time.

## Depended on by

- **dashboard** (`terra-incognita`'s consumer) — POSTs base64 image(s), reads
  `bbox_xyxy` + `category_id` + `class_name` + `score`.
- **training** (`terra-incognita`) — implements the pyfunc wrapper that honors
  this contract; this wrapper is the single most important artifact in that repo.
- **infra** — provisions the long-running serving container; its runtime IAM is
  ~none because images arrive base64 and the model is baked in
  (see [mlflow-topology.md](./mlflow-topology.md)).

## Rule

Do not redefine this elsewhere. Reference this file. To change the serving I/O
contract, change it here.
