# Contract: Bounding-box format (`xyxy` end-to-end)

**Governs:** the single internal bounding-box representation used across serving
output, dashboard storage, and IoU math — and the only two points where
conversion is allowed.

## Spec

**Single internal format: `xyxy`.** Both `GroundTruth` and `Prediction` store
absolute-pixel `[x1, y1, x2, y2]`. The served YOLO model emits pixel `xyxy`, and
IoU/NMS math is the CV-standard in `xyxy` (torchvision `box_iou`, etc.). Mixing
`xywh` storage with `xyxy` math would force a conversion at every comparison — a
bug magnet.

Conversions happen at exactly two boundaries:

- **COCO ingest:** convert ground-truth `xywh → xyxy` **once** when parsing the
  COCO file into SQLite (`x2 = x + w`, `y2 = y + h`). This is the only conversion
  on ingest.
- **Display:** convert to whatever the renderer wants at the drawing boundary
  only.

**Storage columns.** `GroundTruth` and `Prediction` use
`bbox_x1, bbox_y1, bbox_x2, bbox_y2`. IoU is computed in `xyxy`.

**Serving output.** The pyfunc returns boxes as `bbox_xyxy = [x1, y1, x2, y2]` in
absolute pixels (see [serving-io.md](./serving-io.md)).

## Depended on by

- **dashboard** — stores GT + predictions as `bbox_x1/y1/x2/y2`; converts COCO GT
  `xywh→xyxy` once on ingest; computes IoU in `xyxy`; converts at display time.
- **training** (`terra-incognita`) — the pyfunc emits absolute-pixel `bbox_xyxy`.
  (Note: the COCO→YOLO *training* conversion to normalized center format is a
  separate, training-internal transform and does not affect this stored/served
  contract.)

## Rule

Do not redefine this elsewhere. Reference this file. To change the internal
box format, change it here.
