"""
detector.py - Finds bottles in an image (Unit III: object detection).

Uses a pre-trained YOLOv8-nano (trained on COCO, which already has a
"bottle" class), so no detector training is needed.
If ultralytics isn't installed, it falls back to the classical
edge + contour method from cv_pipeline.py.
"""
import config
from cv_pipeline import find_bottle_boxes


class BottleDetector:
    def __init__(self, use_yolo=True):
        self.model = None
        self.mode = "contour"
        if use_yolo:
            try:
                from ultralytics import YOLO
                self.model = YOLO(config.YOLO_WEIGHTS)
                self.mode = "yolo"
            except Exception as e:  # not installed / no internet for first download
                print(f"[detector] YOLO not available ({e}). Using contour detection.")

    def detect(self, img):
        """Returns list of (x1, y1, x2, y2, score)."""
        if self.model is not None:
            result = self.model(img, classes=[config.COCO_BOTTLE_ID],
                                conf=config.YOLO_CONF, verbose=False)[0]
            out = []
            for box, score in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                out.append((x1, y1, x2, y2, float(score)))
            return out
        return [(x1, y1, x2, y2, None) for (x1, y1, x2, y2) in find_bottle_boxes(img)]


def iou(box_a, box_b):
    """Intersection over Union of two boxes (x1, y1, x2, y2) - Unit IV metric."""
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
