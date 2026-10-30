"""
sort.py - THE MAIN APP: Plastic Bottle Sorting Assistant.

    python sort.py --image samples/bottle.jpg     # one photo
    python sort.py --folder samples/              # every photo in a folder
    python sort.py                                # live webcam

Options:
    --rules     force the rule-based classifier (no trained model needed)
    --no-yolo   use classical contour detection instead of YOLO

Pipeline: detect bottles (YOLO / contours) -> crop each bottle ->
classify (MobileNetV2 or rules) -> show which bin it goes to.
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import cv2

import config
from camera_utils import open_camera, read_frame
from cv_pipeline import rule_based_classify
from detector import BottleDetector


class Sorter:
    def __init__(self, use_yolo=True, force_rules=False):
        self.detector = BottleDetector(use_yolo=use_yolo)
        self.cnn = None
        if not force_rules and config.MODEL_PATH.exists():
            try:
                from classifier import BottleClassifier
                self.cnn = BottleClassifier()
            except Exception as e:
                print(f"[sorter] Could not load CNN ({e}). Using rule-based classifier.")
        self.classifier_name = "MobileNetV2" if self.cnn else "Rule-based"
        print(f"Detector: {self.detector.mode} | Classifier: {self.classifier_name}")

    def classify(self, crop):
        if self.cnn:
            return self.cnn.predict(crop)
        label, conf, _ = rule_based_classify(crop)
        return label, conf

    def process(self, frame, whole_image_fallback=False):
        """Returns annotated frame and list of (label, conf, box)."""
        h, w = frame.shape[:2]
        boxes = self.detector.detect(frame)
        if not boxes and whole_image_fallback:       # single close-up bottle photo
            boxes = [(0, 0, w, h, None)]

        results = []
        out = frame.copy()
        for x1, y1, x2, y2, _ in boxes:
            pad_x, pad_y = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            label, conf = self.classify(crop)
            if conf < config.MIN_CLASSIFY_CONF:
                bin_name, color = config.DEFAULT_BIN
            else:
                bin_name, color = config.BINS.get(label, config.DEFAULT_BIN)
            results.append((label, conf, (x1, y1, x2, y2)))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            text = f"{bin_name} {conf * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(th + 38, y1)                     # stay below the summary bar
            tx = min(x1, max(0, w - tw - 8))          # stay inside the image
            cv2.rectangle(out, (tx, ty - th - 8), (tx + tw + 8, ty), color, -1)
            cv2.putText(out, text, (tx + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2)

        # Summary bar at the top
        counts = Counter(r[0] for r in results)
        summary = f"Bottles: {len(results)}  " + "  ".join(
            f"{config.BINS[c][0].split(' - ')[0]}:{counts.get(c, 0)}" for c in config.CLASSES)
        cv2.rectangle(out, (0, 0), (w, 30), (30, 30, 30), -1)
        cv2.putText(out, summary, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out, results


def show(window, img):
    try:
        cv2.imshow(window, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # no display available - result is still saved


def run_images(sorter, paths, display=True):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"skip (cannot read): {p}")
            continue
        out, results = sorter.process(img, whole_image_fallback=True)
        save_path = config.OUTPUT_DIR / f"sorted_{Path(p).stem}.jpg"
        cv2.imwrite(str(save_path), out)
        print(f"\n{p}:")
        for label, conf, box in results:
            bin_name = (config.BINS.get(label, config.DEFAULT_BIN)[0]
                        if conf >= config.MIN_CLASSIFY_CONF else config.DEFAULT_BIN[0])
            print(f"  {label:12s} {conf * 100:5.1f}%  -> {bin_name}")
        print(f"  saved -> {save_path}")
        if display:
            show("Plastic Bottle Sorter", out)


def run_webcam(sorter, camera=None):
    cap = open_camera(camera)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    prev = time.time()
    print("Press 's' to save a snapshot, 'q' to quit.")
    while True:
        frame = read_frame(cap)
        if frame is None:
            print("Camera stopped sending frames.")
            break
        out, _ = sorter.process(frame)
        now = time.time()
        fps = 1.0 / max(1e-6, now - prev)
        prev = now
        cv2.putText(out, f"FPS {fps:.1f} | {sorter.classifier_name}", (8, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Plastic Bottle Sorter (q to quit)", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = config.OUTPUT_DIR / f"snapshot_{int(now)}.jpg"
            cv2.imwrite(str(path), out)
            print(f"saved {path}")
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Plastic Bottle Sorting Assistant")
    ap.add_argument("--image", help="path to one image")
    ap.add_argument("--folder", help="folder of images")
    ap.add_argument("--rules", action="store_true", help="use rule-based classifier")
    ap.add_argument("--no-yolo", action="store_true", help="use contour detection")
    ap.add_argument("--no-show", action="store_true", help="don't open a window")
    ap.add_argument("--camera", type=int, default=None, help="camera number for live mode")
    args = ap.parse_args()

    sorter = Sorter(use_yolo=not args.no_yolo, force_rules=args.rules)
    if args.image:
        run_images(sorter, [args.image], display=not args.no_show)
    elif args.folder:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in exts)
        run_images(sorter, paths, display=not args.no_show)
    else:
        run_webcam(sorter, args.camera)


if __name__ == "__main__":
    main()
