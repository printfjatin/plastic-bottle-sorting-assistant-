"""
capture_dataset.py - Build your own dataset with the laptop webcam (Unit I: image acquisition).

    python capture_dataset.py
    python capture_dataset.py --camera 1     # if camera 0 is the wrong one

Hold a bottle inside the green box, then press:
    1 = clear_pet    2 = colored_pet    3 = opaque_hdpe    q = quit
Only the area inside the box is saved, into data/<class>/.
Tip: 80-150 photos per class, different angles, lighting and backgrounds.
"""
import argparse
import time

import cv2

import config
from camera_utils import open_camera, read_frame


def guide_box(frame):
    h, w = frame.shape[:2]
    bw, bh = int(w * 0.35), int(h * 0.85)
    x1, y1 = (w - bw) // 2, (h - bh) // 2
    return x1, y1, x1 + bw, y1 + bh


def main():
    ap = argparse.ArgumentParser(description="Capture bottle photos for training")
    ap.add_argument("--camera", type=int, default=None,
                    help="camera number (default: first one that works)")
    args = ap.parse_args()

    for c in config.CLASSES:
        (config.DATA_DIR / c).mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.camera)
    keys = {ord(str(i + 1)): c for i, c in enumerate(config.CLASSES)}
    print("Video window open (check behind other windows / in the Dock if you can't see it).")

    while True:
        frame = read_frame(cap)
        if frame is None:
            print("Camera stopped sending frames.")
            break
        x1, y1, x2, y2 = guide_box(frame)
        view = frame.copy()
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        counts = [f"{i + 1}:{c}={len(list((config.DATA_DIR / c).glob('*.jpg')))}"
                  for i, c in enumerate(config.CLASSES)]
        cv2.putText(view, "  ".join(counts), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        cv2.putText(view, "press 1/2/3 to save, q to quit", (10, view.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("Capture dataset", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in keys:
            cls = keys[key]
            path = config.DATA_DIR / cls / f"{cls}_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(path), frame[y1:y2, x1:x2])
            print(f"saved {path.name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
