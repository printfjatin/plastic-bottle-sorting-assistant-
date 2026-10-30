"""
visualize_pipeline.py - Shows EVERY classical CV step on one image (Units I-III).

    python visualize_pipeline.py samples/bottle.jpg

Saves outputs/pipeline_<name>.png - a 4x4 grid you can put straight into
your report / PPT and explain in the viva.
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import cv_pipeline as cv


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python visualize_pipeline.py <image>")
    path = Path(sys.argv[1])
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    scale = 640 / max(img.shape[:2])                       # keep it fast
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv.to_gray(img)
    eq = cv.equalize_histogram(gray)
    blur = cv.denoise(gray, "gaussian")
    sobel = cv.sobel_edges(blur)
    canny = cv.canny_edges(blur)
    kps, _ = cv.orb_keypoints(gray)
    fast = cv.fast_keypoints(gray)
    pyr = cv.gaussian_pyramid(img, 2)[2]
    otsu = cv.otsu_threshold(blur)
    morph = cv.morph_clean(255 - otsu)
    region = cv.region_growing(blur)
    kmeans, _, _ = cv.kmeans_segment(img, 3)
    meanshift = cv.mean_shift_segment(img)
    slic = cv.slic_superpixels(img)

    detected = img.copy()
    for x1, y1, x2, y2 in cv.find_bottle_boxes(img):
        crop = img[y1:y2, x1:x2]
        label, conf, _ = cv.rule_based_classify(crop)
        color = config.BINS[label][1]
        cv2.rectangle(detected, (x1, y1), (x2, y2), color, 3)
        cv2.putText(detected, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    panels = [
        ("1. Original", img), ("2. Grayscale", gray), ("3. Histogram equalized", eq),
        ("4. Gaussian filter", blur), ("5. Sobel edges", sobel), ("6. Canny edges", canny),
        (f"7. ORB keypoints ({len(kps)})", cv.draw_keypoints(img, kps)),
        (f"8. FAST corners ({len(fast)})", cv.draw_keypoints(img, fast)),
        ("9. Pyramid level 2 (1/4)", pyr), ("10. Otsu threshold", otsu),
        ("11. Morphology (open+close)", morph), ("12. Region growing", region),
        ("13. K-means (k=3)", kmeans), ("14. Mean-shift", meanshift),
        ("15. SLIC superpixels", slic if slic is not None else np.zeros_like(img)),
        ("16. Contour detection + rules", detected),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for ax, (title, im) in zip(axes.flat, panels):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"Plastic Bottle Sorter - CV pipeline on {path.name}", fontsize=15)
    fig.tight_layout()
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / f"pipeline_{path.stem}.png"
    fig.savefig(out, dpi=110)

    # Histogram before / after equalization (also Unit II)
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.hist(gray.ravel(), 256, range=(0, 256), alpha=0.6, label="original")
    ax2.hist(eq.ravel(), 256, range=(0, 256), alpha=0.6, label="equalized")
    ax2.set_title("Histogram equalization"); ax2.legend()
    fig2.tight_layout()
    fig2.savefig(config.OUTPUT_DIR / f"histogram_{path.stem}.png", dpi=110)
    print(f"saved {out}")
    print(f"saved {config.OUTPUT_DIR / f'histogram_{path.stem}.png'}")


if __name__ == "__main__":
    main()
