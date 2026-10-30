"""
cv_pipeline.py - Classical Computer Vision functions (Units I, II, III).

Every function here is a small, separate step so you can explain each one
in your viva. visualize_pipeline.py runs all of them on one image.
"""
import cv2
import numpy as np


# =====================================================================
# UNIT II - Image preprocessing
# =====================================================================
def to_gray(img):
    """BGR colour image -> single-channel grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize_histogram(gray):
    """Global histogram equalization - spreads out intensity values."""
    return cv2.equalizeHist(gray)


def clahe(gray):
    """Adaptive (local) histogram equalization - better for uneven lighting."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def denoise(img, method="gaussian"):
    """Filtering: gaussian (smooth), median (salt-pepper noise), bilateral (keeps edges)."""
    if method == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 0)
    if method == "median":
        return cv2.medianBlur(img, 5)
    if method == "bilateral":
        return cv2.bilateralFilter(img, 9, 75, 75)
    raise ValueError(f"unknown filter: {method}")


# =====================================================================
# UNIT II - Edge detection
# =====================================================================
def sobel_edges(gray):
    """Sobel: gradient in x and y, combined into magnitude."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def canny_edges(gray, low=50, high=150):
    """Canny: smoothing + gradient + non-max suppression + hysteresis."""
    return cv2.Canny(gray, low, high)


# =====================================================================
# UNIT II - Corner / keypoint detection
# (SURF is patented and not in standard OpenCV, so we use ORB + FAST,
#  which are free and much faster.)
# =====================================================================
def fast_keypoints(gray, threshold=25):
    fast = cv2.FastFeatureDetector_create(threshold=threshold)
    return fast.detect(gray, None)


def orb_keypoints(gray, n_features=300):
    """ORB = FAST keypoints + BRIEF descriptors (rotation invariant)."""
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def draw_keypoints(img, keypoints):
    return cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)


# =====================================================================
# UNIT II - Image pyramids & morphology
# =====================================================================
def gaussian_pyramid(img, levels=3):
    """Returns [original, 1/2, 1/4, ...] - multi-scale representation."""
    pyramid = [img]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def morph_clean(mask, k=5):
    """Opening removes small noise, closing fills small holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)


# =====================================================================
# UNIT III - Segmentation
# =====================================================================
def otsu_threshold(gray):
    """Otsu automatically picks the best global threshold."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def adaptive_threshold(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 5)


def region_growing(gray, seed=None, tolerance=12):
    """Region growing from a seed pixel (default: image centre) using flood fill."""
    h, w = gray.shape
    seed = seed or (w // 2, h // 2)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(gray.copy(), mask, seed, 255,
                  loDiff=tolerance, upDiff=tolerance, flags=flags)
    return mask[1:-1, 1:-1]


def kmeans_segment(img, k=3):
    """K-means clustering on pixel colours. Returns (segmented image, labels, centres)."""
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img.shape)
    return segmented, labels.reshape(img.shape[:2]), centers


def mean_shift_segment(img):
    """Mean-shift filtering - groups pixels with similar colour and position."""
    return cv2.pyrMeanShiftFiltering(img, sp=21, sr=51)


def slic_superpixels(img, n_segments=200):
    """SLIC superpixels (needs scikit-image). Returns image with boundaries drawn."""
    try:
        from skimage.segmentation import slic, mark_boundaries
    except ImportError:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segments = slic(rgb, n_segments=n_segments, compactness=10, start_label=1)
    marked = mark_boundaries(rgb, segments)
    return cv2.cvtColor((marked * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# =====================================================================
# UNIT III - Classical object detection (fallback when YOLO isn't used)
# =====================================================================
def find_bottle_boxes(img, min_area_ratio=0.02, max_boxes=5):
    """
    Find bottle-like objects using edges + morphology + contours.
    Works best on a plain background (table / sheet of paper).
    Returns list of (x1, y1, x2, y2).
    """
    h, w = img.shape[:2]
    gray = denoise(clahe(to_gray(img)), "gaussian")
    edges = canny_edges(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.dilate(edges, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area_ratio * w * h:
            continue
        if bw * bh > 0.95 * w * h:          # whole frame - not an object
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect < 1.3:                     # bottles are long (standing or lying)
            continue
        boxes.append((x, y, x + bw, y + bh))
        if len(boxes) >= max_boxes:
            break
    return boxes


# =====================================================================
# Rule-based classifier (works with NO training - the "easy" baseline)
# =====================================================================
def center_region(crop, margin=0.15):
    h, w = crop.shape[:2]
    my, mx = int(h * margin), int(w * margin)
    region = crop[my:h - my, mx:w - mx]
    return region if region.size else crop


def dominant_color(crop, k=3):
    """Most common colour (BGR) in the bottle using K-means."""
    small = cv2.resize(center_region(crop), (64, 64))
    _, labels, centers = kmeans_segment(small, k)
    counts = np.bincount(labels.flatten(), minlength=k)
    return tuple(int(v) for v in centers[np.argmax(counts)])


def extract_features(crop):
    """Hand-crafted colour + texture features of a bottle crop."""
    region = center_region(crop)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    coloured_ratio = float(np.mean((s > 80) & (v > 40)))   # strongly coloured pixels
    edges = canny_edges(denoise(to_gray(region)), 50, 150)
    return {
        "coloured_ratio": coloured_ratio,
        "mean_saturation": float(s.mean()),
        "mean_brightness": float(v.mean()),
        "brightness_std": float(v.std()),
        "edge_density": float(np.mean(edges > 0)),        # clear plastic = many reflections
        "dominant_bgr": dominant_color(crop),
    }


def rule_based_classify(crop):
    """
    Simple rules on HSV colour + edges:
      - lots of saturated pixels      -> coloured PET (green / blue bottles)
      - bright, smooth, few edges     -> opaque HDPE (milk / shampoo bottles)
      - otherwise (see-through)       -> clear PET (water bottles)
    Returns (label, confidence, features).
    """
    f = extract_features(crop)
    if f["coloured_ratio"] > 0.35:
        return "colored_pet", min(0.99, 0.5 + f["coloured_ratio"] / 2), f
    if f["mean_brightness"] > 150 and f["brightness_std"] < 45 and f["edge_density"] < 0.08:
        return "opaque_hdpe", 0.7, f
    return "clear_pet", 0.6, f
