"""
config.py - All project settings in one place.
Change values here instead of editing the other files.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"            # data/<class_name>/*.jpg
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "bottle_classifier.pt"

# ---- Sorting classes (folder names inside data/) ----
CLASSES = ["clear_pet", "colored_pet", "opaque_hdpe"]

# Which bin each class goes to, and the box colour (BGR for OpenCV)
BINS = {
    "clear_pet":   ("BIN A - Clear PET",      (255, 170, 0)),   # blue
    "colored_pet": ("BIN B - Coloured PET",   (0, 200, 0)),     # green
    "opaque_hdpe": ("BIN C - HDPE / Opaque",  (0, 210, 255)),   # yellow
}
DEFAULT_BIN = ("UNKNOWN - Manual check", (0, 0, 255))           # red

# ---- Deep learning (Unit IV) ----
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_HEAD = 5        # phase 1: train only the new classifier layer
EPOCHS_FINETUNE = 5    # phase 2: fine-tune last MobileNet blocks
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
VAL_SPLIT = 0.2
SEED = 42

# ---- Object detection (Unit III) ----
YOLO_WEIGHTS = "yolov8n.pt"   # downloads automatically the first time (~6 MB)
COCO_BOTTLE_ID = 39           # "bottle" class in the COCO dataset
YOLO_CONF = 0.35

# Below this confidence the bottle is sent to "manual check"
MIN_CLASSIFY_CONF = 0.45
