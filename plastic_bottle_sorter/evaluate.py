"""
evaluate.py - Model evaluation (Unit IV: accuracy, precision/recall, confusion matrix, IoU).

    python evaluate.py

Compares the trained CNN against the rule-based (classical CV) method on the
same validation images - a nice "Classical CV vs Deep Learning" table for
your report.
"""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from torchvision.datasets import ImageFolder

import config
from classifier import BottleClassifier
from cv_pipeline import rule_based_classify
from detector import iou


def save_confusion(y_true, y_pred, classes, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    if not config.MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run train.py first.")
    clf = BottleClassifier()
    ckpt = torch.load(config.MODEL_PATH, map_location="cpu")
    dataset = ImageFolder(config.DATA_DIR)
    samples = [dataset.samples[i] for i in ckpt["val_idx"] if i < len(dataset.samples)]

    y_true, y_cnn, y_rules = [], [], []
    for path, label_idx in samples:
        img = cv2.imread(path)
        if img is None:
            continue
        y_true.append(dataset.classes[label_idx])
        y_cnn.append(clf.predict(img)[0])
        y_rules.append(rule_based_classify(img)[0])

    classes = dataset.classes
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Validation images: {len(y_true)}\n")
    for name, preds in [("MobileNetV2 (deep learning)", y_cnn),
                        ("Rule-based (classical CV)", y_rules)]:
        print(f"===== {name} =====")
        print(f"Accuracy: {accuracy_score(y_true, preds):.3f}")
        print(classification_report(y_true, preds, labels=classes, zero_division=0))

    save_confusion(y_true, y_cnn, classes, "MobileNetV2",
                   config.OUTPUT_DIR / "confusion_matrix_cnn.png")
    save_confusion(y_true, y_rules, classes, "Rule-based",
                   config.OUTPUT_DIR / "confusion_matrix_rules.png")
    print(f"Confusion matrices saved in {config.OUTPUT_DIR}/")

    # IoU example (detection metric). mAP for the YOLO detector is explained in README.
    gt, pred = (50, 40, 150, 300), (60, 50, 160, 290)
    print(f"\nIoU example: ground truth {gt} vs predicted {pred} -> IoU = {iou(gt, pred):.3f}")


if __name__ == "__main__":
    main()
