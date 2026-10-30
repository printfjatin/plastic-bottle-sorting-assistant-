"""
train.py - Train the MobileNetV2 bottle classifier (Unit IV: transfer learning + fine-tuning).

    python train.py

Phase 1: freeze the pre-trained MobileNet, train only the new last layer.
Phase 2: unfreeze the last few blocks and fine-tune with a small learning rate.
Saves: models/bottle_classifier.pt and outputs/training_curves.png
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

import config
from classifier import build_model, eval_transform, get_device, train_transform


def make_loaders():
    train_full = ImageFolder(config.DATA_DIR, transform=train_transform)
    val_full = ImageFolder(config.DATA_DIR, transform=eval_transform)
    n = len(train_full)
    if n < 10:
        raise SystemExit("Not enough images in data/. Run capture_dataset.py first "
                         "(aim for 80+ photos per class).")
    g = torch.Generator().manual_seed(config.SEED)
    idx = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * config.VAL_SPLIT))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=0)
    print(f"Classes: {train_full.classes}")
    print(f"Images: {n} total -> {len(train_idx)} train / {len(val_idx)} val")
    return train_loader, val_loader, train_full.classes, val_idx


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            count += x.size(0)
    return total_loss / count, correct / count


def main():
    device = get_device()
    print(f"Using device: {device}")
    train_loader, val_loader, classes, val_idx = make_loaders()

    model = build_model(len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_acc = 0.0
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    def save_if_best(val_acc):
        nonlocal best_acc
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "val_idx": val_idx}, config.MODEL_PATH)

    phases = [
        ("Phase 1: train classifier head", config.EPOCHS_HEAD, config.LR_HEAD, 0),
        ("Phase 2: fine-tune last 4 blocks", config.EPOCHS_FINETUNE, config.LR_FINETUNE, 4),
    ]
    for name, epochs, lr, unfreeze_blocks in phases:
        print(f"\n=== {name} ===")
        for p in model.features.parameters():
            p.requires_grad = False
        if unfreeze_blocks:
            for p in model.features[-unfreeze_blocks:].parameters():
                p.requires_grad = True
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tl, ta = run_epoch(model, train_loader, criterion, device, optimizer)
            vl, va = run_epoch(model, val_loader, criterion, device)
            for k, v in zip(history, (ta, va, tl, vl)):
                history[k].append(v)
            save_if_best(va)
            print(f"epoch {epoch}/{epochs}  train_acc={ta:.3f}  val_acc={va:.3f}  "
                  f"train_loss={tl:.3f}  val_loss={vl:.3f}  ({time.time() - t0:.0f}s)")

    # Training curves for your report
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_acc"], label="train")
    ax[0].plot(history["val_acc"], label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["train_loss"], label="train")
    ax[1].plot(history["val_loss"], label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    for a in ax:
        a.axvline(config.EPOCHS_HEAD - 0.5, color="gray", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "training_curves.png", dpi=150)

    print(f"\nBest validation accuracy: {best_acc:.3f}")
    print(f"Model saved to {config.MODEL_PATH}")
    print(f"Curves saved to {config.OUTPUT_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    main()
