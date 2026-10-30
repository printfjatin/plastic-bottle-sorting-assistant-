"""
classifier.py - MobileNetV2 bottle classifier (Unit IV: CNNs + transfer learning).

MobileNetV2 is pre-trained on ImageNet. We replace its last layer with a
3-class layer (clear / coloured / opaque) and train it on our photos.
Runs on the Apple M-series GPU (MPS) automatically if available.
"""
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

import config

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet stats

# Augmentation for training. NOTE: no hue jitter - colour IS the label here!
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_device():
    if torch.backends.mps.is_available():   # Apple Silicon GPU
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes, pretrained=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


class BottleClassifier:
    """Loads the trained model and predicts the class of one bottle crop."""

    def __init__(self, path=config.MODEL_PATH):
        self.device = get_device()
        ckpt = torch.load(path, map_location=self.device)
        self.classes = ckpt["classes"]
        self.model = build_model(len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, bgr_crop):
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        x = eval_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu()
        idx = int(probs.argmax())
        return self.classes[idx], float(probs[idx])
