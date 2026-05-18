from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from models import BaselineCNN, ImprovedCNN, build_resnet18

CLASS_NAMES = ["benign", "malignant"]
IMAGE_SIZE = (224, 224)

# Baseline + ResNet were trained with ImageNet normalization (notebook sections 4 & 10)
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Improved CNN was trained with 0.5 normalization (notebook section 9)
IMPROVED_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

MODEL_REGISTRY = {
    "baseline": {
        "factory": BaselineCNN,
        "weights": "best_baseline_model.pth",
        "transform": IMAGENET_TRANSFORM,
    },
    "improved": {
        "factory": ImprovedCNN,
        "weights": "best_improved_model.pth",
        "transform": IMPROVED_TRANSFORM,
    },
    "resnet": {
        "factory": lambda: build_resnet18(len(CLASS_NAMES)),
        "weights": "resnet18_melanoma_best.pth",
        "transform": IMAGENET_TRANSFORM,
    },
}

DEFAULT_MODEL = "improved"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(base_dir, device=None):
    device = device or get_device()
    base_dir = Path(base_dir)
    loaded = {}

    for key, spec in MODEL_REGISTRY.items():
        weights_path = base_dir / spec["weights"]
        if not weights_path.is_file():
            continue
        model = spec["factory"]()
        try:
            state = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        loaded[key] = model

    return loaded, device


def predict_tensor(output: torch.Tensor):
    """Map model logits to label. benign=0, malignant=1 (ImageFolder alphabetical order)."""
    logits = output.squeeze()

    if logits.dim() == 0 or logits.numel() == 1:
        malignant_prob = torch.sigmoid(logits).item()
        if malignant_prob > 0.5:
            label = CLASS_NAMES[1]  # malignant
            confidence = malignant_prob
        else:
            label = CLASS_NAMES[0]  # benign
            confidence = 1.0 - malignant_prob
        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "malignant_probability": round(malignant_prob * 100, 2),
        }

    probs = torch.softmax(logits, dim=0)
    predicted_idx = int(torch.argmax(probs).item())
    malignant_prob = probs[1].item()
    return {
        "label": CLASS_NAMES[predicted_idx],
        "confidence": round(probs[predicted_idx].item() * 100, 2),
        "malignant_probability": round(malignant_prob * 100, 2),
    }


def run_inference(model_key: str, model, image: Image.Image, device):
    transform = MODEL_REGISTRY[model_key]["transform"]
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    return predict_tensor(output)
