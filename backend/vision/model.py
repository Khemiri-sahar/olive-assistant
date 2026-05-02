"""
vision/model.py — MobileNetV3-based CNN for olive leaf disease classification.

Uses transfer learning from ImageNet pretrained MobileNetV3-Large.
Chosen for speed (< 100ms inference on CPU) while maintaining accuracy.

Classes: Healthy / Peacock Eye / Anthracnose / Verticillium / Cercospora
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

# ── Disease class configuration ───────────────────────────────────────────────
NUM_CLASSES = 5

CLASS_NAMES = {
    0: "Healthy",
    1: "Peacock Eye",
    2: "Anthracnose",
    3: "Verticillium Wilt",
    4: "Cercospora Leaf Spot",
}

# ── Image transforms ──────────────────────────────────────────────────────────
# Training: aggressive augmentation to handle real field photos
TRAIN_TRANSFORM = T.Compose([
    T.RandomResizedCrop(224, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(30),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    T.RandomGrayscale(p=0.05),
    # Simulate JPEG compression artifacts
    T.ToTensor(),
    T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Inference: standard resize + normalize only
INFERENCE_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build MobileNetV3-Large with custom classification head."""
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    # Replace classifier head for our number of classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


class OliveCNN:
    """
    Inference wrapper for the trained olive disease CNN.
    Loads weights from disk, runs inference on PIL images.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.50,
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"CNN using device: {self.device}")

        self.model = build_model(pretrained=False)

        if model_path and Path(model_path).exists():
            logger.info(f"Loading CNN weights from {model_path}")
            state = torch.load(model_path, map_location=self.device)
            # Support both raw state_dict and checkpoint dicts
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            logger.info("CNN weights loaded")
        else:
            logger.warning(
                f"⚠️  No CNN weights at {model_path} — model is UNTRAINED (random predictions)"
            )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Dict:
        """
        Run inference on a PIL Image.

        Returns dict with:
            class_id:    int
            class_name:  str (English)
            confidence:  float [0, 1]
            low_conf:    bool — True if confidence < threshold
            all_scores:  dict of {class_name: probability}
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        class_id   = int(np.argmax(probs))
        confidence = float(probs[class_id])

        return {
            "class_id":   class_id,
            "class_name": CLASS_NAMES.get(class_id, "Unknown"),
            "confidence": confidence,
            "low_conf":   confidence < self.confidence_threshold,
            "all_scores": {
                CLASS_NAMES[i]: float(probs[i])
                for i in range(len(probs))
            },
        }

    def predict_bytes(self, image_bytes: bytes) -> Dict:
        """Convenience method: run inference on raw image bytes."""
        import io
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict(image)