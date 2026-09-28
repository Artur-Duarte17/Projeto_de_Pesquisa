from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from retrieval.adapters.image_io import open_image, oriented_rgb
from retrieval.domain.similarity import l2_normalize


def torch_device(device: str) -> torch.device:
    if device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_torch_device(requested: str, actual: torch.device) -> str:
    if actual.type == "cuda":
        name = torch.cuda.get_device_name(actual)
        return f"requested={requested} actual=cuda device={name}"
    if requested.lower() == "cuda":
        return "requested=cuda actual=cpu reason=torch.cuda.is_available() is False"
    return "requested=cpu actual=cpu"


def build_resnet50_feature_extractor(device: str = "cpu", weights_name: str = "imagenet"):
    import torch.nn as nn
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.DEFAULT if weights_name != "none" else None
    model = resnet50(weights=weights)
    extractor = nn.Sequential(*list(model.children())[:-1])
    dev = torch_device(device)
    extractor.eval().to(dev)
    transform = weights.transforms() if weights is not None else ResNet50_Weights.DEFAULT.transforms()
    return extractor, transform, dev


def extract_global_embedding(image_path: Path, extractor, transform, device) -> np.ndarray:
    with open_image(image_path) as image:
        img = oriented_rgb(image, image_path)
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = extractor(x).flatten(1).cpu().numpy()[0]
    return l2_normalize(feat.astype(np.float32))


def extract_global_embeddings_batch(
    tensors: list[torch.Tensor],
    extractor,
    device: torch.device,
) -> np.ndarray:
    """Extract normalized descriptors for one image batch, preserving input order."""
    if not tensors:
        return np.empty((0, 0), dtype=np.float32)
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        features = extractor(batch).flatten(1)
        features = functional.normalize(features, p=2, dim=1)
    return features.cpu().numpy().astype(np.float32)
