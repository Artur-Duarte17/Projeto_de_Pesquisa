from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from retrieval_common import cosine_scores, image_ids_for_query_path, l2_normalize


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
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = extractor(x).flatten(1).cpu().numpy()[0]
    return l2_normalize(feat.astype(np.float32))


def load_global_index(index_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    emb_path = index_dir / "global_embeddings.npy"
    meta_path = index_dir / "global_metadata.csv"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing global index files in {index_dir}")
    E = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    if len(E) != len(meta):
        raise ValueError(f"Global index mismatch: {len(E)} embeddings vs {len(meta)} metadata rows")
    return E, meta


def search_global_index(
    query_emb: np.ndarray,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    topk: int | None = 10,
    query_path: Path | None = None,
    exclude_image_ids: set[str] | None = None,
) -> pd.DataFrame:
    scores = cosine_scores(embeddings, query_emb)
    rows = metadata.copy()
    rows["score"] = scores
    excluded = set(exclude_image_ids or set())
    if query_path is not None:
        excluded.update(image_ids_for_query_path(query_path, metadata))
    if excluded:
        rows = rows[~rows["image_id"].astype(str).isin(excluded)].copy()
    rows = rows.sort_values("score", ascending=False)
    if topk is not None:
        if topk < 0:
            raise ValueError("topk must be non-negative or None")
        rows = rows.head(topk)
    rows = rows.reset_index(drop=True)
    rows.insert(0, "rank", np.arange(1, len(rows) + 1))
    rows["matched_face_id"] = ""
    rows["bbox"] = ""
    return rows[
        [
            "rank",
            "image_id",
            "image_path",
            "score",
            "matched_face_id",
            "bbox",
        ]
    ]
