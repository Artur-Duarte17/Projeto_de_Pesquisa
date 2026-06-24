from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from retrieval_common import cosine_scores, l2_normalize, resolve_stored_path


def torch_device(device: str) -> torch.device:
    if device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    topk: int = 10,
    query_path: Path | None = None,
) -> pd.DataFrame:
    scores = cosine_scores(embeddings, query_emb)
    rows = metadata.copy()
    rows["score"] = scores
    if query_path is not None:
        q_resolved = query_path.resolve()
        keep = []
        for p in rows["image_path"].astype(str):
            keep.append(resolve_stored_path(p).resolve() != q_resolved)
        rows = rows.loc[keep].copy()
    rows = rows.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
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
