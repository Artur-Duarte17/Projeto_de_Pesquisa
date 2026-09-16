from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from face_lib import search_face_index
from global_lib import search_global_index
from retrieval_common import cosine_to_unit


def fuse_cosine_score_matrices(
    face_scores: np.ndarray,
    global_scores: np.ndarray,
    face_available: np.ndarray,
    face_weight: float,
    global_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert cosine matrices to unit scores and combine them with fixed weights."""
    if face_scores.shape != global_scores.shape or face_scores.shape != face_available.shape:
        raise ValueError("Face, global, and availability matrices must have the same shape")
    if face_weight < 0.0 or global_weight < 0.0:
        raise ValueError("Fusion weights must be non-negative")
    if not np.isclose(face_weight + global_weight, 1.0, atol=1e-9):
        raise ValueError("Fusion weights must sum to one")

    face_unit = np.zeros(face_scores.shape, dtype=np.float32)
    face_unit[face_available] = np.asarray(
        cosine_to_unit(face_scores[face_available]), dtype=np.float32
    )
    global_unit = np.asarray(cosine_to_unit(global_scores), dtype=np.float32)
    fused = face_weight * face_unit + global_weight * global_unit
    return fused.astype(np.float32), face_unit, global_unit


def combine_face_global_results(
    face_results: pd.DataFrame,
    global_results: pd.DataFrame,
    face_weight: float = 0.7,
    global_weight: float = 0.3,
    topk: int | None = 10,
) -> pd.DataFrame:
    beta = global_weight
    alpha = face_weight
    records: dict[str, dict] = {}

    for row in global_results.itertuples(index=False):
        image_id = str(row.image_id)
        records.setdefault(
            image_id,
            {
                "image_id": image_id,
                "image_path": row.image_path,
                "face_score": 0.0,
                "global_score": 0.0,
                "matched_face_id": "",
                "bbox": "",
            },
        )
        records[image_id]["global_score"] = float(cosine_to_unit(float(row.score)))

    for row in face_results.itertuples(index=False):
        image_id = str(row.image_id)
        records.setdefault(
            image_id,
            {
                "image_id": image_id,
                "image_path": row.image_path,
                "face_score": 0.0,
                "global_score": 0.0,
                "matched_face_id": "",
                "bbox": "",
            },
        )
        records[image_id]["face_score"] = float(cosine_to_unit(float(row.score)))
        records[image_id]["matched_face_id"] = row.matched_face_id
        records[image_id]["bbox"] = row.bbox

    rows = []
    for rec in records.values():
        rec["score"] = alpha * rec["face_score"] + beta * rec["global_score"]
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "image_id",
                "image_path",
                "score",
                "face_score",
                "global_score",
                "matched_face_id",
                "bbox",
            ]
        )
    out = out.sort_values("score", ascending=False)
    if topk is not None:
        if topk < 0:
            raise ValueError("topk must be non-negative or None")
        out = out.head(topk)
    out = out.reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out[
        [
            "rank",
            "image_id",
            "image_path",
            "score",
            "face_score",
            "global_score",
            "matched_face_id",
            "bbox",
        ]
    ]


def search_fusion(
    query_path: Path,
    query_face_emb,
    query_global_emb,
    face_embeddings,
    face_metadata,
    global_embeddings,
    global_metadata,
    topk: int | None,
    face_weight: float,
    global_weight: float,
    threshold: float = -1.0,
    exclude_image_ids: set[str] | None = None,
) -> pd.DataFrame:
    face_limit = int(face_metadata["image_id"].nunique()) if not face_metadata.empty else topk
    global_limit = len(global_metadata) if not global_metadata.empty else topk
    face_results = search_face_index(
        query_face_emb,
        face_embeddings,
        face_metadata,
        topk=face_limit,
        threshold=threshold,
        query_path=query_path,
        exclude_image_ids=exclude_image_ids,
    )
    global_results = search_global_index(
        query_global_emb,
        global_embeddings,
        global_metadata,
        topk=global_limit,
        query_path=query_path,
        exclude_image_ids=exclude_image_ids,
    )
    return combine_face_global_results(
        face_results,
        global_results,
        face_weight=face_weight,
        global_weight=global_weight,
        topk=topk,
    )
