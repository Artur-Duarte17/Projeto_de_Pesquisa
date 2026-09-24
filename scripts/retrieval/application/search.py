from __future__ import annotations

import numpy as np
import pandas as pd

from retrieval.domain.fusion import combine_face_global_results
from retrieval.domain.similarity import cosine_scores


def rank_face_index(
    query_emb: np.ndarray,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    topk: int | None = 10,
    threshold: float = -1.0,
    excluded_image_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Rank one result per photograph; path/hash exclusions arrive from an adapter."""
    scores = cosine_scores(embeddings, query_emb)
    rows = metadata.copy()
    rows["score"] = scores

    excluded = set(excluded_image_ids or set())
    if excluded:
        rows = rows[~rows["image_id"].astype(str).isin(excluded)].copy()

    if threshold > -1.0:
        rows = rows[rows["score"] >= threshold].copy()

    rows = rows.sort_values("score", ascending=False)
    best = rows.groupby("image_id", as_index=False).head(1)
    best = best.sort_values("score", ascending=False)
    if topk is not None:
        if topk < 0:
            raise ValueError("topk must be non-negative or None")
        best = best.head(topk)
    best = best.reset_index(drop=True)
    best.insert(0, "rank", np.arange(1, len(best) + 1))
    best["matched_face_id"] = best["face_id"]
    best["bbox"] = (
        best["bbox_x1"].astype(str)
        + ","
        + best["bbox_y1"].astype(str)
        + ","
        + best["bbox_x2"].astype(str)
        + ","
        + best["bbox_y2"].astype(str)
    )
    return best[
        [
            "rank",
            "image_id",
            "image_path",
            "score",
            "matched_face_id",
            "bbox",
        ]
    ]


def rank_global_index(
    query_emb: np.ndarray,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    topk: int | None = 10,
    excluded_image_ids: set[str] | None = None,
) -> pd.DataFrame:
    scores = cosine_scores(embeddings, query_emb)
    rows = metadata.copy()
    rows["score"] = scores
    excluded = set(excluded_image_ids or set())
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


def rank_fused_index(
    query_face_emb: np.ndarray,
    query_global_emb: np.ndarray,
    face_embeddings: np.ndarray,
    face_metadata: pd.DataFrame,
    global_embeddings: np.ndarray,
    global_metadata: pd.DataFrame,
    topk: int | None,
    face_weight: float,
    global_weight: float,
    threshold: float = -1.0,
    face_excluded_image_ids: set[str] | None = None,
    global_excluded_image_ids: set[str] | None = None,
) -> pd.DataFrame:
    face_limit = int(face_metadata["image_id"].nunique()) if not face_metadata.empty else topk
    global_limit = len(global_metadata) if not global_metadata.empty else topk
    face_results = rank_face_index(
        query_face_emb,
        face_embeddings,
        face_metadata,
        topk=face_limit,
        threshold=threshold,
        excluded_image_ids=face_excluded_image_ids,
    )
    global_results = rank_global_index(
        query_global_emb,
        global_embeddings,
        global_metadata,
        topk=global_limit,
        excluded_image_ids=global_excluded_image_ids,
    )
    return combine_face_global_results(
        face_results,
        global_results,
        face_weight=face_weight,
        global_weight=global_weight,
        topk=topk,
    )
