"""Connect query-path exclusions to the in-memory retrieval use cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retrieval.adapters.files import image_ids_for_query_path
from retrieval.application.search import rank_face_index, rank_fused_index, rank_global_index


def search_face_index(
    query_emb: np.ndarray,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    topk: int | None = 10,
    threshold: float = -1.0,
    query_path: Path | None = None,
    exclude_image_ids: set[str] | None = None,
) -> pd.DataFrame:
    excluded = set(exclude_image_ids or set())
    if query_path is not None:
        excluded.update(image_ids_for_query_path(query_path, metadata))
    return rank_face_index(
        query_emb,
        embeddings,
        metadata,
        topk=topk,
        threshold=threshold,
        excluded_image_ids=excluded,
    )


def search_global_index(
    query_emb: np.ndarray,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    topk: int | None = 10,
    query_path: Path | None = None,
    exclude_image_ids: set[str] | None = None,
) -> pd.DataFrame:
    excluded = set(exclude_image_ids or set())
    if query_path is not None:
        excluded.update(image_ids_for_query_path(query_path, metadata))
    return rank_global_index(
        query_emb,
        embeddings,
        metadata,
        topk=topk,
        excluded_image_ids=excluded,
    )


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
    explicit = set(exclude_image_ids or set())
    face_excluded = explicit | image_ids_for_query_path(query_path, face_metadata)
    global_excluded = explicit | image_ids_for_query_path(query_path, global_metadata)
    return rank_fused_index(
        query_face_emb,
        query_global_emb,
        face_embeddings,
        face_metadata,
        global_embeddings,
        global_metadata,
        topk=topk,
        face_weight=face_weight,
        global_weight=global_weight,
        threshold=threshold,
        face_excluded_image_ids=face_excluded,
        global_excluded_image_ids=global_excluded,
    )
