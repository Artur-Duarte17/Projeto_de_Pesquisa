from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr) + eps
    return arr / norm


def l2_normalize_matrix(mat: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True) + eps
    return arr / norm


def cosine_scores(embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
    E = l2_normalize_matrix(embeddings)
    q = l2_normalize(query)
    return E @ q


def cosine_to_unit(score: float | np.ndarray) -> float | np.ndarray:
    return np.clip((score + 1.0) / 2.0, 0.0, 1.0)
