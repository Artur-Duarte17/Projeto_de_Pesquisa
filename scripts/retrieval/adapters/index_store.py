from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_face_index(index_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    emb_path = index_dir / "face_embeddings.npy"
    meta_path = index_dir / "face_metadata.csv"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing face index files in {index_dir}")
    E = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    if len(E) != len(meta):
        raise ValueError(f"Face index mismatch: {len(E)} embeddings vs {len(meta)} metadata rows")
    return E, meta


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
