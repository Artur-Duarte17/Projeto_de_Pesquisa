from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from retrieval_common import cosine_scores, image_ids_for_query_path, l2_normalize, rel_to_root


def ctx_id_from_device(device: str) -> int:
    return 0 if device.lower() == "cuda" else -1


def build_face_app(device: str = "cpu", det_size: int = 640):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id_from_device(device), det_size=(det_size, det_size))
    return app


def face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def sorted_faces(faces) -> list:
    return sorted(faces, key=lambda f: (float(f.bbox[1]), float(f.bbox[0])))


def _face_bbox_values(face) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in face.bbox)


def _face_contains_point(face, target_x: float, target_y: float) -> bool:
    x1, y1, x2, y2 = _face_bbox_values(face)
    return x1 <= target_x <= x2 and y1 <= target_y <= y2


def _annotated_point_sort_key(face, target_x: float, target_y: float) -> tuple[float, ...]:
    x1, y1, x2, y2 = _face_bbox_values(face)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    center_distance_squared = (center_x - target_x) ** 2 + (center_y - target_y) ** 2
    return (center_distance_squared, face_area(face), y1, x1, y2, x2)


def pick_query_face(
    faces,
    mode: str = "largest",
    index: int = 0,
    target_x: float | None = None,
    target_y: float | None = None,
):
    if not faces:
        return None
    if mode == "index":
        ordered = sorted_faces(faces)
        if index < 0 or index >= len(ordered):
            return None
        return ordered[index]
    if mode == "annotated_eye_midpoint":
        if target_x is None or target_y is None:
            raise ValueError("Annotated-eye selection requires target_x and target_y")
        containing = [
            face for face in faces if _face_contains_point(face, float(target_x), float(target_y))
        ]
        if not containing:
            return None
        return min(
            containing,
            key=lambda face: _annotated_point_sort_key(
                face,
                float(target_x),
                float(target_y),
            ),
        )
    if mode != "largest":
        raise ValueError(f"Unsupported query-face mode: {mode}")
    return sorted(faces, key=face_area, reverse=True)[0]


def embedding_from_face(face) -> np.ndarray | None:
    emb = None
    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
        emb = face.normed_embedding
    elif hasattr(face, "embedding") and face.embedding is not None:
        emb = face.embedding
    if emb is None:
        return None
    return l2_normalize(np.asarray(emb, dtype=np.float32).flatten())


def bbox_string(face) -> str:
    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    return f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}"


def query_embedding_from_image(
    image_path: Path,
    app,
    query_face_mode: str = "largest",
    query_face_index: int = 0,
    query_target_x: float | None = None,
    query_target_y: float | None = None,
) -> tuple[np.ndarray, str]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read query image: {image_path}")
    faces = app.get(img)
    face = pick_query_face(
        faces,
        mode=query_face_mode,
        index=query_face_index,
        target_x=query_target_x,
        target_y=query_target_y,
    )
    if face is None:
        if query_face_mode == "annotated_eye_midpoint":
            raise RuntimeError(
                f"No detected face contains the annotated-eye midpoint in query image: {image_path}"
            )
        raise RuntimeError(f"No face found in query image: {image_path}")
    emb = embedding_from_face(face)
    if emb is None:
        raise RuntimeError(f"No embedding found for query face: {image_path}")
    return emb, bbox_string(face)


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


def search_face_index(
    query_emb: np.ndarray,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    topk: int | None = 10,
    threshold: float = -1.0,
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


def query_id_from_path(path: Path) -> str:
    return Path(rel_to_root(path)).stem
