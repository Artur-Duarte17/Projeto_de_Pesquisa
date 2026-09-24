from __future__ import annotations

import numpy as np

from retrieval.domain.similarity import l2_normalize


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
