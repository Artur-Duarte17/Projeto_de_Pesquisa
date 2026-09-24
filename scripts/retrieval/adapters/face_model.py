from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from retrieval.domain.face_policy import (
    bbox_string,
    embedding_from_face,
    pick_query_face,
    sorted_faces,
)


def ctx_id_from_device(device: str) -> int:
    return 0 if device.lower() == "cuda" else -1


def build_face_app(device: str = "cpu", det_size: int = 640):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id_from_device(device), det_size=(det_size, det_size))
    return app


def detect_faces_in_image(image_path: Path, app) -> tuple[np.ndarray, list]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read query image: {image_path}")
    return img, sorted_faces(app.get(img))


def query_embedding_from_image(
    image_path: Path,
    app,
    query_face_mode: str = "largest",
    query_face_index: int = 0,
    query_target_x: float | None = None,
    query_target_y: float | None = None,
) -> tuple[np.ndarray, str]:
    _, faces = detect_faces_in_image(image_path, app)
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
