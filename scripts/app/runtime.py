"""Local UI state and bounded caches; scientific operations stay in retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from project_paths import OUTPUTS_DIR
from retrieval.adapters.face_model import build_face_app, detect_faces_in_image
from retrieval.adapters.global_model import build_resnet50_feature_extractor, extract_global_embedding
from retrieval.adapters.index_store import load_face_index, load_global_index
from retrieval.adapters.search_gateway import search_face_index, search_fusion, search_global_index
from retrieval.domain.face_policy import embedding_from_face

FACE_MODE = "Buscar pessoa"
GLOBAL_MODE = "Buscar imagem semelhante"
FUSION_MODE = "Fusão face + global"
PERSONAL_PROFILE = "Uso pessoal"
RESEARCH_PROFILE = "Pesquisa"


@dataclass(frozen=True)
class ReferenceFace:
    bbox: tuple[float, float, float, float]
    vector: np.ndarray | None


@dataclass(frozen=True)
class SearchSettings:
    profile: str
    source_name: str
    face_index: Path
    global_index: Path
    mode: str = FACE_MODE
    device: str = "cpu"
    threshold: float = 0.35
    topk: int | None = None
    face_weight: float = 0.7


@st.cache_resource(max_entries=2)
def cached_face_app(device: str, det_size: int):
    return build_face_app(device=device, det_size=det_size)


@st.cache_resource(max_entries=2)
def cached_resnet(device: str):
    return build_resnet50_feature_extractor(device=device, weights_name="imagenet")


@st.cache_data(max_entries=1)
def cached_face_index(index_dir: str, revision: tuple):
    return load_face_index(Path(index_dir))


@st.cache_data(max_entries=1)
def cached_global_index(index_dir: str, revision: tuple):
    return load_global_index(Path(index_dir))


def index_revision(index_dir: Path, kind: str) -> tuple:
    """Cheap cache invalidation, not a substitute for scientific manifest hashes."""
    revision = []
    for name in (f"{kind}_embeddings.npy", f"{kind}_metadata.csv"):
        try:
            info = (index_dir / name).stat()
            revision.append((info.st_mtime_ns, info.st_size))
        except FileNotFoundError:
            revision.append(None)
    return tuple(revision)


def save_upload(uploaded_file) -> Path:
    content = bytes(uploaded_file.getbuffer())
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}:
        raise ValueError("Formato de imagem não suportado")
    directory = OUTPUTS_DIR / "app_uploads"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{hashlib.sha256(content).hexdigest()}{suffix}"
    if not output.exists():
        output.write_bytes(content)
    return output


def annotated_faces_preview(image_bgr: np.ndarray, faces: list, max_side: int = 1200) -> np.ndarray:
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    preview = cv2.resize(image_bgr, (max(1, round(width * scale)), max(1, round(height * scale))))
    preview_height, preview_width = preview.shape[:2]
    for number, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = [int(round(float(value) * scale)) for value in face.bbox]
        x1, x2 = [max(0, min(preview_width - 1, value)) for value in (x1, x2)]
        y1, y2 = [max(0, min(preview_height - 1, value)) for value in (y1, y2)]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(preview, str(number), (x1, max(24, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


@st.cache_data(max_entries=2)
def cached_reference_faces(query_path: str, device: str) -> tuple[np.ndarray, list[ReferenceFace]]:
    image, faces = detect_faces_in_image(Path(query_path), cached_face_app(device, 640))
    preview = annotated_faces_preview(image, faces)
    references = [ReferenceFace(tuple(float(value) for value in face.bbox), embedding_from_face(face)) for face in faces]
    # Cache only the small annotated preview and vectors, never the full image.
    return preview, references


def search_context(settings: SearchSettings, query_path: Path, face_number: int | None) -> tuple:
    face_revision = index_revision(settings.face_index, "face") if settings.mode != GLOBAL_MODE else ()
    global_revision = index_revision(settings.global_index, "global") if settings.mode != FACE_MODE else ()
    return (settings, str(query_path), face_number, face_revision, global_revision)


def execute_search(settings: SearchSettings, query_path: Path, face: ReferenceFace | None) -> pd.DataFrame:
    if settings.mode in {FACE_MODE, FUSION_MODE}:
        if face is None or face.vector is None:
            raise RuntimeError("O rosto escolhido não pôde ser usado na busca. Envie outra foto.")
        face_embeddings, face_metadata = cached_face_index(str(settings.face_index), index_revision(settings.face_index, "face"))
    if settings.mode in {GLOBAL_MODE, FUSION_MODE}:
        global_embeddings, global_metadata = cached_global_index(str(settings.global_index), index_revision(settings.global_index, "global"))
        extractor, transform, device = cached_resnet(settings.device)
        global_query = extract_global_embedding(query_path, extractor, transform, device)
    if settings.mode == FACE_MODE:
        return search_face_index(face.vector, face_embeddings, face_metadata, topk=settings.topk, threshold=settings.threshold, query_path=query_path)
    if settings.mode == GLOBAL_MODE:
        return search_global_index(global_query, global_embeddings, global_metadata, topk=settings.topk, query_path=query_path)
    if settings.mode != FUSION_MODE:
        raise ValueError("Tipo de busca inválido")
    return search_fusion(query_path, face.vector, global_query, face_embeddings, face_metadata, global_embeddings, global_metadata,
                         topk=settings.topk, face_weight=settings.face_weight, global_weight=1.0 - settings.face_weight, threshold=settings.threshold)


def clear_album_caches() -> None:
    cached_face_index.clear()
    cached_global_index.clear()
    st.session_state.pop("last_search", None)


def open_album_action(action: str, album_key: str = "__new__") -> None:
    st.session_state["app_page"] = "Meus álbuns"
    st.session_state["album_action"] = action
    st.session_state["album_edit_selection"] = album_key
    if album_key != "__new__":
        st.session_state["album_remove_selection"] = album_key


def use_album(album_key: str) -> None:
    st.session_state["album_selection"] = album_key
    st.session_state["app_page"] = "Buscar fotos"
