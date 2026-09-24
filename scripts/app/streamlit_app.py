from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import OUTPUTS_DIR
from retrieval.adapters.face_model import build_face_app, detect_faces_in_image
from retrieval.adapters.global_model import build_resnet50_feature_extractor, extract_global_embedding
from retrieval.adapters.index_store import load_face_index, load_global_index
from retrieval.adapters.search_gateway import search_face_index, search_fusion, search_global_index
from retrieval.domain.face_policy import bbox_string, embedding_from_face


UPLOAD_DIR = OUTPUTS_DIR / "app_uploads"

INDEX_PRESETS = {
    "Gallagher final": (
        OUTPUTS_DIR / "final" / "gallagher" / "face_index",
        OUTPUTS_DIR / "final" / "gallagher" / "global_index",
    ),
    "Colecao preparada": (
        OUTPUTS_DIR / "collections" / "default" / "face",
        OUTPUTS_DIR / "collections" / "default" / "global",
    ),
    "Personalizado": (
        OUTPUTS_DIR / "collections" / "default" / "face",
        OUTPUTS_DIR / "collections" / "default" / "global",
    ),
}

MODE_DESCRIPTIONS = {
    "Buscar pessoa": (
        "Busca por identidade: o sistema detecta um rosto na imagem enviada e retorna "
        "fotos inteiras onde uma pessoa parecida aparece."
    ),
    "Buscar imagem semelhante": (
        "Busca global: o sistema compara a imagem inteira e retorna fotos visualmente "
        "parecidas, sem tentar reconhecer a identidade das pessoas."
    ),
    "Fusao face + global": (
        "Fusao: combina o score facial com o score global. Na versao atual, a fusao "
        "e uma demonstracao experimental, nao necessariamente melhor que a busca facial."
    ),
}


@st.cache_resource
def cached_face_app(device: str, det_size: int):
    return build_face_app(device=device, det_size=det_size)


@st.cache_resource
def cached_resnet(device: str, weights: str):
    return build_resnet50_feature_extractor(device=device, weights_name=weights)


@st.cache_data
def cached_face_index(index_dir: str):
    E, meta = load_face_index(Path(index_dir))
    return E, meta


@st.cache_data
def cached_global_index(index_dir: str):
    E, meta = load_global_index(Path(index_dir))
    return E, meta


def save_upload(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    out = UPLOAD_DIR / uploaded_file.name
    out.write_bytes(uploaded_file.getbuffer())
    return out


def show_gallery(results: pd.DataFrame) -> None:
    if results.empty:
        st.warning("Nenhum resultado encontrado.")
        return
    cols = st.columns(3)
    for i, row in enumerate(results.itertuples(index=False)):
        with cols[i % 3]:
            img_path = Path(str(row.image_path))
            if not img_path.is_absolute():
                img_path = ROOT / img_path
            st.image(str(img_path), caption=f"#{row.rank} score={float(row.score):.3f}", use_container_width=True)
            st.caption(Path(str(row.image_path)).name)


def simplified_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    out = pd.DataFrame()
    out["rank"] = results["rank"].astype(int)
    out["score"] = results["score"].astype(float).round(4)
    out["arquivo"] = results["image_path"].astype(str).map(lambda p: Path(p).name)
    out["caminho"] = results["image_path"].astype(str)
    if "face_score" in results.columns:
        out["score_face"] = results["face_score"].astype(float).round(4)
    if "global_score" in results.columns:
        out["score_global"] = results["global_score"].astype(float).round(4)
    return out


def annotated_faces_preview(image_bgr: np.ndarray, faces: list) -> np.ndarray:
    preview = image_bgr.copy()
    height, width = preview.shape[:2]
    for index, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 255), 3)
        cv2.putText(
            preview,
            str(index),
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(page_title="Recuperacao Fotografica", layout="wide")
    st.title("Recuperacao Fotografica - Face + CBIR Global")

    with st.sidebar:
        search_mode = st.selectbox("Modo", ["Buscar pessoa", "Buscar imagem semelhante", "Fusao face + global"])
        preset = st.selectbox("Acervo", list(INDEX_PRESETS.keys()))
        topk = st.slider("Top-K", 1, 50, 10)
        threshold = st.slider("Threshold facial", -1.0, 1.0, 0.35, 0.01)
        device = st.selectbox("Device", ["cpu", "cuda"])
        default_face_index, default_global_index = INDEX_PRESETS[preset]
        face_index_dir = st.text_input("Face index", str(default_face_index))
        global_index_dir = st.text_input("Global index", str(default_global_index))
        if search_mode == "Fusao face + global":
            face_weight = st.slider("Peso face", 0.0, 1.0, 0.7, 0.1)
        else:
            face_weight = 0.7
        global_weight = 1.0 - face_weight

    st.info(MODE_DESCRIPTIONS[search_mode])
    uploaded = st.file_uploader("Imagem de consulta", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("Envie uma imagem para iniciar a busca.")
        return

    query_path = save_upload(uploaded)
    st.image(str(query_path), caption="Consulta", width=320)

    selected_face = None
    selected_bbox = None
    if search_mode in {"Buscar pessoa", "Fusao face + global"}:
        try:
            face_app = cached_face_app(device, 640)
            query_image, detected_faces = detect_faces_in_image(query_path, face_app)
        except Exception as exc:
            st.error(f"Falha ao detectar rostos na consulta: {exc}")
            return
        if not detected_faces:
            st.error("Nenhum rosto foi detectado na imagem de consulta.")
            return
        st.image(
            annotated_faces_preview(query_image, detected_faces),
            caption="Rostos detectados",
            width=520,
        )
        selected_index = st.selectbox(
            "Pessoa que deve ser procurada",
            range(len(detected_faces)),
            format_func=lambda index: f"Rosto {index + 1}",
        )
        selected_face = detected_faces[int(selected_index)]
        selected_bbox = bbox_string(selected_face)
        st.caption(f"Caixa selecionada: {selected_bbox}")

    if not st.button("Buscar"):
        return

    try:
        if search_mode == "Buscar pessoa":
            E, meta = cached_face_index(face_index_dir)
            q = embedding_from_face(selected_face)
            if q is None:
                raise RuntimeError("O rosto selecionado não possui embedding.")
            results = search_face_index(q, E, meta, topk=topk, threshold=threshold, query_path=query_path)
        elif search_mode == "Buscar imagem semelhante":
            E, meta = cached_global_index(global_index_dir)
            extractor, transform, torch_device = cached_resnet(device, "imagenet")
            q = extract_global_embedding(query_path, extractor, transform, torch_device)
            results = search_global_index(q, E, meta, topk=topk, query_path=query_path)
        else:
            face_E, face_meta = cached_face_index(face_index_dir)
            global_E, global_meta = cached_global_index(global_index_dir)
            extractor, transform, torch_device = cached_resnet(device, "imagenet")
            face_q = embedding_from_face(selected_face)
            if face_q is None:
                raise RuntimeError("O rosto selecionado não possui embedding.")
            global_q = extract_global_embedding(query_path, extractor, transform, torch_device)
            results = search_fusion(
                query_path,
                face_q,
                global_q,
                face_E,
                face_meta,
                global_E,
                global_meta,
                topk=topk,
                face_weight=face_weight,
                global_weight=global_weight,
                threshold=threshold,
            )
    except Exception as exc:
        st.error(f"Falha na busca: {exc}")
        return

    st.subheader("Resultados")
    st.dataframe(simplified_results(results), use_container_width=True, hide_index=True)
    show_gallery(results)
    with st.expander("Tabela tecnica completa"):
        st.dataframe(results, use_container_width=True)


if __name__ == "__main__":
    main()
