from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from face_lib import build_face_app, load_face_index, query_embedding_from_image, search_face_index
from fusion_lib import search_fusion
from global_lib import (
    build_resnet50_feature_extractor,
    extract_global_embedding,
    load_global_index,
    search_global_index,
)
from project_paths import OUTPUTS_DIR


UPLOAD_DIR = OUTPUTS_DIR / "app_uploads"

INDEX_PRESETS = {
    "Gallagher (demo face + global)": (
        OUTPUTS_DIR / "face_index_gallagher",
        OUTPUTS_DIR / "global_index_gallagher",
    ),
    "Padrao do projeto": (
        OUTPUTS_DIR / "face_index",
        OUTPUTS_DIR / "global_index",
    ),
    "Personalizado": (
        OUTPUTS_DIR / "face_index",
        OUTPUTS_DIR / "global_index",
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
    if search_mode in {"Buscar pessoa", "Fusao face + global"}:
        st.warning(
            "Se a imagem de consulta tiver mais de um rosto, esta versao usa automaticamente "
            "o maior rosto detectado. A selecao manual do rosto e uma melhoria futura."
        )

    uploaded = st.file_uploader("Imagem de consulta", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("Envie uma imagem para iniciar a busca.")
        return

    query_path = save_upload(uploaded)
    st.image(str(query_path), caption="Consulta", width=320)

    if not st.button("Buscar"):
        return

    try:
        if search_mode == "Buscar pessoa":
            E, meta = cached_face_index(face_index_dir)
            app = cached_face_app(device, 640)
            q, _ = query_embedding_from_image(query_path, app)
            results = search_face_index(q, E, meta, topk=topk, threshold=threshold, query_path=query_path)
        elif search_mode == "Buscar imagem semelhante":
            E, meta = cached_global_index(global_index_dir)
            extractor, transform, torch_device = cached_resnet(device, "imagenet")
            q = extract_global_embedding(query_path, extractor, transform, torch_device)
            results = search_global_index(q, E, meta, topk=topk, query_path=query_path)
        else:
            face_E, face_meta = cached_face_index(face_index_dir)
            global_E, global_meta = cached_global_index(global_index_dir)
            app = cached_face_app(device, 640)
            extractor, transform, torch_device = cached_resnet(device, "imagenet")
            face_q, _ = query_embedding_from_image(query_path, app)
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
