"""Presentation-only components shared by the personal and research screens."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from retrieval.adapters.image_preview import load_gallery_preview

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PER_PAGE = 24


def apply_style() -> None:
    st.markdown("""
        <style>
        .block-container { max-width: 1360px; padding-top: 2rem; padding-bottom: 3rem; }
        h1 { letter-spacing: -0.035em; }
        h2, h3 { letter-spacing: -0.015em; }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.16); }
        [data-testid="stImage"] img { border-radius: 10px; }
        .st-key-results_gallery [data-testid="stImage"] img {
            width: 100%; height: 230px; object-fit: contain;
        }
        [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
        </style>
        """, unsafe_allow_html=True)


def simplified_results(results: pd.DataFrame) -> pd.DataFrame:
    columns = {"Posição": results["rank"].astype(int),
               "Arquivo": results["image_path"].astype(str).map(lambda value: Path(value).name),
               "Similaridade": results["score"].astype(float).round(4)}
    if "face_score" in results:
        columns["Facial"] = results["face_score"].round(4)
    if "global_score" in results:
        columns["Global"] = results["global_score"].round(4)
    return pd.DataFrame(columns)


def show_gallery(results: pd.DataFrame, show_scores: bool = False) -> None:
    columns = st.columns(3)
    for position, row in enumerate(results.itertuples(index=False)):
        with columns[position % 3], st.container(border=True):
            path = Path(str(row.image_path))
            if not path.is_absolute():
                path = ROOT / path
            try:
                st.image(load_gallery_preview(path), use_container_width=True)
            except (OSError, ValueError, cv2.error, RuntimeError):
                st.warning("Prévia indisponível. Confira se a foto ainda está na pasta original.")
            st.caption(f"#{row.rank} · {path.name}")
            if show_scores:
                st.caption(f"Similaridade: {float(row.score):.4f}")


@st.fragment
def show_search_results(results: pd.DataFrame, show_technical: bool = False, elapsed: float | None = None) -> None:
    st.divider()
    heading, count = st.columns([4, 1])
    heading.subheader("Fotos encontradas")
    count.metric("Resultados", len(results))
    st.caption("Confira as fotos: semelhança facial não comprova a identidade da pessoa.")
    if results.empty:
        st.info("Nenhuma foto passou pelos critérios desta busca. Tente outra referência ou ajuste o rigor da busca.")
        return
    pages = (len(results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
    if pages > 1:
        current = st.session_state.get("results_page", 1)
        if not 1 <= current <= pages:
            st.session_state["results_page"] = 1
        page = int(st.number_input("Página dos resultados", min_value=1, max_value=pages, step=1, key="results_page"))
    else:
        page = 1
    first = (page - 1) * RESULTS_PER_PAGE
    visible = results.iloc[first:first + RESULTS_PER_PAGE]
    st.caption(f"Mostrando {first + 1}–{first + len(visible)} de {len(results)} resultados. Página {page} de {pages}.")
    with st.container(key="results_gallery"):
        show_gallery(visible, show_scores=show_technical)
    if show_technical:
        with st.expander("Dados da busca e tabela técnica"):
            if elapsed is not None:
                st.caption(f"Tempo da chamada de busca: {elapsed:.3f} s. Não inclui análise facial prévia nem exibição das fotos.")
            st.dataframe(simplified_results(visible), use_container_width=True, hide_index=True)
            st.dataframe(visible, use_container_width=True, hide_index=True)
