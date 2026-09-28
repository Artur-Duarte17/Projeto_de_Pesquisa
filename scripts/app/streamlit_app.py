from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.album_views import show_albums_page
from app.components import apply_style, show_search_results
from app.runtime import (
    FACE_MODE, FUSION_MODE, GLOBAL_MODE, PERSONAL_PROFILE, RESEARCH_PROFILE,
    SearchSettings, cached_reference_faces, execute_search, open_album_action,
    save_upload, search_context,
)
from project_paths import OUTPUTS_DIR
from retrieval.adapters.album_store import Album, album_summary, discover_albums
from retrieval.adapters.image_preview import load_gallery_preview
from retrieval.domain.face_policy import bbox_string

PAGES = ["Buscar fotos", "Meus álbuns"]
RESEARCH_SOURCES = ["Meu álbum", "Gallagher final", "Índices personalizados"]


def manual_index_path(value: str) -> Path:
    path = Path(value.strip().strip('"'))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def initialize_session(albums: list[Album]) -> None:
    ready = {str(album.output_dir): album for album in albums if album.ready}
    pending = st.session_state.pop("pending_album", None)
    if pending in ready:
        st.session_state["album_selection"] = pending
        st.session_state["app_page"] = "Buscar fotos"
    if st.session_state.pop("album_removed", False):
        st.session_state["app_page"] = "Meus álbuns"
        st.session_state["album_action"] = "Álbuns cadastrados" if albums else "Preparar / atualizar"
    if st.session_state.get("album_selection") not in ready:
        st.session_state.pop("album_selection", None)
    if st.session_state.get("app_page") not in PAGES:
        st.session_state["app_page"] = "Buscar fotos"
    if st.session_state.get("ui_profile") not in {PERSONAL_PROFILE, RESEARCH_PROFILE}:
        st.session_state["ui_profile"] = PERSONAL_PROFILE


def show_sidebar(albums: list[Album]) -> tuple[str, SearchSettings | None, Album | None, bool, str]:
    ready = {str(album.output_dir): album for album in albums if album.ready}
    with st.sidebar:
        st.markdown("## CIBIR")
        st.caption("Seus álbuns, suas fotografias.")
        page = st.radio("Navegação", PAGES, key="app_page", label_visibility="collapsed")
        st.divider()
        profile = st.radio("Perfil de uso", [PERSONAL_PROFILE, RESEARCH_PROFILE], key="ui_profile",
                           help="Uso pessoal trabalha com seus álbuns. Pesquisa libera acervos e parâmetros experimentais.")
        source = "Meu álbum"
        if profile == RESEARCH_PROFILE and page == "Buscar fotos":
            source = st.selectbox("Fonte das fotos", RESEARCH_SOURCES, key="research_source",
                                  help="Escolha de onde vêm as fotos. Em Meu álbum, selecione abaixo qual álbum usar.")
        album = None
        settings = None
        if page == "Buscar fotos" and source == "Meu álbum":
            if ready:
                key = st.selectbox("Álbum", list(ready), key="album_selection", format_func=lambda item: ready[item].name)
                album = ready[key]
                settings = SearchSettings(profile, album.name, album.face_index, album.global_index)
                summary = album_summary(album)
                if summary.get("images_scanned") is not None:
                    st.caption(f"{summary['images_scanned']} fotos analisadas neste álbum")
            else:
                st.caption("Nenhum álbum pronto. Cadastre sua pasta em Meus álbuns.")
        elif page == "Buscar fotos" and profile == RESEARCH_PROFILE:
            if source == "Gallagher final":
                settings = SearchSettings(profile, source, OUTPUTS_DIR / "final/gallagher/face_index", OUTPUTS_DIR / "final/gallagher/global_index")
            else:
                face = st.text_input("Diretório do índice facial", key="research_face_path", placeholder="Caminho do índice face")
                global_dir = st.text_input("Diretório do índice global", key="research_global_path", placeholder="Caminho do índice global")
                settings = SearchSettings(profile, source, manual_index_path(face) if face.strip() else ROOT,
                                          manual_index_path(global_dir) if global_dir.strip() else ROOT) if face.strip() or global_dir.strip() else None
        with st.expander("Preferências de processamento"):
            device = st.selectbox("Processamento", ["cpu", "cuda"], key="ui_device",
                                  format_func=lambda item: "CPU · padrão" if item == "cpu" else "GPU · requer CUDA configurada")
            st.caption("Esta preferência vale para a preparação dos álbuns e para as buscas.")
        show_technical = False
        if page == "Buscar fotos":
            modes = [FACE_MODE, GLOBAL_MODE] + ([FUSION_MODE] if profile == RESEARCH_PROFILE else [])
            mode = st.selectbox("Tipo de busca", modes, key=f"search_mode_{profile}",
                                format_func=lambda item: {FACE_MODE: "Encontrar uma pessoa", GLOBAL_MODE: "Encontrar fotos parecidas", FUSION_MODE: "Fusão facial + global · experimental"}[item])
            topk = None
            threshold = 0.35
            face_weight = 0.7
            with st.expander("Parâmetros de pesquisa" if profile == RESEARCH_PROFILE else "Ajustes da busca"):
                all_results = st.checkbox("Mostrar todos os resultados", value=True, key=f"all_results_{profile}")
                if not all_results:
                    topk = int(st.number_input("Máximo de resultados", min_value=1, value=10, step=1, key=f"result_limit_{profile}"))
                if mode != GLOBAL_MODE:
                    threshold = st.slider("Rigor da busca facial", -1.0, 1.0, -1.0 if profile == RESEARCH_PROFILE else 0.35, 0.01, key=f"threshold_{profile}")
                    st.caption("Um valor maior exige mais semelhança, mas pode omitir fotos corretas. Não é uma porcentagem de certeza.")
                if mode == FUSION_MODE:
                    face_weight = st.slider("Peso facial na fusão", 0.0, 1.0, 0.7, 0.05, key="research_face_weight")
                    st.caption(f"Peso global: {1.0 - face_weight:.2f}. A fusão não tem superioridade garantida.")
                show_technical = st.checkbox("Mostrar dados técnicos", value=profile == RESEARCH_PROFILE, key=f"show_technical_{profile}")
            if settings is not None:
                settings = SearchSettings(profile, settings.source_name, settings.face_index, settings.global_index,
                                          mode, device, threshold, topk, face_weight)
        st.divider()
        st.caption("Fotos e buscas ficam neste computador. Mantenha as pastas originais disponíveis.")
    return page, settings, album, show_technical, device


def show_empty_search() -> None:
    with st.container(border=True):
        st.subheader("Vamos preparar seu primeiro álbum?")
        st.write("Escolha uma pasta com suas fotografias. A aplicação organiza os índices necessários e depois você pode buscar quantas vezes quiser.")
        steps = st.columns(3)
        steps[0].markdown("**1. Prepare um álbum**\n\nInforme um nome e a pasta das fotos.")
        steps[1].markdown("**2. Envie uma referência**\n\nEscolha uma foto da pessoa que procura.")
        steps[2].markdown("**3. Encontre suas fotos**\n\nConfira os resultados, organizados em páginas.")
        st.button("Preparar meu primeiro álbum", type="primary", on_click=open_album_action, args=("Preparar / atualizar",))


def show_search_page(settings: SearchSettings | None, album: Album | None, show_technical: bool) -> None:
    st.title("Buscar fotos")
    st.caption("Encontre pessoas ou imagens parecidas em um álbum preparado neste computador.")
    if settings is None:
        if st.session_state["ui_profile"] == PERSONAL_PROFILE:
            show_empty_search()
        else:
            st.info("Escolha um álbum pronto ou informe os índices de pesquisa. Não há um acervo científico selecionado automaticamente.")
        return
    with st.container(border=True):
        information, manage = st.columns([4, 1])
        information.subheader(settings.source_name)
        information.caption("O álbum inteiro participa da busca. Os resultados são exibidos em páginas de 24 fotos.")
        if album is not None:
            manage.button("Gerenciar álbum", on_click=open_album_action, args=("Álbuns cadastrados", str(album.output_dir)), use_container_width=True)
    if settings.profile == RESEARCH_PROFILE:
        st.info("Modo pesquisa · exploração interativa. Esta tela não executa nem substitui os protocolos científicos de avaliação.")
        if settings.mode == FUSION_MODE:
            st.caption("A fusão combina identidade facial e conteúdo da imagem inteira; não se presume que melhore a recuperação.")
        with st.expander("Acervo e configuração usados nesta busca"):
            st.json({"acervo": settings.source_name, "modo": settings.mode, "processamento": settings.device,
                     "limite_resultados": settings.topk, "threshold_facial": settings.threshold if settings.mode != GLOBAL_MODE else None,
                     "peso_facial": settings.face_weight if settings.mode == FUSION_MODE else None,
                     "indice_facial": str(settings.face_index) if settings.mode != GLOBAL_MODE else None,
                     "indice_global": str(settings.global_index) if settings.mode != FACE_MODE else None,
                     "modelo_facial": "InsightFace · buffalo_l pré-treinado" if settings.mode != GLOBAL_MODE else None,
                     "modelo_global": "ResNet50 · ImageNet" if settings.mode != FACE_MODE else None})
    if album is not None:
        if album.photo_dir is not None and not album.photo_dir.is_dir():
            st.warning("A pasta original não foi encontrada. Atualize o álbum para informar onde estão as fotos.")
            st.button("Atualizar endereço do álbum", on_click=open_album_action, args=("Preparar / atualizar", str(album.output_dir)))
            return
        summary = album_summary(album)
        if int(summary.get("global_failures") or 0) + int(summary.get("face_read_failures") or 0):
            st.warning("Algumas fotos não puderam ser lidas na preparação. Elas podem estar ausentes dos resultados.")
    reference, selection = st.columns([1, 1.25])
    with reference, st.container(border=True):
        st.subheader("1. Foto de referência")
        uploaded = st.file_uploader("Escolha uma foto", type=["jpg", "jpeg", "png", "bmp", "webp", "heic", "heif"], key="reference_upload",
                                    help="Fotos do iPhone em HEIC/HEIF também são aceitas.")
        st.caption("A foto enviada terá uma cópia local de trabalho. Ela não será publicada ou enviada a um serviço externo.")
        query_path = None
        if uploaded is not None:
            try:
                query_path = save_upload(uploaded)
                st.image(load_gallery_preview(query_path, max_side=720), caption=uploaded.name, use_container_width=True)
            except (OSError, ValueError, cv2.error, RuntimeError) as exc:
                st.error(f"Não foi possível abrir esta foto: {exc}")
                return
    face = None
    face_number = None
    with selection, st.container(border=True):
        st.subheader("2. Escolha o que procurar")
        if query_path is None:
            st.info("Envie uma foto ao lado para continuar.")
        elif settings.mode == GLOBAL_MODE:
            st.write("A busca compara a fotografia inteira: cores, formas e conteúdo visual.")
            st.caption("Este modo procura imagens parecidas, não a identidade de uma pessoa.")
        else:
            try:
                with st.spinner("Identificando os rostos da referência…"):
                    preview, faces = cached_reference_faces(str(query_path), settings.device)
            except Exception as exc:
                st.error(f"Não foi possível analisar os rostos: {exc}")
                return
            if not faces:
                st.warning("Não encontramos um rosto nesta foto. Tente uma referência mais nítida.")
                return
            st.image(preview, caption="Os números identificam os rostos detectados.", use_container_width=True)
            face_number = st.selectbox("Pessoa que deve ser procurada", range(len(faces)),
                                        format_func=lambda number: f"Pessoa {number + 1}",
                                        key=f"reference_face_{query_path.stem}_{settings.device}", disabled=len(faces) == 1)
            face = faces[int(face_number)]
            if face.vector is None:
                st.warning("Esse rosto não pôde ser usado na busca. Escolha outro ou envie outra foto.")
                return
            if show_technical:
                st.caption(f"Caixa facial selecionada: {bbox_string(face)}")
        clicked = st.button("Buscar fotos no álbum", type="primary", disabled=query_path is None, use_container_width=True)
    if query_path is None:
        return
    context = search_context(settings, query_path, face_number)
    if clicked:
        try:
            with st.spinner(f"Buscando em {settings.source_name}…"):
                started = time.perf_counter()
                results = execute_search(settings, query_path, face)
                elapsed = time.perf_counter() - started
        except Exception as exc:
            st.error(f"Não foi possível concluir a busca: {exc}")
            return
        st.session_state["last_search"] = {"context": context, "results": results, "elapsed": elapsed}
        st.session_state["results_page"] = 1
    previous = st.session_state.get("last_search")
    if previous is not None and previous["context"] == context:
        show_search_results(previous["results"], show_technical, previous["elapsed"])
    elif previous is not None:
        st.caption("A foto, o álbum ou os parâmetros mudaram. Clique em Buscar fotos no álbum para atualizar os resultados.")


def main() -> None:
    st.set_page_config(page_title="CIBIR · Álbuns e pesquisa", page_icon="📷", layout="wide", initial_sidebar_state="expanded")
    apply_style()
    albums = discover_albums()
    initialize_session(albums)
    page, settings, album, show_technical, device = show_sidebar(albums)
    notice = st.session_state.pop("app_notice", None)
    if notice:
        st.success(notice)
    if page == "Meus álbuns":
        show_albums_page(albums, device)
    else:
        show_search_page(settings, album, show_technical)


if __name__ == "__main__":
    main()
