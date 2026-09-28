"""Album management screens; filesystem operations belong to the adapter."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.runtime import clear_album_caches, open_album_action, use_album
from retrieval.adapters import album_store
from retrieval.adapters.album_store import Album, AlbumPreparationError

ALBUM_ACTIONS = ["Álbuns cadastrados", "Preparar / atualizar", "Remover"]


def show_album_list(albums: list[Album]) -> None:
    if not albums:
        st.info("Seu primeiro álbum começa com uma pasta de fotos neste computador.")
        st.button("Preparar meu primeiro álbum", type="primary", on_click=open_album_action, args=("Preparar / atualizar",))
        return
    for album in albums:
        available = album.photo_dir is None or album.photo_dir.is_dir()
        with st.container(border=True):
            description, actions = st.columns([3, 2])
            with description:
                st.subheader(album.name)
                if not album.ready:
                    st.caption("Preparação incompleta · tente preparar novamente")
                elif not available:
                    st.caption("Pasta de fotos indisponível · atualize o endereço")
                else:
                    summary = album_store.album_summary(album)
                    count = summary.get("images_scanned")
                    st.caption(f"Pronto para buscar · {count} fotos analisadas" if count is not None else "Pronto para buscar")
                if album.photo_dir:
                    st.caption(f"Pasta original: {album.photo_dir}")
            key = str(album.output_dir)
            with actions:
                st.button("Buscar neste álbum", key=f"use_{key}", disabled=not album.ready or not available,
                          on_click=use_album, args=(key,), use_container_width=True)
                update, remove = st.columns(2)
                update.button("Atualizar", key=f"update_{key}", on_click=open_album_action, args=("Preparar / atualizar", key), use_container_width=True)
                remove.button("Remover", key=f"remove_{key}", on_click=open_album_action, args=("Remover", key), use_container_width=True)


def show_album_preparation(albums: list[Album], device: str) -> None:
    st.subheader("Preparar ou atualizar um álbum")
    st.caption("Faça isso na primeira vez ou quando suas fotos mudarem. Não é necessário repetir a preparação a cada busca.")
    by_key = {str(album.output_dir): album for album in albums}
    options = ["__new__", *by_key]
    if st.session_state.get("album_edit_selection") not in options:
        st.session_state["album_edit_selection"] = "__new__"
    key = st.selectbox("Qual álbum preparar?", options, key="album_edit_selection",
                       format_func=lambda item: "Adicionar novo álbum" if item == "__new__" else (
                           f"Atualizar {by_key[item].name}" if by_key[item].ready else f"Tentar novamente: {by_key[item].name}"))
    selected = by_key.get(key)
    with st.container(border=True), st.form(f"album_preparation_{key}"):
        name = st.text_input("Nome do álbum", value=selected.name if selected else "", key=f"album_name_{key}", placeholder="Ex.: Família ou Viagem")
        folder = st.text_input("Pasta com as fotos neste computador", value=str(selected.photo_dir) if selected and selected.photo_dir else "",
                               key=f"album_folder_{key}", placeholder=r"C:\Users\SeuNome\Pictures\Meu álbum",
                               help="Copie o endereço da pasta no Explorador de Arquivos e cole aqui. As subpastas também serão lidas.")
        st.caption("As fotos do álbum ficam na pasta original. Esta etapa não as copia, altera, apaga ou envia a um serviço externo.")
        st.caption("Fotos compatíveis: JPG/JPEG, PNG, BMP, WebP e HEIC/HEIF. Vídeos não são incluídos.")
        if selected:
            st.info("A atualização substitui os índices no mesmo destino, sem criar cópias v2.")
        submitted = st.form_submit_button("Atualizar álbum" if selected else "Preparar álbum", type="primary", use_container_width=True)
    if not submitted:
        return
    log = ""
    failed = False
    with st.status("Preparando seu álbum…", expanded=True) as status:
        st.write("Analisando as fotos e os rostos. Isso pode levar alguns minutos; não feche a aplicação.")
        try:
            if not folder.strip():
                raise ValueError("Informe a pasta das fotos")
            output = selected.output_dir if selected else album_store.album_output_dir(name)
            log = album_store.prepare_album(Path(folder.strip().strip('"')), output, name, device=device, overwrite=selected is not None)
        except (ValueError, OSError, AlbumPreparationError) as exc:
            status.update(label="Não foi possível concluir a preparação", state="error")
            st.error(str(exc))
            log = getattr(exc, "log", log)
            failed = True
        else:
            status.update(label="Álbum preparado", state="complete", expanded=False)
    if failed:
        if log:
            with st.expander("Detalhes para diagnóstico"):
                st.code(log[-16000:])
        return
    clear_album_caches()
    st.session_state["pending_album"] = str(output)
    st.session_state["app_notice"] = f"Álbum {name.strip()} preparado. Agora escolha uma foto de referência."
    st.rerun()


def show_album_removal(albums: list[Album]) -> None:
    st.subheader("Remover um álbum da aplicação")
    if not albums:
        st.info("Não há álbuns cadastrados para remover.")
        return
    by_key = {str(album.output_dir): album for album in albums}
    if st.session_state.get("album_remove_selection") not in by_key:
        st.session_state["album_remove_selection"] = next(iter(by_key))
    key = st.selectbox("Álbum que deseja remover", list(by_key), key="album_remove_selection", format_func=lambda item: by_key[item].name)
    album = by_key[key]
    with st.container(border=True):
        st.warning(f"Remover {album.name} apaga somente os índices e registros gerados pela aplicação. As fotos originais ficam intactas.")
        st.caption("Para voltar a usar esse álbum, será necessário prepará-lo novamente.")
        with st.form(f"album_removal_{key}"):
            confirmed = st.checkbox(f"Confirmo que quero remover o álbum {album.name} da aplicação")
            submitted = st.form_submit_button("Remover álbum")
    if not submitted:
        return
    try:
        album_store.remove_album(album.output_dir, confirmed=confirmed)
    except (OSError, ValueError) as exc:
        st.error(f"Não foi possível remover o álbum: {exc}")
        return
    clear_album_caches()
    st.session_state["app_notice"] = f"Álbum {album.name} removido. As fotos originais foram preservadas."
    st.session_state["album_removed"] = True
    st.rerun()


def show_albums_page(albums: list[Album], device: str) -> None:
    title, add = st.columns([4, 1])
    title.title("Meus álbuns")
    add.button("Adicionar álbum", on_click=open_album_action, args=("Preparar / atualizar",), use_container_width=True)
    st.caption("Organize os álbuns da aplicação. Suas fotografias continuam nas pastas originais.")
    if st.session_state.get("album_action") not in ALBUM_ACTIONS:
        st.session_state["album_action"] = "Álbuns cadastrados" if albums else "Preparar / atualizar"
    action = st.radio("Gerenciar álbuns", ALBUM_ACTIONS, key="album_action", horizontal=True)
    if action == "Álbuns cadastrados":
        show_album_list(albums)
    elif action == "Preparar / atualizar":
        show_album_preparation(albums, device)
    else:
        show_album_removal(albums)
