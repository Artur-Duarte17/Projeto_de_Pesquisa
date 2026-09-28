"""Explicit local acceptance run with real models; not part of unit-test discovery.

Streamlit AppTest drives the actual screens. Only the file-uploader widget is
supplied programmatically, because AppTest does not implement browser uploads.
Photo decoding, face detection, searches and preparation subprocesses are real.
Private reports stay in ignored outputs/, and only a new validation album may
be updated or removed. This is not an identity-accuracy benchmark.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import JpegImagePlugin
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.runtime import FUSION_MODE, GLOBAL_MODE, cached_face_app
from retrieval.adapters.album_store import album_output_dir, discover_albums
from retrieval.adapters.files import list_images, sha256_file
from retrieval.adapters.run_paths import require_empty_target, resolve_run_root


def snapshot(folder: Path) -> dict[str, str]:
    return {str(path.relative_to(folder)): sha256_file(path)
            for path in sorted(folder.rglob("*")) if path.is_file()}


def button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def require_healthy(app: AppTest) -> None:
    if len(app.exception):
        raise AssertionError("Streamlit exception: " + str(app.exception[0].message))
    if len(app.error):
        raise AssertionError("Application error: " + str(app.error[0].value))


class ReferenceUpload(io.BytesIO):
    def __init__(self, path: Path):
        super().__init__(path.read_bytes())
        self.name = path.name


def main() -> int:
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--album-name", required=True,
                        help="New, dedicated validation album; existing albums are never reused")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    photos = args.photo_dir.resolve()
    if not photos.is_dir():
        raise FileNotFoundError(photos)
    run = resolve_run_root(args.run_root, project_root=ROOT, outputs_root=ROOT / "outputs",
                           baseline_root=ROOT / "outputs/final")
    require_empty_target(run)
    album_dir = album_output_dir(args.album_name)
    if album_dir.exists():
        raise FileExistsError("Acceptance requires a NEW album, never an existing user album")
    if album_dir == photos or album_dir in photos.parents or photos in album_dir.parents:
        raise ValueError("Validation indexes and original photographs must be disjoint")
    run.mkdir(parents=True, exist_ok=True)
    report: dict = {"status": "running", "kind": "functional_not_identity_accuracy",
                   "file_uploader": "programmatic_input_no_browser_upload", "checks": []}
    report_path = run / "acceptance_report.json"

    def checkpoint(label: str, **details) -> None:
        report["checks"].append({"check": label, **details})
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[OK] " + label + " " + json.dumps(details, ensure_ascii=False), flush=True)

    originals = snapshot(photos)
    existing = {str(album.output_dir): snapshot(album.output_dir) for album in discover_albums()}
    checkpoint("originals_and_existing_albums_frozen", files=len(originals), albums=len(existing))
    started = time.perf_counter()
    try:
        app = AppTest.from_file(str(ROOT / "scripts/app/streamlit_app.py"), default_timeout=900).run()
        require_healthy(app)
        assert app.radio(key="ui_profile").value == "Uso pessoal"
        assert not any(item.label == "Fonte das fotos" for item in app.selectbox)
        checkpoint("personal_default")
        app.radio(key="app_page").set_value("Meus álbuns").run()
        button(app, "Adicionar álbum").click().run()
        app.selectbox(key="ui_device").set_value(args.device).run()
        next(item for item in app.text_input if item.label == "Nome do álbum").set_value(args.album_name)
        next(item for item in app.text_input if item.label == "Pasta com as fotos neste computador").set_value(str(photos))
        print("[RUN] preparing real validation album", flush=True)
        button(app, "Preparar álbum").click().run(timeout=900)
        require_healthy(app)
        assert app.selectbox(key="album_selection").value == str(album_dir)
        global_manifest = json.loads((album_dir / "global/global_index_manifest.json").read_text())
        face_manifest = json.loads((album_dir / "face/face_index_manifest.json").read_text())
        global_meta = pd.read_csv(album_dir / "global/global_metadata.csv")
        face_meta = pd.read_csv(album_dir / "face/face_metadata.csv")
        scanned = len(list_images(photos))
        assert global_manifest["extra"]["images_scanned"] == scanned
        assert global_manifest["extra"]["failures"] == 0
        assert face_manifest["extra"].get("read_failures", 0) == 0
        assert len(global_meta) == scanned
        heic_count = sum(Path(value).suffix.lower() in {".heic", ".heif"}
                         for value in global_meta.image_path)
        checkpoint("real_preparation", images=scanned, heic=heic_count,
                   faces=len(face_meta), read_failures=0,
                   reduced_jpegs=global_manifest["extra"].get("reduced_image_count", 0))

        # AppTest retains obsolete form nodes after st.rerun(); a fresh session
        # avoids testing that simulator artifact instead of the application's UI.
        app = AppTest.from_file(str(ROOT / "scripts/app/streamlit_app.py"), default_timeout=900)
        app.session_state["album_selection"] = str(album_dir)
        app.session_state["ui_device"] = args.device
        app.run()

        # Prefer a HEIC photo with multiple faces to cover both boundaries.
        counts = face_meta.groupby("image_path").size().sort_values(ascending=False)
        multiple = [Path(path) for path, count in counts.items() if count > 1]
        if not multiple:
            raise AssertionError("This acceptance run requires a multi-face reference")
        reference = next((path for path in multiple if path.suffix.lower() in {".heic", ".heif"}), multiple[0])
        reference = reference if reference.is_absolute() else ROOT / reference
        upload = ReferenceUpload(reference)
        with mock.patch("streamlit.file_uploader", return_value=upload):
            app.run(timeout=900)
            require_healthy(app)
            face_control = next(item for item in app.selectbox if item.label == "Pessoa que deve ser procurada")
            assert len(face_control.options) >= 2
            checkpoint("multi_face_reference", format=reference.suffix.lower(), faces=len(face_control.options))
            button(app, "Buscar fotos no álbum").click().run(timeout=900)
            require_healthy(app)
            first_results = app.session_state["last_search"]["results"].copy()
            assert not first_results.empty
            assert first_results.image_path.is_unique
            selected = app.session_state["last_search"]["context"][2]
            face_control = next(item for item in app.selectbox if item.label == "Pessoa que deve ser procurada")
            face_control.set_value(1 if selected != 1 else 0).run(timeout=900)
            assert "Fotos encontradas" not in [item.value for item in app.subheader]
            button(app, "Buscar fotos no álbum").click().run(timeout=900)
            require_healthy(app)
            assert app.session_state["last_search"]["context"][2] != selected
            checkpoint("explicit_person_switch", first_results=len(first_results))
            app.selectbox(key="search_mode_Uso pessoal").set_value(GLOBAL_MODE).run(timeout=900)
            button(app, "Buscar fotos no álbum").click().run(timeout=900)
            require_healthy(app)
            global_results = app.session_state["last_search"]["results"].copy()
            assert len(global_results) > 50
            query_hash = sha256_file(reference)
            source_ids = set(global_meta.loc[global_meta.sha256 == query_hash, "image_id"].astype(str))
            assert not source_ids.intersection(global_results.image_id.astype(str))
            pages = (len(global_results) + 23) // 24
            app.number_input(key="results_page").set_value(pages).run(timeout=900)
            require_healthy(app)
            assert len(app.session_state["last_search"]["results"]) == len(global_results)
            checkpoint("global_search_and_pagination", results=len(global_results), pages=pages,
                       source_excluded=True)
            app.radio(key="ui_profile").set_value("Pesquisa").run(timeout=900)
            assert app.selectbox(key="research_source").label == "Fonte das fotos"
            assert app.selectbox(key="research_source").value == "Meu álbum"
            app.selectbox(key="search_mode_Pesquisa").set_value(FUSION_MODE).run(timeout=900)
            button(app, "Buscar fotos no álbum").click().run(timeout=900)
            require_healthy(app)
            fused = app.session_state["last_search"]["results"]
            assert not fused.empty and np.isfinite(fused.score).all()
            assert not source_ids.intersection(fused.image_id.astype(str))
            checkpoint("experimental_fusion", results=len(fused))

        # The test album alone is updated through the real UI/subprocesses.
        before_update = {name: sha256_file(album_dir / name) for name in (
            "face/face_embeddings.npy", "face/face_metadata.csv",
            "global/global_embeddings.npy", "global/global_metadata.csv")}
        app.radio(key="app_page").set_value("Meus álbuns").run()
        app.radio(key="album_action").set_value("Preparar / atualizar").run()
        app.selectbox(key="album_edit_selection").set_value(str(album_dir)).run()
        print("[RUN] updating only the validation album", flush=True)
        button(app, "Atualizar álbum").click().run(timeout=900)
        require_healthy(app)
        after_update = {name: sha256_file(album_dir / name) for name in before_update}
        assert before_update == after_update
        assert not Path(str(album_dir) + "_v2").exists()
        checkpoint("update_same_destination", same_hashes=True)

        app = AppTest.from_file(str(ROOT / "scripts/app/streamlit_app.py"), default_timeout=900)
        app.session_state["album_selection"] = str(album_dir)
        app.session_state["ui_device"] = args.device
        app.session_state["search_mode_Uso pessoal"] = GLOBAL_MODE
        app.run()

        # Probe the real oversized JPEG, without creating or rewriting a photo.
        reduced = global_manifest["extra"].get("reduced_images", [])
        if not reduced:
            raise AssertionError("This run must include a real JPEG using controlled reduced decoding")
        large_path = Path(reduced[0]["image_path"])
        large_path = large_path if large_path.is_absolute() else ROOT / large_path
        with mock.patch("streamlit.file_uploader", return_value=ReferenceUpload(large_path)):
            app.run(timeout=900)
            button(app, "Buscar fotos no álbum").click().run(timeout=900)
            require_healthy(app)
            assert not app.session_state["last_search"]["results"].empty
            with JpegImagePlugin.JpegImageFile(large_path) as header:
                dimensions = header.size
            checkpoint("oversized_jpeg_upload_global_search_and_preview", source_pixels=dimensions[0] * dimensions[1])

        providers = {name: model.session.get_providers() for name, model in cached_face_app(args.device, 640).models.items()
                     if hasattr(model, "session")}
        checkpoint("actual_face_providers", requested=args.device, models=providers)
        app.radio(key="app_page").set_value("Meus álbuns").run()
        app.radio(key="album_action").set_value("Remover").run()
        app.selectbox(key="album_remove_selection").set_value(str(album_dir)).run()
        button(app, "Remover álbum").click().run()
        assert album_dir.exists() and len(app.error)
        next(item for item in app.checkbox if item.label.startswith("Confirmo que quero remover")).check()
        button(app, "Remover álbum").click().run()
        require_healthy(app)
        assert not album_dir.exists()
        checkpoint("confirmed_removal_only_test_album", refused_without_confirmation=True)
        assert snapshot(photos) == originals
        assert all(Path(folder).is_dir() and snapshot(Path(folder)) == hashes for folder, hashes in existing.items())
        checkpoint("all_originals_and_preexisting_albums_unchanged", files=len(originals))
        report.update(status="passed", elapsed_seconds=time.perf_counter() - started)
    except BaseException as exc:
        report.update(status="failed", error=str(exc), elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] acceptance report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
