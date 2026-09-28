from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.runtime import ReferenceFace
from retrieval.adapters.album_store import Album

SUMMARY = {"images_scanned": 5, "global_failures": 0, "face_read_failures": 0}


def button(app, label):
    return next(item for item in app.button if item.label == label)


def results_pagination_app():
    import pandas as pd
    from app.components import show_search_results

    results = pd.DataFrame({
        "rank": list(range(1, 62)),
        "score": [0.9] * 61,
        "image_path": [f"photo_{number}.jpg" for number in range(1, 62)],
    })
    show_search_results(results)


class AppOnboardingTests(unittest.TestCase):
    def test_without_albums_opens_personal_onboarding_without_running_models(self):
        with mock.patch("retrieval.adapters.album_store.discover_albums", return_value=[]), mock.patch(
            "retrieval.adapters.album_store.prepare_album"
        ) as prepare, mock.patch("app.runtime.execute_search") as search:
            app = AppTest.from_file(str(ROOT / "scripts/app/streamlit_app.py")).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.radio(key="ui_profile").value, "Uso pessoal")
        self.assertEqual(app.radio(key="app_page").value, "Buscar fotos")
        self.assertIn("Preparar meu primeiro álbum", [item.label for item in app.button])
        self.assertNotIn("Fonte das fotos", [item.label for item in app.selectbox])
        self.assertEqual(len(app.file_uploader), 0)
        prepare.assert_not_called()
        search.assert_not_called()

    def test_ready_album_supplies_indexes_without_manual_paths_or_gallagher(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            album = Album("Família", folder / "indexes", folder, True)
            with mock.patch("retrieval.adapters.album_store.discover_albums", return_value=[album]), mock.patch(
                "retrieval.adapters.album_store.album_summary", return_value=SUMMARY
            ):
                app = AppTest.from_file(str(ROOT / "scripts/app/streamlit_app.py"))
                app.session_state["manual_indexes"] = True
                app.session_state["preset"] = "Gallagher final"
                app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.radio(key="ui_profile").value, "Uso pessoal")
        self.assertEqual(app.selectbox(key="album_selection").value, str(album.output_dir))
        self.assertFalse(app.checkbox(key="show_technical_Uso pessoal").value)
        self.assertNotIn("Fonte das fotos", [item.label for item in app.selectbox])
        self.assertEqual(len(app.text_input), 0)

    def test_research_requires_opt_in_and_defaults_to_personal_album_not_gallagher(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            album = Album("Família", folder / "indexes", folder, True)
            with mock.patch("retrieval.adapters.album_store.discover_albums", return_value=[album]), mock.patch(
                "retrieval.adapters.album_store.album_summary", return_value=SUMMARY
            ), mock.patch("app.runtime.execute_search") as search:
                app = AppTest.from_file(str(ROOT / "scripts/app/streamlit_app.py")).run()
                app.radio(key="ui_profile").set_value("Pesquisa").run()
                self.assertEqual(app.selectbox(key="research_source").label, "Fonte das fotos")
                self.assertEqual(app.selectbox(key="research_source").value, "Meu álbum")
                app.selectbox(key="research_source").set_value("Gallagher final").run()
                self.assertEqual(app.selectbox(key="research_source").value, "Gallagher final")
                app.radio(key="ui_profile").set_value("Uso pessoal").run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("Fonte das fotos", [item.label for item in app.selectbox])
        self.assertEqual(app.selectbox(key="album_selection").value, str(album.output_dir))
        search.assert_not_called()

    def test_submit_prepares_once_and_selects_new_album_for_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            photos = task / "photos"
            photos.mkdir()
            output = task / "collections" / "familia"
            album = Album("Família", output, photos, True)
            discovered = []

            def fake_prepare(*args, **kwargs):
                discovered.append(album)
                return "mock preparation completed"

            with mock.patch("retrieval.adapters.album_store.discover_albums", side_effect=lambda: list(discovered)), mock.patch(
                "retrieval.adapters.album_store.album_output_dir", return_value=output
            ), mock.patch("retrieval.adapters.album_store.album_summary", return_value=SUMMARY), mock.patch(
                "retrieval.adapters.album_store.prepare_album", side_effect=fake_prepare
            ) as prepare:
                app = AppTest.from_file(str(ROOT / "scripts/app/streamlit_app.py")).run()
                button(app, "Preparar meu primeiro álbum").click().run()
                self.assertEqual(app.radio(key="app_page").value, "Meus álbuns")
                next(item for item in app.text_input if item.label == "Nome do álbum").set_value("Família")
                next(item for item in app.text_input if item.label == "Pasta com as fotos neste computador").set_value(str(photos))
                button(app, "Preparar álbum").click().run()

            self.assertEqual(len(app.exception), 0)
            prepare.assert_called_once_with(photos, output, "Família", device="cpu", overwrite=False)
            self.assertEqual(app.radio(key="app_page").value, "Buscar fotos")
            self.assertEqual(app.selectbox(key="album_selection").value, str(output))
            self.assertIn("preparado", app.success[0].value)

    def test_result_controls_allow_all_or_a_limit_above_fifty(self):
        with mock.patch("retrieval.adapters.album_store.discover_albums", return_value=[]):
            app = AppTest.from_file(str(ROOT / "scripts/app/streamlit_app.py")).run()
            self.assertTrue(app.checkbox(key="all_results_Uso pessoal").value)
            app.checkbox(key="all_results_Uso pessoal").uncheck().run()
            app.number_input(key="result_limit_Uso pessoal").set_value(200).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.number_input(key="result_limit_Uso pessoal").value, 200)

    def test_removal_requires_confirmation_in_the_interface(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            album = Album("Família", folder / "indexes", folder, True)
            discovered = [album]

            def fake_remove(output_dir, *, confirmed):
                if not confirmed:
                    raise ValueError("Confirme a remoção dos índices deste álbum")
                discovered.clear()

            with mock.patch("retrieval.adapters.album_store.discover_albums", side_effect=lambda: list(discovered)), mock.patch(
                "retrieval.adapters.album_store.album_summary", return_value=SUMMARY
            ), mock.patch("retrieval.adapters.album_store.remove_album", side_effect=fake_remove) as removal:
                app = AppTest.from_file(str(ROOT / "scripts/app/streamlit_app.py")).run()
                app.radio(key="app_page").set_value("Meus álbuns").run()
                app.radio(key="album_action").set_value("Remover").run()
                button(app, "Remover álbum").click().run()
                removal.assert_called_once_with(album.output_dir, confirmed=False)
                removal.reset_mock()
                next(item for item in app.checkbox if item.label.startswith("Confirmo")).check()
                button(app, "Remover álbum").click().run()
            self.assertEqual(len(app.exception), 0)
            removal.assert_called_once_with(album.output_dir, confirmed=True)
            self.assertEqual(app.radio(key="app_page").value, "Meus álbuns")
            self.assertIn("fotos originais foram preservadas", app.success[0].value.lower())

    def test_pagination_keeps_results_beyond_fifty_and_loads_only_visible_photos(self):
        with mock.patch("app.components.load_gallery_preview", return_value=np.zeros((4, 4, 3), dtype=np.uint8)) as preview:
            app = AppTest.from_function(results_pagination_app).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(preview.call_count, 24)
            self.assertTrue(any("1–24 de 61" in item.value for item in app.caption))
            preview.reset_mock()
            app.number_input(key="results_page").set_value(3).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(preview.call_count, 13)
            self.assertTrue(any("49–61 de 61" in item.value for item in app.caption))
            self.assertEqual(len(app.dataframe), 0)

    def test_saved_results_follow_selected_face_and_visual_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            album = Album("Família", folder / "indexes", folder, True)
            query = folder / "reference.jpg"
            preview = np.zeros((8, 8, 3), dtype=np.uint8)
            face = ReferenceFace((0.0, 0.0, 7.0, 7.0), np.asarray([1.0, 0.0], dtype=np.float32))
            other_face = ReferenceFace((8.0, 0.0, 14.0, 7.0), np.asarray([0.0, 1.0], dtype=np.float32))
            results = pd.DataFrame({"rank": [1], "score": [0.9], "image_path": ["mock_result.jpg"]})
            uploaded = mock.Mock(name="uploaded photo")
            uploaded.name = "reference.jpg"
            with mock.patch("retrieval.adapters.album_store.discover_albums", return_value=[album]), mock.patch(
                "retrieval.adapters.album_store.album_summary", return_value=SUMMARY
            ), mock.patch("streamlit.file_uploader", return_value=uploaded), mock.patch(
                "app.runtime.save_upload", return_value=query
            ), mock.patch("retrieval.adapters.image_preview.load_gallery_preview", return_value=preview), mock.patch(
                "app.components.load_gallery_preview", return_value=preview
            ), mock.patch("app.runtime.cached_reference_faces", return_value=(preview, [face, other_face])), mock.patch(
                "app.runtime.execute_search", return_value=results
            ) as search:
                app = AppTest.from_file(str(ROOT / "scripts/app/streamlit_app.py")).run()
                button(app, "Buscar fotos no álbum").click().run()
                self.assertEqual(search.call_count, 1)
                self.assertIs(search.call_args.args[2], face)
                self.assertIn("Fotos encontradas", [item.value for item in app.subheader])
                app.checkbox(key="show_technical_Uso pessoal").check().run()
                self.assertEqual(search.call_count, 1)
                self.assertEqual(len(app.exception), 0)
                self.assertIn("Fotos encontradas", [item.value for item in app.subheader])
                next(item for item in app.selectbox if item.label == "Pessoa que deve ser procurada").set_value(1).run()
                self.assertEqual(search.call_count, 1)
                self.assertNotIn("Fotos encontradas", [item.value for item in app.subheader])
                button(app, "Buscar fotos no álbum").click().run()
                self.assertEqual(search.call_count, 2)
                self.assertIs(search.call_args.args[2], other_face)
                app.slider(key="threshold_Uso pessoal").set_value(0.5).run()
                self.assertEqual(search.call_count, 2)
                self.assertNotIn("Fotos encontradas", [item.value for item in app.subheader])
