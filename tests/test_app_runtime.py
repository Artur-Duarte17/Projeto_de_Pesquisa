from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.runtime import (
    FACE_MODE, PERSONAL_PROFILE, ReferenceFace, SearchSettings,
    annotated_faces_preview, cached_reference_faces, execute_search,
    save_upload, search_context,
)


class AppRuntimeTests(unittest.TestCase):
    def test_default_settings_are_personal_face_search_without_result_cap(self):
        settings = SearchSettings(PERSONAL_PROFILE, "Família", Path("family/face"), Path("family/global"))
        self.assertEqual(settings.mode, FACE_MODE)
        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.threshold, 0.35)
        self.assertIsNone(settings.topk)

    def test_uploads_use_content_names_and_do_not_rewrite_same_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo = SimpleNamespace(name="../../same_name.JPG", getbuffer=lambda: b"first synthetic photo")
            with mock.patch("app.runtime.OUTPUTS_DIR", root):
                first = save_upload(photo)
                timestamp = first.stat().st_mtime_ns
                again = save_upload(photo)
                other = save_upload(SimpleNamespace(name="same_name.JPG", getbuffer=lambda: b"other synthetic photo"))
            self.assertEqual(first, again)
            self.assertNotEqual(first, other)
            self.assertEqual(first.parent, root / "app_uploads")
            self.assertEqual(first.suffix, ".jpg")
            self.assertEqual(first.stat().st_mtime_ns, timestamp)
            self.assertEqual(first.read_bytes(), b"first synthetic photo")
            self.assertEqual(other.read_bytes(), b"other synthetic photo")

    def test_annotated_preview_is_bounded_and_does_not_change_source_pixels(self):
        image = np.zeros((600, 900, 3), dtype=np.uint8)
        before = image.copy()
        face = SimpleNamespace(bbox=np.asarray([200, 200, 600, 400], dtype=np.float32))
        preview = annotated_faces_preview(image, [face], max_side=90)
        self.assertEqual(preview.shape, (60, 90, 3))
        self.assertTrue(np.array_equal(image, before))
        self.assertFalse(np.all(preview == 0))

    def test_cached_face_reference_reuses_detection_and_keeps_original_coordinates(self):
        face = SimpleNamespace(bbox=np.asarray([2, 3, 12, 13], dtype=np.float32), embedding=np.asarray([3, 4], dtype=np.float32))
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        provider = object()
        cached_reference_faces.clear()
        try:
            with mock.patch("app.runtime.cached_face_app", return_value=provider), mock.patch(
                "app.runtime.detect_faces_in_image", return_value=(image, [face])
            ) as detection:
                preview, references = cached_reference_faces("mock_reference.jpg", "cpu")
                again, repeated = cached_reference_faces("mock_reference.jpg", "cpu")
            detection.assert_called_once_with(Path("mock_reference.jpg"), provider)
            self.assertEqual(references[0].bbox, (2.0, 3.0, 12.0, 13.0))
            np.testing.assert_allclose(references[0].vector, [0.6, 0.8])
            np.testing.assert_array_equal(preview, again)
            np.testing.assert_array_equal(references[0].vector, repeated[0].vector)
        finally:
            cached_reference_faces.clear()

    def test_index_changes_invalidate_saved_search_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "face_embeddings.npy").write_bytes(b"synthetic embedding")
            metadata = folder / "face_metadata.csv"
            metadata.write_text("first", encoding="utf-8")
            settings = SearchSettings(PERSONAL_PROFILE, "Família", folder, folder)
            initial = search_context(settings, Path("query.jpg"), 0)
            self.assertEqual(initial, search_context(settings, Path("query.jpg"), 0))
            metadata.write_text("changed metadata", encoding="utf-8")
            self.assertNotEqual(initial, search_context(settings, Path("query.jpg"), 0))
            self.assertNotEqual(initial, search_context(settings, Path("other_query.jpg"), 0))

    def test_ui_face_search_delegates_unchanged_vector_and_parameters_to_shared_gateway(self):
        settings = SearchSettings(PERSONAL_PROFILE, "Família", Path("face"), Path("global"))
        vector = np.asarray([0.6, 0.8], dtype=np.float32)
        face = ReferenceFace((0, 0, 10, 10), vector)
        embeddings = np.asarray([[1, 0]], dtype=np.float32)
        metadata = pd.DataFrame({"image_id": ["synthetic"]})
        expected = pd.DataFrame({"rank": [1]})
        with mock.patch("app.runtime.cached_face_index", return_value=(embeddings, metadata)), mock.patch(
            "app.runtime.search_face_index", return_value=expected
        ) as search:
            result = execute_search(settings, Path("query.jpg"), face)
        self.assertIs(result, expected)
        self.assertIs(search.call_args.args[0], vector)
        self.assertIs(search.call_args.args[1], embeddings)
        self.assertIs(search.call_args.args[2], metadata)
        self.assertEqual(search.call_args.kwargs, {"topk": None, "threshold": 0.35, "query_path": Path("query.jpg")})
