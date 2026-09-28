from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval.adapters.image_preview import load_gallery_preview


class GalleryPreviewTests(unittest.TestCase):
    def test_regular_image_is_reduced_in_memory(self):
        image = Image.new("RGB", (1600, 800), color=(10, 20, 30))
        with mock.patch("retrieval.adapters.image_io.Image.open", return_value=image):
            preview = load_gallery_preview(Path("ordinary.jpg"), max_side=400)
        self.assertEqual(preview.shape, (200, 400, 3))
        self.assertEqual(preview[0, 0].tolist(), [10, 20, 30])

    def test_oversized_jpeg_uses_shared_controlled_decoder(self):
        reduced = Image.new("RGB", (1600, 800), color=(10, 20, 30))
        with mock.patch(
            "retrieval.adapters.image_io.Image.open",
            side_effect=Image.DecompressionBombError("oversized"),
        ), mock.patch(
            "retrieval.adapters.image_io._open_large_jpeg", return_value=reduced
        ) as decode:
            preview = load_gallery_preview(Path("oversized.jpg"))
        decode.assert_called_once_with(Path("oversized.jpg"))
        self.assertEqual(preview.shape, (600, 1200, 3))
        self.assertEqual(preview[0, 0].tolist(), [10, 20, 30])

    def test_oversized_non_jpeg_is_reported(self):
        with mock.patch(
            "retrieval.adapters.image_io.Image.open",
            side_effect=Image.DecompressionBombError("oversized"),
        ):
            with self.assertRaisesRegex(ValueError, "supported safe preview size"):
                load_gallery_preview(Path("oversized.png"))
