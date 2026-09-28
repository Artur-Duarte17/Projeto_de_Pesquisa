from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
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

    def test_oversized_image_uses_reduced_opencv_decode(self):
        reduced_bgr = np.zeros((1800, 900, 3), dtype=np.uint8)
        reduced_bgr[:, :, 0] = 255
        with mock.patch(
            "retrieval.adapters.image_io.Image.open",
            side_effect=Image.DecompressionBombError("oversized"),
        ), mock.patch(
            "retrieval.adapters.image_preview.cv2.imread", return_value=reduced_bgr
        ) as read:
            preview = load_gallery_preview(Path("oversized.jpg"))
        read.assert_called_once_with("oversized.jpg", cv2.IMREAD_REDUCED_COLOR_8)
        self.assertEqual(preview.shape, (1200, 600, 3))
        self.assertEqual(preview[0, 0].tolist(), [0, 0, 255])

    def test_unreadable_oversized_image_is_reported(self):
        with mock.patch(
            "retrieval.adapters.image_io.Image.open",
            side_effect=Image.DecompressionBombError("oversized"),
        ), mock.patch("retrieval.adapters.image_preview.cv2.imread", return_value=None):
            with self.assertRaisesRegex(ValueError, "Could not create preview"):
                load_gallery_preview(Path("unreadable.jpg"))
