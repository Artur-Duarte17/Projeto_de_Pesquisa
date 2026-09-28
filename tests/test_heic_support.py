from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval.adapters.files import list_images
from retrieval.adapters.image_io import is_heif, open_image, oriented_rgb, read_image_bgr
from retrieval.adapters.image_preview import load_gallery_preview


class HeicSupportTests(unittest.TestCase):
    def test_heic_files_are_discovered_in_nested_folders(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            nested = folder / "trip"
            nested.mkdir()
            heic = folder / "family.HEIC"
            heif = nested / "portrait.heif"
            movie = nested / "clip.mov"
            for path in (heic, heif, movie):
                path.touch()
            self.assertEqual(set(list_images(folder)), {heic, heif})

    def test_heic_face_decoder_converts_rgb_to_bgr(self):
        image = Image.new("RGB", (4, 3), color=(10, 20, 30))
        path = Path("family.HEIC")
        with mock.patch("retrieval.adapters.image_io._register_heif_opener") as register, mock.patch(
            "retrieval.adapters.image_io.Image.open", return_value=image
        ):
            decoded = read_image_bgr(path)
        self.assertTrue(is_heif(path))
        register.assert_called_once_with()
        self.assertEqual(decoded.shape, (3, 4, 3))
        self.assertEqual(decoded[0, 0].tolist(), [30, 20, 10])

    def test_heic_preview_is_bounded_without_changing_original(self):
        image = Image.new("RGB", (1600, 800), color=(10, 20, 30))
        with mock.patch("retrieval.adapters.image_io._register_heif_opener"), mock.patch(
            "retrieval.adapters.image_io.Image.open", return_value=image
        ):
            preview = load_gallery_preview(Path("family.heic"), max_side=400)
        self.assertEqual(preview.shape, (200, 400, 3))
        self.assertEqual(preview[0, 0].tolist(), [10, 20, 30])

    def test_jpeg_still_uses_existing_opencv_decoder(self):
        expected = np.zeros((3, 4, 3), dtype=np.uint8)
        with mock.patch("retrieval.adapters.image_io.cv2.imread", return_value=expected) as read:
            result = read_image_bgr(Path("ordinary.jpg"))
        self.assertIs(result, expected)
        read.assert_called_once_with("ordinary.jpg")

    def test_open_heic_requests_plugin_registration(self):
        image = Image.new("RGB", (4, 3))
        with mock.patch("retrieval.adapters.image_io._register_heif_opener") as register, mock.patch(
            "retrieval.adapters.image_io.Image.open", return_value=image
        ) as opened:
            self.assertIs(open_image(Path("family.heif")), image)
        register.assert_called_once_with()
        opened.assert_called_once_with(Path("family.heif"))

    def test_heic_orientation_is_applied_without_changing_jpeg_policy(self):
        image = Image.new("RGB", (8, 4))
        image.getexif()[274] = 6
        self.assertEqual(oriented_rgb(image, Path("portrait.heic")).size, (4, 8))
        self.assertEqual(oriented_rgb(image, Path("portrait.jpg")).size, (8, 4))

    def test_real_heic_codec_reads_face_global_and_preview(self):
        # Synthetic temporary photograph: no private album or mock decoder.
        try:
            from pillow_heif import register_heif_opener
        except ImportError:
            self.fail("Instale pillow-heif==1.8.0 no ambiente antes de executar este teste")
        register_heif_opener(thumbnails=False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.heic"
            image = Image.new("RGB", (64, 48), color=(40, 80, 120))
            image.save(path, format="HEIF", quality=90)
            decoded_bgr = read_image_bgr(path)
            self.assertIsNotNone(decoded_bgr)
            self.assertEqual(decoded_bgr.shape, (48, 64, 3))
            with open_image(path) as source:
                global_rgb = np.asarray(oriented_rgb(source, path))
            np.testing.assert_array_equal(decoded_bgr[:, :, ::-1], global_rgb)
            self.assertEqual(load_gallery_preview(path, max_side=32).shape, (24, 32, 3))
