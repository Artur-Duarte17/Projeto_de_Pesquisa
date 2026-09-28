from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval.adapters.image_io import (
    DECODE_POLICY_KEY,
    _open_large_jpeg,
    image_source_size,
    open_image,
)
from retrieval.adapters.image_preview import load_gallery_preview

spec = importlib.util.spec_from_file_location(
    "global_index_for_large_jpeg_tests", ROOT / "scripts/global/01_index_global_resnet.py"
)
GLOBAL_INDEX = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GLOBAL_INDEX)


class LargeJpegTests(unittest.TestCase):
    def test_real_jpeg_reduces_before_loading_and_keeps_source_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "photo.jpg"
            Image.new("RGB", (64, 48), color=(40, 80, 120)).save(path)
            # Simulate the real 200 MP guard error using a small JPEG. The
            # production limit is never disabled or changed by the reader.
            with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 100):
                with open_image(path) as image:
                    self.assertEqual(Image.MAX_IMAGE_PIXELS, 100)
                    self.assertEqual(image_source_size(image), (64, 48))
                    self.assertEqual(image.size, (8, 6))
                    self.assertEqual(image.convert("RGB").size, (8, 6))
                    self.assertEqual(image.info[DECODE_POLICY_KEY], "jpeg_reduced_1_8")
                prepared = GLOBAL_INDEX.prepare_image(path, lambda image: image.size)
                self.assertIsNone(prepared["error"])
                self.assertEqual((prepared["width"], prepared["height"]), (64, 48))
                self.assertEqual(prepared["tensor"], (8, 6))
                self.assertEqual(prepared["decode_policy"], "jpeg_reduced_1_8")
                self.assertEqual(load_gallery_preview(path).shape, (6, 8, 3))

    def test_jpeg_above_source_cap_is_closed_without_decoding(self):
        source = mock.Mock()
        source.size = (20_000, 20_000)
        with mock.patch(
            "retrieval.adapters.image_io.JpegImagePlugin.JpegImageFile", return_value=source
        ):
            with self.assertRaisesRegex(ValueError, "256 megapixels"):
                _open_large_jpeg(Path("too_large.jpg"))
        source.draft.assert_not_called()
        source.close.assert_called_once_with()

    def test_failed_decoder_reduction_is_closed_without_loading(self):
        source = mock.Mock()
        source.size = (12_240, 16_320)
        source.width, source.height = source.size
        with mock.patch(
            "retrieval.adapters.image_io.JpegImagePlugin.JpegImageFile", return_value=source
        ):
            with self.assertRaisesRegex(ValueError, "safe resolution"):
                _open_large_jpeg(Path("not_reduced.jpg"))
        source.close.assert_called_once_with()
        source.load.assert_not_called()

    def test_normal_jpeg_keeps_existing_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ordinary.jpg"
            Image.new("RGB", (64, 48)).save(path)
            with open_image(path) as image:
                self.assertEqual(image.size, (64, 48))
                self.assertNotIn(DECODE_POLICY_KEY, image.info)

    def test_non_jpeg_does_not_receive_size_exception(self):
        with mock.patch(
            "retrieval.adapters.image_io.Image.open",
            side_effect=Image.DecompressionBombError("oversized"),
        ), mock.patch("retrieval.adapters.image_io._open_large_jpeg") as decoder:
            with self.assertRaises(Image.DecompressionBombError):
                open_image(Path("oversized.png"))
        decoder.assert_not_called()
