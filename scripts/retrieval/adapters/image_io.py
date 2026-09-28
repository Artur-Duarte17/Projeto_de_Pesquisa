"""Local image decoding, including optional HEIF/HEIC photographs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, JpegImagePlugin


HEIF_EXTS = {".heic", ".heif"}
JPEG_EXTS = {".jpg", ".jpeg"}
SOURCE_SIZE_KEY = "_retrieval_source_size"
DECODE_POLICY_KEY = "_retrieval_decode_policy"
LARGE_JPEG_POLICY = {
    "max_source_pixels": 256_000_000,
    "max_decoded_pixels": 8_000_000,
    "decoder_reduction": 8,
}


def is_heif(path: Path) -> bool:
    return path.suffix.lower() in HEIF_EXTS


@lru_cache(maxsize=1)
def _register_heif_opener() -> None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError as exc:
        raise RuntimeError(
            "Para abrir HEIC/HEIF, instale pillow-heif==1.8.0 no ambiente do projeto"
        ) from exc
    register_heif_opener(thumbnails=False)


def open_image(path: Path) -> Image.Image:
    if is_heif(path):
        _register_heif_opener()
    try:
        return Image.open(path)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        if path.suffix.lower() not in JPEG_EXTS:
            raise
        return _open_large_jpeg(path)


def _open_large_jpeg(path: Path) -> Image.Image:
    # The JPEG-specific constructor reads the header without decoding pixels.
    # Apply explicit source and decoded limits before any pixel allocation;
    # do not change Image.MAX_IMAGE_PIXELS or enable truncated-file loading.
    try:
        source = JpegImagePlugin.JpegImageFile(path)
    except SyntaxError as exc:
        raise ValueError("Controlled oversized-image decoding only accepts valid JPEGs") from exc
    try:
        width, height = source.size
        if width * height > LARGE_JPEG_POLICY["max_source_pixels"]:
            raise ValueError("JPEG exceeds the controlled limit of 256 megapixels")
        reduction = LARGE_JPEG_POLICY["decoder_reduction"]
        source.draft("RGB", (max(1, width // reduction), max(1, height // reduction)))
        if source.width * source.height > LARGE_JPEG_POLICY["max_decoded_pixels"]:
            raise ValueError("JPEG decoder could not reduce the image to a safe resolution")
        source.info[SOURCE_SIZE_KEY] = (width, height)
        source.info[DECODE_POLICY_KEY] = "jpeg_reduced_1_8"
        return source
    except BaseException:
        source.close()
        raise


def image_source_size(image: Image.Image) -> tuple[int, int]:
    return tuple(image.info.get(SOURCE_SIZE_KEY, image.size))


def oriented_rgb(image: Image.Image, path: Path) -> Image.Image:
    if is_heif(path):
        image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def read_image_bgr(path: Path) -> np.ndarray | None:
    if not is_heif(path):
        return cv2.imread(str(path))
    try:
        with open_image(path) as image:
            rgb = np.asarray(oriented_rgb(image, path))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (OSError, ValueError):
        return None
