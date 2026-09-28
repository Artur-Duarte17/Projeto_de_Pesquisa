"""Local image decoding, including optional HEIF/HEIC photographs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


HEIF_EXTS = {".heic", ".heif"}


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
    return Image.open(path)


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
