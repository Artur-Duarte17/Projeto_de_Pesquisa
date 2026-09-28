"""Create bounded, in-memory previews for the local result gallery."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from retrieval.adapters.image_io import open_image, oriented_rgb


def load_gallery_preview(image_path: Path, max_side: int = 1200) -> np.ndarray:
    if max_side <= 0:
        raise ValueError("max_side must be positive")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with open_image(image_path) as source:
                source.draft("RGB", (max_side, max_side))
                source.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                return np.asarray(oriented_rgb(source, image_path))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f"Image exceeds the supported safe preview size: {image_path}") from exc
