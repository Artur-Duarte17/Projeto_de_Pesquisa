"""Create bounded, in-memory previews for the local result gallery."""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
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
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        # Keep Pillow's safety limit intact. Decode a reduced-resolution copy
        # instead of handing the original, very large image to Streamlit.
        reduced = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_8)
        if reduced is None:
            raise ValueError(f"Could not create preview for {image_path}")
        height, width = reduced.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        if scale < 1.0:
            reduced = cv2.resize(
                reduced,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(reduced, cv2.COLOR_BGR2RGB)
