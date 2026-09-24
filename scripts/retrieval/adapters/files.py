from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from retrieval.domain.metrics import parse_image_ids

ROOT = Path(__file__).resolve().parents[3]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder: Path, max_images: int | None = None) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(folder.rglob(f"*{ext}"))
        paths.extend(folder.rglob(f"*{ext.upper()}"))
    out = sorted({p for p in paths if ".ipynb_checkpoints" not in str(p)})
    if max_images is not None:
        out = out[:max_images]
    return out


def rel_to_root(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def query_id_from_path(path: Path) -> str:
    return Path(rel_to_root(path)).stem


def resolve_stored_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return ROOT / p


def stable_image_id(path: Path, base_dir: Path) -> str:
    try:
        rel = path.resolve().relative_to(base_dir.resolve()).as_posix().lower()
    except ValueError:
        rel = rel_to_root(path).lower()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
    return digest


def load_inventory_image_ids(
    inventory_csv: Path,
    input_dir: Path,
    image_paths: Iterable[Path],
) -> dict[Path, str]:
    inventory = pd.read_csv(inventory_csv, dtype=str).fillna("")
    if "image_id" not in inventory.columns:
        raise ValueError("Inventory CSV must contain an image_id column")
    if "image_path" not in inventory.columns and "file_name" not in inventory.columns:
        raise ValueError("Inventory CSV must contain image_path or file_name")

    mapping: dict[Path, str] = {}
    used_ids: dict[str, Path] = {}
    for row in inventory.to_dict(orient="records"):
        image_id = str(row["image_id"]).strip()
        if not image_id:
            raise ValueError("Inventory CSV contains an empty image_id")
        stored_path = str(row.get("image_path", "")).strip()
        if stored_path:
            candidate = resolve_stored_path(stored_path).resolve()
        else:
            file_name = str(row.get("file_name", "")).strip()
            if not file_name:
                raise ValueError(f"Inventory row {image_id} has no usable image path")
            candidate = (input_dir / file_name).resolve()
        if candidate in mapping:
            raise ValueError(f"Inventory contains a duplicate path: {candidate}")
        if image_id in used_ids:
            raise ValueError(
                f"Inventory contains duplicate image_id {image_id}: "
                f"{used_ids[image_id]} and {candidate}"
            )
        mapping[candidate] = image_id
        used_ids[image_id] = candidate

    required_paths = {path.resolve() for path in image_paths}
    missing = sorted(str(path) for path in required_paths - set(mapping))
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"Inventory does not map {len(missing)} indexed image(s): {preview}"
        )
    return {path: mapping[path] for path in required_paths}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parent_label(path: Path) -> str:
    return path.parent.name


def read_image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return int(w), int(h)


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def relevance_from_csv(path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(path)
    required = {"query_id", "image_id", "relevant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in relevance CSV: {sorted(missing)}")
    rel: dict[str, set[str]] = {}
    for row in df.itertuples(index=False):
        if int(getattr(row, "relevant")):
            rel.setdefault(str(getattr(row, "query_id")), set()).add(str(getattr(row, "image_id")))
    return rel


def image_ids_for_query_path(query_path: Path, metadata: pd.DataFrame) -> set[str]:
    if "image_id" not in metadata.columns:
        return set()

    matched: set[str] = set()
    resolved_query = resolve_stored_path(str(query_path)).resolve()

    if "image_path" in metadata.columns:
        for row in metadata[["image_id", "image_path"]].drop_duplicates().itertuples(index=False):
            try:
                stored_path = resolve_stored_path(str(row.image_path)).resolve()
            except OSError:
                continue
            if stored_path == resolved_query:
                matched.add(str(row.image_id))

    if "sha256" in metadata.columns and resolved_query.is_file():
        query_hash = sha256_file(resolved_query)
        hashes = metadata["sha256"].fillna("").astype(str).str.lower()
        ids = metadata.loc[hashes == query_hash.lower(), "image_id"].astype(str)
        matched.update(ids.tolist())

    return matched


def excluded_image_ids_for_query(row: object, metadata: pd.DataFrame) -> set[str]:
    excluded: set[str] = set()
    for column in ("source_image_id", "exclude_image_ids"):
        if hasattr(row, column):
            excluded.update(parse_image_ids(getattr(row, column)))
    if hasattr(row, "query_path"):
        excluded.update(image_ids_for_query_path(Path(str(getattr(row, "query_path"))), metadata))
    return excluded


def build_query_exclusions(
    queries: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, set[str]]:
    exclusions: dict[str, set[str]] = {}
    for row in queries.itertuples(index=False):
        query_id = str(getattr(row, "query_id"))
        exclusions[query_id] = excluded_image_ids_for_query(row, metadata)
    return exclusions


def exclude_query_images_from_relevance(
    queries: pd.DataFrame,
    metadata: pd.DataFrame,
    relevance: dict[str, set[str]],
) -> dict[str, set[str]]:
    from retrieval.domain.metrics import apply_relevance_exclusions

    exclusions = build_query_exclusions(queries, metadata)
    return apply_relevance_exclusions(relevance, exclusions)


def save_visual_grid(
    results: pd.DataFrame,
    out_path: Path,
    title: str,
    max_images: int = 12,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = results.head(max_images)
    if subset.empty:
        return
    cols = min(4, len(subset))
    rows = int(np.ceil(len(subset) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_arr = np.array(axes).reshape(-1)
    for ax in axes_arr:
        ax.axis("off")
    for ax, row in zip(axes_arr, subset.itertuples(index=False)):
        img_path = resolve_stored_path(str(getattr(row, "image_path")))
        img = cv2.imread(str(img_path))
        if img is None:
            ax.set_title("missing")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        score = getattr(row, "score", None)
        label = f"rank {getattr(row, 'rank')}"
        if score is not None:
            label += f" | {float(score):.3f}"
        ax.set_title(label)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
