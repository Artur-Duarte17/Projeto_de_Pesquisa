from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

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


def l2_normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr) + eps
    return arr / norm


def l2_normalize_matrix(mat: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True) + eps
    return arr / norm


def cosine_scores(embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
    E = l2_normalize_matrix(embeddings)
    q = l2_normalize(query)
    return E @ q


def cosine_to_unit(score: float | np.ndarray) -> float | np.ndarray:
    return np.clip((score + 1.0) / 2.0, 0.0, 1.0)


def read_image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return int(w), int(h)


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for image_id in ranked_ids[:k] if image_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for image_id in ranked_ids[:k] if image_id in relevant_ids)
    return hits / len(relevant_ids)


def average_precision(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    hits = 0
    precisions = []
    seen: set[str] = set()
    for rank, image_id in enumerate(ranked_ids, start=1):
        if image_id in seen:
            continue
        seen.add(image_id)
        if image_id in relevant_ids:
            hits += 1
            precisions.append(hits / rank)
    if not precisions:
        return 0.0
    return float(sum(precisions) / len(relevant_ids))


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


def relevance_from_labels(
    queries: pd.DataFrame,
    metadata: pd.DataFrame,
    label_col: str,
) -> dict[str, set[str]]:
    rel: dict[str, set[str]] = {}
    if "target_label" not in queries.columns:
        return rel
    if label_col not in metadata.columns:
        return rel
    for row in queries.itertuples(index=False):
        query_id = str(getattr(row, "query_id"))
        target = str(getattr(row, "target_label"))
        ids = metadata.loc[metadata[label_col].astype(str) == target, "image_id"].astype(str)
        rel[query_id] = set(ids.tolist())
    return rel


def parse_image_ids(value: object) -> set[str]:
    if value is None:
        return set()
    try:
        if bool(pd.isna(value)):
            return set()
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return set()
    for separator in (";", "|"):
        text = text.replace(separator, ",")
    return {item.strip() for item in text.split(",") if item.strip()}


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


def apply_relevance_exclusions(
    relevance: dict[str, set[str]],
    exclusions: dict[str, set[str]],
) -> dict[str, set[str]]:
    query_ids = set(relevance) | set(exclusions)
    return {
        query_id: set(relevance.get(query_id, set())) - set(exclusions.get(query_id, set()))
        for query_id in query_ids
    }


def exclude_query_images_from_relevance(
    queries: pd.DataFrame,
    metadata: pd.DataFrame,
    relevance: dict[str, set[str]],
) -> dict[str, set[str]]:
    exclusions = build_query_exclusions(queries, metadata)
    return apply_relevance_exclusions(relevance, exclusions)


def summarize_rankings(
    rankings: dict[str, list[str]],
    relevance: dict[str, set[str]],
    ks: Iterable[int] = (5, 10),
) -> pd.DataFrame:
    rows = []
    for query_id, ranked_ids in rankings.items():
        rel = relevance.get(query_id, set())
        row = {
            "query_id": query_id,
            "num_relevant": len(rel),
            "ranking_size": len(ranked_ids),
            "average_precision": average_precision(ranked_ids, rel),
        }
        for k in ks:
            row[f"precision_at_{k}"] = precision_at_k(ranked_ids, rel, k)
            row[f"recall_at_{k}"] = recall_at_k(ranked_ids, rel, k)
            row[f"false_positives_at_{k}"] = sum(
                1 for image_id in ranked_ids[:k] if image_id not in rel
            )
        rows.append(row)
    return pd.DataFrame(rows)


def select_results_for_storage(results: pd.DataFrame, save_topk: int) -> pd.DataFrame:
    if save_topk <= 0:
        raise ValueError("save_topk must be positive")
    return results.head(save_topk).copy()


def aggregate_metrics(per_query: pd.DataFrame, method: str) -> pd.DataFrame:
    if per_query.empty:
        return pd.DataFrame([{"method": method, "num_queries": 0}])
    metric_cols = [
        c
        for c in per_query.columns
        if c.startswith("precision_at_")
        or c.startswith("recall_at_")
        or c.startswith("false_positives_at_")
        or c == "average_precision"
        or c == "query_time_ms"
    ]
    row = {"method": method, "num_queries": int(len(per_query))}
    for col in metric_cols:
        row[col] = float(per_query[col].mean())
    if "average_precision" in per_query.columns:
        row["mAP"] = float(per_query["average_precision"].mean())
    return pd.DataFrame([row])


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
