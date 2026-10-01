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
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for image_id in top if image_id in relevant_ids)
    return hits / len(top)


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


def exclude_query_images_from_relevance(
    queries: pd.DataFrame,
    metadata: pd.DataFrame,
    relevance: dict[str, set[str]],
) -> dict[str, set[str]]:
    if "query_path" not in queries.columns or "image_path" not in metadata.columns:
        return relevance
    out = {qid: set(ids) for qid, ids in relevance.items()}
    image_paths = metadata[["image_id", "image_path"]].drop_duplicates()
    resolved_to_ids: dict[Path, set[str]] = {}
    for row in image_paths.itertuples(index=False):
        try:
            resolved = resolve_stored_path(str(row.image_path)).resolve()
        except OSError:
            continue
        resolved_to_ids.setdefault(resolved, set()).add(str(row.image_id))

    for row in queries.itertuples(index=False):
        query_id = str(getattr(row, "query_id"))
        try:
            query_path = resolve_stored_path(str(getattr(row, "query_path"))).resolve()
        except OSError:
            continue
        for image_id in resolved_to_ids.get(query_path, set()):
            out.setdefault(query_id, set()).discard(image_id)
    return out


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
