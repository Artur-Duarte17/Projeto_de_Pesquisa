from __future__ import annotations

from typing import Iterable

import pandas as pd


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


def apply_relevance_exclusions(
    relevance: dict[str, set[str]],
    exclusions: dict[str, set[str]],
) -> dict[str, set[str]]:
    query_ids = set(relevance) | set(exclusions)
    return {
        query_id: set(relevance.get(query_id, set())) - set(exclusions.get(query_id, set()))
        for query_id in query_ids
    }


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
