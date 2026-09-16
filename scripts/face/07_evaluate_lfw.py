from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from face_lib import load_face_index
from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval_common import (
    aggregate_metrics,
    apply_relevance_exclusions,
    parse_image_ids,
    relevance_from_csv,
)
from run_manifest import write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen LFW protocol with indexed query embeddings."
    )
    parser.add_argument(
        "--queries-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "lfw_face_queries.csv",
    )
    parser.add_argument(
        "--relevance-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "lfw_face_relevance.csv",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-008_lfw_face_index",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-009_lfw_face_evaluation",
    )
    parser.add_argument("--save-topk", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-queries", type=int, default=None)
    return parser.parse_args()


def largest_face_row(rows: pd.DataFrame) -> pd.Series:
    widths = rows["bbox_x2"].astype(float) - rows["bbox_x1"].astype(float)
    heights = rows["bbox_y2"].astype(float) - rows["bbox_y1"].astype(float)
    areas = widths.clip(lower=0.0) * heights.clip(lower=0.0)
    return rows.loc[areas.idxmax()]


def metrics_from_codes(
    ranked_codes: np.ndarray,
    relevant_codes: set[int],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float | int]:
    if not relevant_codes:
        row: dict[str, float | int] = {
            "num_relevant": 0,
            "ranking_size": int(len(ranked_codes)),
            "average_precision": 0.0,
        }
        for k in ks:
            row[f"precision_at_{k}"] = 0.0
            row[f"recall_at_{k}"] = 0.0
        return row

    relevant_array = np.fromiter(sorted(relevant_codes), dtype=np.int64)
    hits = np.isin(ranked_codes, relevant_array, assume_unique=True)
    relevant_total = len(relevant_codes)
    hit_positions = np.flatnonzero(hits) + 1
    cumulative_hits = np.arange(1, len(hit_positions) + 1, dtype=np.float64)
    average_precision = float(
        np.sum(cumulative_hits / hit_positions, dtype=np.float64) / relevant_total
    )

    row = {
        "num_relevant": relevant_total,
        "ranking_size": int(len(ranked_codes)),
        "average_precision": average_precision,
    }
    for k in ks:
        hits_at_k = int(hits[:k].sum())
        row[f"precision_at_{k}"] = hits_at_k / k
        row[f"recall_at_{k}"] = hits_at_k / relevant_total
    return row


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    args = parse_args()
    if args.save_topk <= 0:
        raise ValueError("save-topk must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")

    queries = pd.read_csv(args.queries_csv, dtype=str)
    if args.max_queries is not None:
        queries = queries.head(args.max_queries).copy()
    required_query_columns = {"query_id", "query_path", "source_image_id"}
    missing_query_columns = sorted(required_query_columns - set(queries.columns))
    if missing_query_columns:
        raise ValueError(f"Missing query columns: {missing_query_columns}")
    if queries.empty:
        raise ValueError("No LFW queries were provided")

    embeddings, metadata = load_face_index(args.index_dir)
    required_metadata_columns = {
        "image_id",
        "image_path",
        "embedding_row",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    }
    missing_metadata_columns = sorted(required_metadata_columns - set(metadata.columns))
    if missing_metadata_columns:
        raise ValueError(f"Missing face metadata columns: {missing_metadata_columns}")

    metadata = metadata.sort_values("embedding_row").reset_index(drop=True)
    expected_rows = np.arange(len(metadata), dtype=np.int64)
    if not np.array_equal(metadata["embedding_row"].to_numpy(dtype=np.int64), expected_rows):
        raise ValueError("Face metadata is not aligned with the embedding matrix")
    if not np.isfinite(embeddings).all():
        raise ValueError("The face index contains non-finite embeddings")

    image_table = (
        metadata[["image_id", "image_path"]]
        .drop_duplicates("image_id")
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    image_ids = image_table["image_id"].astype(str).tolist()
    image_id_to_code = {image_id: code for code, image_id in enumerate(image_ids)}
    face_image_codes_np = metadata["image_id"].astype(str).map(image_id_to_code).to_numpy(np.int64)
    faces_by_image_code = {
        code: np.flatnonzero(face_image_codes_np == code)
        for code in range(len(image_ids))
    }

    relevance = relevance_from_csv(args.relevance_csv)
    exclusions: dict[str, set[str]] = {}
    for query in queries.itertuples(index=False):
        query_id = str(query.query_id)
        excluded = set(parse_image_ids(str(query.source_image_id)))
        if hasattr(query, "exclude_image_ids"):
            additional = query.exclude_image_ids
            if pd.notna(additional):
                excluded.update(parse_image_ids(str(additional)))
        if not excluded:
            raise ValueError(f"No explicit source exclusion for query: {query_id}")
        exclusions[query_id] = excluded
    relevance = apply_relevance_exclusions(relevance, exclusions)

    query_embeddings = []
    query_bboxes = []
    for query in queries.itertuples(index=False):
        source_image_id = str(query.source_image_id)
        source_rows = metadata[metadata["image_id"].astype(str) == source_image_id]
        if source_rows.empty:
            raise ValueError(f"Source image is absent from the index: {source_image_id}")
        selected_face = largest_face_row(source_rows)
        embedding_row = int(selected_face["embedding_row"])
        query_embeddings.append(np.asarray(embeddings[embedding_row], dtype=np.float32))
        query_bboxes.append(
            ",".join(
                f"{float(selected_face[column]):.1f}"
                for column in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
            )
        )

    query_matrix_np = np.vstack(query_embeddings).astype(np.float32)
    device = torch.device(args.device)
    gallery_tensor = functional.normalize(
        torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device), dim=1
    )
    query_tensor = functional.normalize(torch.from_numpy(query_matrix_np).to(device), dim=1)
    face_image_codes = torch.from_numpy(face_image_codes_np).to(device)

    topk_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for batch_start in range(0, len(queries), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(queries))
        batch_queries = query_tensor[batch_start:batch_end]
        synchronize(device)
        started = time.perf_counter()
        face_scores = batch_queries @ gallery_tensor.T
        image_scores = torch.full(
            (len(batch_queries), len(image_ids)),
            -torch.inf,
            dtype=face_scores.dtype,
            device=device,
        )
        image_scores.scatter_reduce_(
            1,
            face_image_codes.unsqueeze(0).expand(len(batch_queries), -1),
            face_scores,
            reduce="amax",
            include_self=True,
        )

        batch_excluded_codes: list[set[int]] = []
        for local_index, query in enumerate(
            queries.iloc[batch_start:batch_end].itertuples(index=False)
        ):
            excluded_codes = {
                image_id_to_code[image_id]
                for image_id in exclusions.get(str(query.query_id), set())
                if image_id in image_id_to_code
            }
            if not excluded_codes:
                raise ValueError(f"No source exclusion was resolved for query: {query.query_id}")
            batch_excluded_codes.append(excluded_codes)
            image_scores[local_index, list(excluded_codes)] = -torch.inf

        ranked_image_codes = torch.argsort(image_scores, dim=1, descending=True)
        synchronize(device)
        elapsed_per_query_ms = (time.perf_counter() - started) * 1000.0 / len(batch_queries)
        ranked_image_codes_np = ranked_image_codes.cpu().numpy()
        face_scores_np = face_scores.cpu().numpy()

        for local_index, query in enumerate(
            queries.iloc[batch_start:batch_end].itertuples(index=False)
        ):
            query_index = batch_start + local_index
            query_id = str(query.query_id)
            excluded_codes = batch_excluded_codes[local_index]
            ranked_codes = np.asarray(
                [
                    int(code)
                    for code in ranked_image_codes_np[local_index]
                    if int(code) not in excluded_codes
                ],
                dtype=np.int64,
            )
            relevant_ids = relevance.get(query_id, set())
            missing_relevant_ids = sorted(set(relevant_ids) - set(image_id_to_code))
            if missing_relevant_ids:
                raise ValueError(
                    f"Relevant images absent from index for {query_id}: {missing_relevant_ids[:3]}"
                )
            relevant_codes = {image_id_to_code[image_id] for image_id in relevant_ids}
            row = {
                "query_id": query_id,
                **metrics_from_codes(ranked_codes, relevant_codes),
                "query_time_ms": elapsed_per_query_ms,
            }
            metric_rows.append(row)

            for rank, image_code in enumerate(ranked_codes[: args.save_topk], start=1):
                candidate_face_rows = faces_by_image_code[int(image_code)]
                winning_face_row = int(
                    candidate_face_rows[
                        np.argmax(face_scores_np[local_index, candidate_face_rows])
                    ]
                )
                result = metadata.iloc[winning_face_row].to_dict()
                result.update(
                    {
                        "query_id": query_id,
                        "query_path": str(query.query_path),
                        "query_bbox": query_bboxes[query_index],
                        "query_time_ms": elapsed_per_query_ms,
                        "score": float(face_scores_np[local_index, winning_face_row]),
                        "rank": rank,
                    }
                )
                topk_rows.append(result)

        print(f"[INFO] evaluated queries: {batch_end}/{len(queries)}")

    topk = pd.DataFrame(topk_rows)
    per_query = pd.DataFrame(metric_rows)
    metrics = aggregate_metrics(per_query, "face_lfw_indexed_queries")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    topk_path = args.output_dir / "face_topk_results.csv"
    per_query_path = args.output_dir / "face_metrics_per_query.csv"
    metrics_path = args.output_dir / "face_metrics.csv"
    topk.to_csv(topk_path, index=False)
    per_query.to_csv(per_query_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    manifest_path = write_run_manifest(
        args.output_dir / "face_run_manifest.json",
        script_path=Path(__file__),
        method="face_lfw_indexed_queries",
        configuration={
            "save_topk": args.save_topk,
            "ranking_depth": "all_eligible_images",
            "metric_ks": [5, 10],
            "threshold": None,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_queries": args.max_queries,
            "query_embedding_source": "largest_face_embedding_from_ex008_index",
        },
        inputs={
            "queries_csv": args.queries_csv,
            "relevance_csv": args.relevance_csv,
            "face_embeddings": args.index_dir / "face_embeddings.npy",
            "face_metadata": args.index_dir / "face_metadata.csv",
        },
        result_files={
            "saved_topk": topk_path,
            "per_query_metrics": per_query_path,
            "aggregate_metrics": metrics_path,
        },
        extra={
            "failed_queries": 0,
            "query_embeddings_reused": int(len(queries)),
            "eligible_gallery_images": int(len(image_ids)),
        },
    )

    print(f"[OK] Face metrics: {metrics_path}")
    print(f"[OK] Run manifest: {manifest_path}")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
