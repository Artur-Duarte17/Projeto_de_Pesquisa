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

from global_lib import load_global_index
from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval_common import (
    aggregate_metrics,
    apply_relevance_exclusions,
    parse_image_ids,
    relevance_from_csv,
    save_visual_grid,
    summarize_rankings,
)
from run_manifest import write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen Holidays protocol with indexed query descriptors."
    )
    parser.add_argument(
        "--queries-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "holidays_global_queries.csv",
    )
    parser.add_argument(
        "--relevance-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "holidays_relevance.csv",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-013_holidays_global_index",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-014_holidays_global_evaluation",
    )
    parser.add_argument("--save-topk", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--save-visual-examples", action="store_true")
    return parser.parse_args()


def normalized_stored_path(value: object) -> str:
    return Path(str(value)).as_posix().lower()


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
    required_query_columns = {"query_id", "query_path"}
    missing_query_columns = sorted(required_query_columns - set(queries.columns))
    if missing_query_columns:
        raise ValueError(f"Missing query columns: {missing_query_columns}")
    if queries.empty:
        raise ValueError("No Holidays queries were provided")
    if queries["query_id"].duplicated().any():
        raise ValueError("The query protocol contains duplicate query IDs")

    embeddings, metadata = load_global_index(args.index_dir)
    required_metadata_columns = {"image_id", "image_path", "embedding_row"}
    missing_metadata_columns = sorted(required_metadata_columns - set(metadata.columns))
    if missing_metadata_columns:
        raise ValueError(f"Missing global metadata columns: {missing_metadata_columns}")

    metadata = metadata.sort_values("embedding_row").reset_index(drop=True)
    expected_rows = np.arange(len(metadata), dtype=np.int64)
    if not np.array_equal(metadata["embedding_row"].to_numpy(np.int64), expected_rows):
        raise ValueError("Global metadata is not aligned with the embedding matrix")
    if metadata["image_id"].astype(str).duplicated().any():
        raise ValueError("The Holidays index contains duplicate image IDs")
    if not np.isfinite(embeddings).all():
        raise ValueError("The Holidays index contains non-finite descriptors")

    image_ids = metadata["image_id"].astype(str).tolist()
    image_id_to_code = {image_id: code for code, image_id in enumerate(image_ids)}
    path_to_code = {
        normalized_stored_path(row.image_path): int(row.embedding_row)
        for row in metadata.itertuples(index=False)
    }

    query_codes = []
    exclusions: dict[str, set[str]] = {}
    for query in queries.itertuples(index=False):
        query_id = str(query.query_id)
        normalized_path = normalized_stored_path(query.query_path)
        if normalized_path not in path_to_code:
            raise ValueError(f"Query image is absent from the index: {query.query_path}")
        source_code = path_to_code[normalized_path]
        source_image_id = image_ids[source_code]
        excluded = {source_image_id}
        if hasattr(query, "exclude_image_ids") and pd.notna(query.exclude_image_ids):
            excluded.update(parse_image_ids(str(query.exclude_image_ids)))
        query_codes.append(source_code)
        exclusions[query_id] = excluded

    relevance = apply_relevance_exclusions(relevance_from_csv(args.relevance_csv), exclusions)
    for query_id in queries["query_id"].astype(str):
        missing_relevant_ids = sorted(relevance.get(query_id, set()) - set(image_id_to_code))
        if missing_relevant_ids:
            raise ValueError(
                f"Relevant images absent from index for {query_id}: {missing_relevant_ids[:3]}"
            )

    device = torch.device(args.device)
    gallery_tensor = functional.normalize(
        torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device), dim=1
    )
    query_tensor = gallery_tensor[torch.as_tensor(query_codes, dtype=torch.long, device=device)]

    rankings: dict[str, list[str]] = {}
    query_times: dict[str, float] = {}
    topk_rows: list[dict[str, object]] = []
    for batch_start in range(0, len(queries), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(queries))
        batch_queries = query_tensor[batch_start:batch_end]
        synchronize(device)
        started = time.perf_counter()
        scores = batch_queries @ gallery_tensor.T

        batch_excluded_codes: list[set[int]] = []
        for local_index, query in enumerate(
            queries.iloc[batch_start:batch_end].itertuples(index=False)
        ):
            excluded_codes = {
                image_id_to_code[image_id]
                for image_id in exclusions[str(query.query_id)]
                if image_id in image_id_to_code
            }
            if not excluded_codes:
                raise ValueError(f"No source exclusion was resolved for query: {query.query_id}")
            batch_excluded_codes.append(excluded_codes)
            scores[local_index, list(excluded_codes)] = -torch.inf

        ranked_codes_tensor = torch.argsort(scores, dim=1, descending=True)
        synchronize(device)
        elapsed_per_query_ms = (time.perf_counter() - started) * 1000.0 / len(batch_queries)
        scores_np = scores.cpu().numpy()
        ranked_codes_np = ranked_codes_tensor.cpu().numpy()

        for local_index, query in enumerate(
            queries.iloc[batch_start:batch_end].itertuples(index=False)
        ):
            query_id = str(query.query_id)
            excluded_codes = batch_excluded_codes[local_index]
            ranked_codes = [
                int(code)
                for code in ranked_codes_np[local_index]
                if int(code) not in excluded_codes
            ]
            ranked_ids = [image_ids[code] for code in ranked_codes]
            rankings[query_id] = ranked_ids
            query_times[query_id] = elapsed_per_query_ms

            for rank, image_code in enumerate(ranked_codes[: args.save_topk], start=1):
                candidate = metadata.iloc[image_code]
                topk_rows.append(
                    {
                        "query_id": query_id,
                        "query_path": str(query.query_path),
                        "source_image_id": image_ids[query_codes[batch_start + local_index]],
                        "rank": rank,
                        "image_id": str(candidate["image_id"]),
                        "image_path": str(candidate["image_path"]),
                        "score": float(scores_np[local_index, image_code]),
                        "matched_face_id": "",
                        "bbox": "",
                        "query_time_ms": elapsed_per_query_ms,
                    }
                )
        print(f"[INFO] evaluated queries: {batch_end}/{len(queries)}")

    topk = pd.DataFrame(topk_rows)
    per_query = summarize_rankings(rankings, relevance, ks=(5, 10))
    per_query["query_time_ms"] = per_query["query_id"].map(query_times).fillna(0.0)
    metrics = aggregate_metrics(per_query, "global_resnet50_holidays_indexed_queries")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    topk_path = args.output_dir / "global_topk_results.csv"
    per_query_path = args.output_dir / "global_metrics_per_query.csv"
    metrics_path = args.output_dir / "global_metrics.csv"
    topk.to_csv(topk_path, index=False)
    per_query.to_csv(per_query_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    result_files = {
        "saved_topk": topk_path,
        "per_query_metrics": per_query_path,
        "aggregate_metrics": metrics_path,
    }
    if args.save_visual_examples:
        first_query_id = str(queries.iloc[0]["query_id"])
        visual_path = args.output_dir / "visual_examples" / "global_examples.png"
        save_visual_grid(
            topk[topk["query_id"] == first_query_id],
            visual_path,
            f"Holidays global retrieval: {first_query_id}",
        )
        result_files["visual_examples"] = visual_path

    manifest_path = write_run_manifest(
        args.output_dir / "global_run_manifest.json",
        script_path=Path(__file__),
        method="global_resnet50_holidays_indexed_queries",
        configuration={
            "save_topk": args.save_topk,
            "ranking_depth": "all_eligible_images",
            "metric_ks": [5, 10],
            "threshold": None,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_queries": args.max_queries,
            "query_embedding_source": "descriptor_from_ex013_index",
        },
        inputs={
            "queries_csv": args.queries_csv,
            "relevance_csv": args.relevance_csv,
            "global_embeddings": args.index_dir / "global_embeddings.npy",
            "global_metadata": args.index_dir / "global_metadata.csv",
            "global_index_manifest": args.index_dir / "global_index_manifest.json",
        },
        result_files=result_files,
        extra={
            "failed_queries": 0,
            "query_embeddings_reused": int(len(queries)),
            "eligible_gallery_images": int(len(metadata)),
        },
    )

    print(f"[OK] Global metrics: {metrics_path}")
    print(f"[OK] Run manifest: {manifest_path}")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
