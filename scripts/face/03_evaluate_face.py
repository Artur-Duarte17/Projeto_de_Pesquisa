from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from face_lib import build_face_app, load_face_index, query_embedding_from_image, search_face_index
from project_paths import OUTPUTS_DIR
from retrieval_common import (
    aggregate_metrics,
    apply_relevance_exclusions,
    build_query_exclusions,
    now_ms,
    relevance_from_csv,
    relevance_from_labels,
    save_visual_grid,
    select_results_for_storage,
    summarize_rankings,
)
from run_manifest import write_run_manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate face retrieval with Precision/Recall metrics.")
    ap.add_argument("--queries-csv", type=Path, required=True)
    ap.add_argument("--relevance-csv", type=Path, default=None)
    ap.add_argument("--index-dir", type=Path, default=OUTPUTS_DIR / "face_index")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "reports")
    ap.add_argument("--save-topk", type=int, default=10)
    ap.add_argument("--topk", type=int, default=None, help="Legacy alias for --save-topk.")
    ap.add_argument("--threshold", type=float, default=-1.0)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--save-visual-examples", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    save_topk = args.topk if args.topk is not None else args.save_topk
    if save_topk <= 0:
        raise ValueError("save-topk must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    queries = pd.read_csv(args.queries_csv)
    if args.max_queries is not None:
        queries = queries.head(args.max_queries)
    for col in ["query_id", "query_path"]:
        if col not in queries.columns:
            raise ValueError(f"Missing column in queries CSV: {col}")

    embeddings, metadata = load_face_index(args.index_dir)
    relevance = (
        relevance_from_csv(args.relevance_csv)
        if args.relevance_csv
        else relevance_from_labels(queries, metadata, "identity")
    )
    exclusions = build_query_exclusions(queries, metadata)
    relevance = apply_relevance_exclusions(relevance, exclusions)
    app = build_face_app(device=args.device, det_size=args.det_size)

    all_results = []
    rankings: dict[str, list[str]] = {}
    query_times: dict[str, float] = {}
    failures = []

    for row in queries.itertuples(index=False):
        query_id = str(getattr(row, "query_id"))
        query_path = Path(str(getattr(row, "query_path")))
        try:
            t0 = now_ms()
            q, query_bbox = query_embedding_from_image(query_path, app)
            res = search_face_index(
                q,
                embeddings,
                metadata,
                topk=None,
                threshold=args.threshold,
                query_path=query_path,
                exclude_image_ids=exclusions.get(query_id, set()),
            )
            elapsed = now_ms() - t0
        except Exception as exc:
            failures.append({"query_id": query_id, "error": repr(exc)})
            rankings[query_id] = []
            query_times[query_id] = 0.0
            continue
        res.insert(0, "query_id", query_id)
        res.insert(1, "query_path", str(query_path))
        res.insert(2, "query_bbox", query_bbox)
        res["query_time_ms"] = elapsed
        all_results.append(select_results_for_storage(res, save_topk))
        rankings[query_id] = res["image_id"].astype(str).tolist()
        query_times[query_id] = elapsed

    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    per_query = summarize_rankings(rankings, relevance, ks=(5, 10))
    if not per_query.empty:
        per_query["query_time_ms"] = per_query["query_id"].map(query_times).fillna(0.0)
    metrics = aggregate_metrics(per_query, "face")

    results_path = args.output_dir / "face_topk_results.csv"
    per_query_path = args.output_dir / "face_metrics_per_query.csv"
    metrics_path = args.output_dir / "face_metrics.csv"
    results.to_csv(results_path, index=False)
    per_query.to_csv(per_query_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    failures_path = args.output_dir / "face_failures.csv"
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)
    elif failures_path.exists():
        failures_path.unlink()
    if args.save_visual_examples and not results.empty:
        save_visual_grid(
            results,
            args.output_dir / "visual_examples" / "face_examples.png",
            "Face retrieval examples",
        )

    result_files = {
        "saved_topk": results_path,
        "per_query_metrics": per_query_path,
        "aggregate_metrics": metrics_path,
    }
    if failures_path.exists():
        result_files["failures"] = failures_path
    manifest_path = write_run_manifest(
        args.output_dir / "face_run_manifest.json",
        script_path=Path(__file__),
        method="face",
        configuration={
            "save_topk": save_topk,
            "ranking_depth": "all_eligible_images",
            "metric_ks": [5, 10],
            "threshold": args.threshold,
            "device": args.device,
            "det_size": args.det_size,
            "max_queries": args.max_queries,
        },
        inputs={
            "queries_csv": args.queries_csv,
            "relevance_csv": args.relevance_csv,
            "face_embeddings": args.index_dir / "face_embeddings.npy",
            "face_metadata": args.index_dir / "face_metadata.csv",
        },
        result_files=result_files,
        extra={"failed_queries": len(failures)},
    )

    print("[OK] Face metrics:", metrics_path)
    print("[OK] Run manifest:", manifest_path)
    print(metrics)
    if failures:
        print(f"[WARN] Query failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
