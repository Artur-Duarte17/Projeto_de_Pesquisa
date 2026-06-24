from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from global_lib import (
    build_resnet50_feature_extractor,
    describe_torch_device,
    extract_global_embedding,
    load_global_index,
    search_global_index,
)
from project_paths import OUTPUTS_DIR
from retrieval_common import (
    aggregate_metrics,
    exclude_query_images_from_relevance,
    now_ms,
    relevance_from_csv,
    relevance_from_labels,
    save_visual_grid,
    summarize_rankings,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate whole-image retrieval.")
    ap.add_argument("--queries-csv", type=Path, required=True)
    ap.add_argument("--relevance-csv", type=Path, default=None)
    ap.add_argument("--index-dir", type=Path, default=OUTPUTS_DIR / "global_index")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "reports")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=-1.0, help="Reserved for CLI compatibility.")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--save-visual-examples", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    queries = pd.read_csv(args.queries_csv)
    if args.max_queries is not None:
        queries = queries.head(args.max_queries)
    for col in ["query_id", "query_path"]:
        if col not in queries.columns:
            raise ValueError(f"Missing column in queries CSV: {col}")

    embeddings, metadata = load_global_index(args.index_dir)
    relevance = (
        relevance_from_csv(args.relevance_csv)
        if args.relevance_csv
        else relevance_from_labels(queries, metadata, "label_or_group")
    )
    if not args.relevance_csv:
        relevance = exclude_query_images_from_relevance(queries, metadata, relevance)
    extractor, transform, device = build_resnet50_feature_extractor(args.device, args.weights)
    print(f"[INFO] Torch device: {describe_torch_device(args.device, device)}")

    all_results = []
    rankings: dict[str, list[str]] = {}
    query_times: dict[str, float] = {}
    failures = []

    for row in queries.itertuples(index=False):
        query_id = str(getattr(row, "query_id"))
        query_path = Path(str(getattr(row, "query_path")))
        try:
            t0 = now_ms()
            q = extract_global_embedding(query_path, extractor, transform, device)
            res = search_global_index(q, embeddings, metadata, topk=args.topk, query_path=query_path)
            elapsed = now_ms() - t0
        except Exception as exc:
            failures.append({"query_id": query_id, "error": repr(exc)})
            rankings[query_id] = []
            query_times[query_id] = 0.0
            continue
        res.insert(0, "query_id", query_id)
        res.insert(1, "query_path", str(query_path))
        res["query_time_ms"] = elapsed
        all_results.append(res)
        rankings[query_id] = res["image_id"].astype(str).tolist()
        query_times[query_id] = elapsed

    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    per_query = summarize_rankings(rankings, relevance, ks=(5, 10))
    if not per_query.empty:
        per_query["query_time_ms"] = per_query["query_id"].map(query_times).fillna(0.0)
    metrics = aggregate_metrics(per_query, "global_resnet50")

    results.to_csv(args.output_dir / "global_topk_results.csv", index=False)
    per_query.to_csv(args.output_dir / "global_metrics_per_query.csv", index=False)
    metrics.to_csv(args.output_dir / "global_metrics.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.output_dir / "global_failures.csv", index=False)
    if args.save_visual_examples and not results.empty:
        save_visual_grid(
            results,
            args.output_dir / "visual_examples" / "global_examples.png",
            "Global retrieval examples",
        )

    print("[OK] Global metrics:", args.output_dir / "global_metrics.csv")
    print(metrics)
    if failures:
        print(f"[WARN] Query failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
