from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from face_lib import build_face_app, load_face_index, query_embedding_from_image
from fusion_lib import search_fusion
from global_lib import (
    build_resnet50_feature_extractor,
    extract_global_embedding,
    load_global_index,
)
from project_paths import OUTPUTS_DIR
from retrieval_common import (
    aggregate_metrics,
    now_ms,
    relevance_from_csv,
    save_visual_grid,
    summarize_rankings,
)


def parse_weight(value: str) -> tuple[float, float]:
    if "/" in value:
        a, b = value.split("/", 1)
    elif "," in value:
        a, b = value.split(",", 1)
    else:
        raise argparse.ArgumentTypeError("Use FACE/GLOBAL, e.g. 0.7/0.3")
    return float(a), float(b)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate face/global fusion retrieval.")
    ap.add_argument("--queries-csv", type=Path, required=True)
    ap.add_argument("--relevance-csv", type=Path, required=True)
    ap.add_argument("--face-index-dir", type=Path, default=OUTPUTS_DIR / "face_index")
    ap.add_argument("--global-index-dir", type=Path, default=OUTPUTS_DIR / "global_index")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "reports")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=-1.0)
    ap.add_argument("--weights-list", nargs="+", type=parse_weight, default=[(0.9, 0.1), (0.7, 0.3), (0.5, 0.5)])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--det-size", type=int, default=640)
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

    relevance = relevance_from_csv(args.relevance_csv)
    face_E, face_meta = load_face_index(args.face_index_dir)
    global_E, global_meta = load_global_index(args.global_index_dir)
    face_app = build_face_app(device=args.device, det_size=args.det_size)
    extractor, transform, device = build_resnet50_feature_extractor(args.device, args.weights)

    all_metric_rows = []
    all_results = []
    failures = []
    first_visual_saved = False

    for face_weight, global_weight in args.weights_list:
        method = f"fusion_{face_weight:.1f}_{global_weight:.1f}".replace(".", "p")
        rankings: dict[str, list[str]] = {}
        query_times: dict[str, float] = {}
        weight_results = []
        for row in queries.itertuples(index=False):
            query_id = str(getattr(row, "query_id"))
            query_path = Path(str(getattr(row, "query_path")))
            try:
                t0 = now_ms()
                face_q, query_bbox = query_embedding_from_image(query_path, face_app)
                global_q = extract_global_embedding(query_path, extractor, transform, device)
                res = search_fusion(
                    query_path,
                    face_q,
                    global_q,
                    face_E,
                    face_meta,
                    global_E,
                    global_meta,
                    topk=args.topk,
                    face_weight=face_weight,
                    global_weight=global_weight,
                    threshold=args.threshold,
                )
                elapsed = now_ms() - t0
            except Exception as exc:
                failures.append({"method": method, "query_id": query_id, "error": repr(exc)})
                rankings[query_id] = []
                query_times[query_id] = 0.0
                continue
            res.insert(0, "method", method)
            res.insert(1, "query_id", query_id)
            res.insert(2, "query_path", str(query_path))
            res.insert(3, "query_bbox", query_bbox)
            res["query_time_ms"] = elapsed
            rankings[query_id] = res["image_id"].astype(str).tolist()
            query_times[query_id] = elapsed
            weight_results.append(res)

        per_query = summarize_rankings(rankings, relevance, ks=(5, 10))
        if not per_query.empty:
            per_query["query_time_ms"] = per_query["query_id"].map(query_times).fillna(0.0)
            per_query["method"] = method
        metrics = aggregate_metrics(per_query, method)
        metrics["face_weight"] = face_weight
        metrics["global_weight"] = global_weight
        all_metric_rows.append(metrics)
        if weight_results:
            wr = pd.concat(weight_results, ignore_index=True)
            all_results.append(wr)
            if args.save_visual_examples and not first_visual_saved:
                save_visual_grid(
                    wr,
                    args.output_dir / "visual_examples" / "fusion_examples.png",
                    "Fusion retrieval examples",
                )
                first_visual_saved = True

    metrics_out = pd.concat(all_metric_rows, ignore_index=True) if all_metric_rows else pd.DataFrame()
    results_out = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    metrics_out.to_csv(args.output_dir / "fusion_metrics.csv", index=False)
    results_out.to_csv(args.output_dir / "fusion_topk_results.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.output_dir / "fusion_failures.csv", index=False)

    print("[OK] Fusion metrics:", args.output_dir / "fusion_metrics.csv")
    print(metrics_out)
    if failures:
        print(f"[WARN] Query failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
