from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import DATA_DIR, OUTPUTS_DIR
from run_manifest import write_run_manifest


METHODS = (
    "face_only",
    "global_context_only",
    "fusion_0p9_0p1",
    "fusion_0p7_0p3",
    "fusion_0p5_0p5",
)
METRICS = (
    "average_precision",
    "precision_at_5",
    "precision_at_10",
    "recall_at_5",
    "recall_at_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and summarize the ten-query Agrishow robustness evaluation."
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-026_agrishow_robustness_evaluation",
    )
    parser.add_argument(
        "--queries-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_robustness_queries.csv",
    )
    parser.add_argument(
        "--relevance-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_robustness_relevance.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-026_agrishow_robustness_analysis",
    )
    parser.add_argument("--delta-tolerance", type=float, default=1e-12)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def classify_delta(value: float, tolerance: float) -> str:
    if value > tolerance:
        return "improved"
    if value < -tolerance:
        return "degraded"
    return "unchanged"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    queries = pd.read_csv(args.queries_csv, dtype=str).fillna("")
    relevance = pd.read_csv(args.relevance_csv, dtype=str).fillna("")
    per_query_path = args.evaluation_dir / "fusion_metrics_per_query.csv"
    aggregate_path = args.evaluation_dir / "fusion_metrics.csv"
    topk_path = args.evaluation_dir / "fusion_topk_results.csv"
    run_manifest_path = args.evaluation_dir / "fusion_run_manifest.json"
    per_query = pd.read_csv(per_query_path)
    aggregate = pd.read_csv(aggregate_path)
    topk = pd.read_csv(topk_path, dtype={"query_id": str, "image_id": str})
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    require(len(queries) == 10 and queries["query_id"].is_unique, "Expected ten unique queries")
    require(queries["source_image_id"].nunique() == 10, "Query sources must be unique")
    require(len(relevance) == 1240, "Expected 1,240 relevance rows")
    require(
        relevance[["query_id", "image_id"]].drop_duplicates().shape[0] == 1240,
        "Relevance query/image pairs are not unique",
    )
    grouped_relevance = relevance.groupby("query_id")
    require(grouped_relevance.size().eq(124).all(), "Each query must have 124 labels")
    require(
        grouped_relevance["relevant"].apply(lambda values: values.astype(int).sum()).eq(91).all(),
        "Each query must have 91 relevant labels before source exclusion",
    )

    require(len(per_query) == 50, "Expected 50 per-query metric rows")
    require(set(per_query["method"]) == set(METHODS), "Unexpected evaluation methods")
    require(per_query.groupby("method").size().eq(10).all(), "Each method must have ten queries")
    require(per_query["num_relevant"].astype(int).eq(90).all(), "Eligible relevance must be 90")
    require(per_query["ranking_size"].astype(int).eq(123).all(), "Ranking depth must be 123")
    require(len(topk) == 500, "Expected 500 saved Top-10 rows")
    require(
        topk.groupby(["method", "query_id"]).size().eq(10).all(),
        "Every method/query pair must contain ten saved results",
    )
    require(
        not topk.duplicated(["method", "query_id", "rank"]).any(),
        "Duplicate method/query/rank rows found",
    )
    source_by_query = queries.set_index("query_id")["source_image_id"].astype(str).to_dict()
    source_leaks = topk.apply(
        lambda row: str(row["image_id"]) == source_by_query[str(row["query_id"])], axis=1
    )
    require(not source_leaks.any(), "A source photograph leaked into Top-10")
    for column in (*METRICS, "face_query_extraction_ms", "retrieval_backend_ms"):
        require(np.isfinite(per_query[column].astype(float)).all(), f"Non-finite values in {column}")

    for method in METHODS:
        current = per_query[per_query["method"] == method]
        reported = aggregate[aggregate["method"] == method]
        require(len(reported) == 1, f"Missing aggregate row for {method}")
        for metric in METRICS:
            expected_column = "mAP" if metric == "average_precision" else metric
            require(
                np.isclose(
                    current[metric].astype(float).mean(),
                    float(reported.iloc[0][expected_column]),
                    atol=1e-12,
                ),
                f"Aggregate {expected_column} differs for {method}",
            )
    require(run_manifest["git"]["dirty"] is False, "EX-026 evaluation tree was dirty")
    require(run_manifest["extra"]["failed_queries"] == 0, "EX-026 recorded failed queries")

    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        current = per_query[per_query["method"] == method]
        for metric in METRICS:
            values = current[metric].astype(float)
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "median": float(values.median()),
                    "maximum": float(values.max()),
                    "minimum_query_id": str(current.loc[values.idxmin(), "query_id"]),
                    "maximum_query_id": str(current.loc[values.idxmax(), "query_id"]),
                    "num_queries": len(values),
                }
            )
    summary = pd.DataFrame(summary_rows)

    wide = per_query.pivot(index="query_id", columns="method", values="average_precision")
    face = wide["face_only"]
    comparison_rows: list[dict[str, object]] = []
    for method in METHODS[1:]:
        deltas = wide[method] - face
        classifications = deltas.apply(lambda value: classify_delta(float(value), args.delta_tolerance))
        comparison_rows.append(
            {
                "method": method,
                "mean_ap_delta_vs_face": float(deltas.mean()),
                "minimum_ap_delta_vs_face": float(deltas.min()),
                "maximum_ap_delta_vs_face": float(deltas.max()),
                "queries_improved": int(classifications.eq("improved").sum()),
                "queries_unchanged": int(classifications.eq("unchanged").sum()),
                "queries_degraded": int(classifications.eq("degraded").sum()),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)

    difficulty = queries[["query_id", "source_image_id"]].copy()
    for method in METHODS:
        method_rows = per_query[per_query["method"] == method].set_index("query_id")
        difficulty[f"{method}_ap"] = difficulty["query_id"].map(method_rows["average_precision"])
    face_rows = per_query[per_query["method"] == "face_only"].set_index("query_id")
    difficulty["face_precision_at_5"] = difficulty["query_id"].map(face_rows["precision_at_5"])
    difficulty["face_precision_at_10"] = difficulty["query_id"].map(face_rows["precision_at_10"])
    difficulty = difficulty.sort_values("face_only_ap").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "robustness_metric_summary.csv"
    comparison_path = output_dir / "method_comparison_to_face.csv"
    difficulty_path = output_dir / "query_difficulty.csv"
    summary.to_csv(summary_path, index=False)
    comparisons.to_csv(comparison_path, index=False)
    difficulty.to_csv(difficulty_path, index=False)
    manifest_path = write_run_manifest(
        output_dir / "robustness_analysis_manifest.json",
        script_path=Path(__file__),
        method="agrishow_2022_ten_query_robustness_audit",
        configuration={
            "methods": list(METHODS),
            "metrics": list(METRICS),
            "delta_tolerance": args.delta_tolerance,
            "summary_statistics": ["mean", "sample_std", "minimum", "median", "maximum"],
        },
        inputs={
            "queries_csv": args.queries_csv,
            "relevance_csv": args.relevance_csv,
            "per_query_metrics": per_query_path,
            "aggregate_metrics": aggregate_path,
            "saved_topk": topk_path,
            "evaluation_manifest": run_manifest_path,
        },
        result_files={
            "metric_summary": summary_path,
            "method_comparison_to_face": comparison_path,
            "query_difficulty": difficulty_path,
        },
        extra={
            "queries": 10,
            "methods": 5,
            "saved_topk_rows": len(topk),
            "source_leaks": 0,
            "failed_queries": 0,
            "worst_face_query_id": str(difficulty.iloc[0]["query_id"]),
            "worst_face_average_precision": float(difficulty.iloc[0]["face_only_ap"]),
        },
    )
    print("EX-026: ROBUSTNESS AUDIT APPROVED")
    print(summary[summary["metric"] == "average_precision"].to_string(index=False))
    print(comparisons.to_string(index=False))
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
