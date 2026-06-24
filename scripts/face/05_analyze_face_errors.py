from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import OUTPUTS_DIR
from retrieval_common import relevance_from_csv, resolve_stored_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze face retrieval errors from Top-K outputs.")
    ap.add_argument(
        "--results-csv",
        type=Path,
        default=OUTPUTS_DIR / "reports" / "gallagher_face" / "face_topk_results.csv",
    )
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=OUTPUTS_DIR / "reports" / "gallagher_face" / "face_metrics_per_query.csv",
    )
    ap.add_argument("--queries-csv", type=Path, required=True)
    ap.add_argument("--relevance-csv", type=Path, required=True)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "reports" / "gallagher_face" / "error_analysis",
    )
    ap.add_argument("--worst-n", type=int, default=5)
    ap.add_argument("--visual-topk", type=int, default=10)
    return ap.parse_args()


def label_results(results: pd.DataFrame, relevance: dict[str, set[str]]) -> pd.DataFrame:
    out = results.copy()
    out["relevant"] = [
        int(str(row.image_id) in relevance.get(str(row.query_id), set()))
        for row in out.itertuples(index=False)
    ]
    out["result_type"] = np.where(out["relevant"] == 1, "true_positive", "false_positive")
    return out


def make_query_summary(labeled: pd.DataFrame, metrics: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    result_summary = (
        labeled.groupby("query_id")
        .agg(
            returned=("image_id", "count"),
            relevant_returned=("relevant", "sum"),
            max_score=("score", "max"),
            min_score=("score", "min"),
            first_false_positive_rank=(
                "rank",
                lambda s: int(s[labeled.loc[s.index, "relevant"] == 0].min())
                if (labeled.loc[s.index, "relevant"] == 0).any()
                else 0,
            ),
        )
        .reset_index()
    )
    summary = metrics.merge(result_summary, on="query_id", how="left")
    keep_cols = [
        "query_id",
        "target_label",
        "source_image_name",
        "source_face_index",
        "query_path",
    ]
    return summary.merge(queries[keep_cols], on="query_id", how="left")


def save_grid(rows: pd.DataFrame, out_path: Path, title: str, max_images: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = rows.head(max_images)
    if subset.empty:
        return
    cols = min(5, len(subset))
    fig_rows = int(np.ceil(len(subset) / cols))
    fig, axes = plt.subplots(fig_rows, cols, figsize=(4 * cols, 3.6 * fig_rows))
    axes_arr = np.array(axes).reshape(-1)
    for ax in axes_arr:
        ax.axis("off")
    for ax, row in zip(axes_arr, subset.itertuples(index=False)):
        img_path = resolve_stored_path(str(row.image_path))
        img = cv2.imread(str(img_path))
        if img is None:
            ax.set_title("missing")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        color = "green" if int(row.relevant) else "red"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(4)
        label = "TP" if int(row.relevant) else "FP"
        ax.set_title(f"{label} r{int(row.rank)} | {float(row.score):.3f}")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(
    out_path: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    false_positives: pd.DataFrame,
) -> None:
    overall = metrics.mean(numeric_only=True)
    worst = summary.sort_values(["average_precision", "precision_at_10"], ascending=True).head(5)
    best = summary.sort_values(["average_precision", "precision_at_10"], ascending=False).head(5)
    lines = [
        "# Gallagher face retrieval error analysis",
        "",
        "## Overall",
        "",
        f"- queries: {len(summary)}",
        f"- Precision@5: {overall.get('precision_at_5', 0):.4f}",
        f"- Precision@10: {overall.get('precision_at_10', 0):.4f}",
        f"- Recall@10: {overall.get('recall_at_10', 0):.4f}",
        f"- mAP: {overall.get('average_precision', 0):.4f}",
        f"- mean query time ms: {overall.get('query_time_ms', 0):.2f}",
        f"- false positives in Top-K: {len(false_positives)}",
        "",
        "## Hardest queries",
        "",
    ]
    for row in worst.itertuples(index=False):
        lines.append(
            f"- {row.query_id}: AP={row.average_precision:.4f}, "
            f"P@10={row.precision_at_10:.4f}, FP@10={row.false_positives_at_10}, "
            f"relevant={row.num_relevant}, source={row.source_image_name}"
        )
    lines.extend(["", "## Best queries", ""])
    for row in best.itertuples(index=False):
        lines.append(
            f"- {row.query_id}: AP={row.average_precision:.4f}, "
            f"P@10={row.precision_at_10:.4f}, Recall@10={row.recall_at_10:.4f}, "
            f"source={row.source_image_name}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Very large identities have low Recall@10 because Top-10 cannot cover hundreds of relevant images.",
            "- False positives usually happen between visually similar faces in the same photo collection.",
            "- For publication, Precision@K and mAP are more informative than Recall@10 when identities have very different numbers of relevant photos.",
            "- The source image is excluded through `source_image_id`, avoiding trivial self-matches from cropped queries.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results_csv)
    metrics = pd.read_csv(args.metrics_csv)
    queries = pd.read_csv(args.queries_csv)
    relevance = relevance_from_csv(args.relevance_csv)

    labeled = label_results(results, relevance)
    summary = make_query_summary(labeled, metrics, queries)
    false_positives = labeled[labeled["relevant"] == 0].copy()
    true_positives = labeled[labeled["relevant"] == 1].copy()
    hardest = summary.sort_values(["average_precision", "precision_at_10"], ascending=True)

    labeled.to_csv(args.output_dir / "topk_labeled.csv", index=False)
    summary.to_csv(args.output_dir / "query_error_summary.csv", index=False)
    false_positives.to_csv(args.output_dir / "false_positives.csv", index=False)
    true_positives.to_csv(args.output_dir / "true_positives.csv", index=False)
    hardest.to_csv(args.output_dir / "hardest_queries.csv", index=False)

    for query_id in hardest["query_id"].head(args.worst_n):
        rows = labeled[labeled["query_id"] == query_id]
        save_grid(
            rows,
            args.output_dir / "visual" / f"{query_id}_topk.png",
            f"{query_id} Top-K: green=TP, red=FP",
            args.visual_topk,
        )

    write_report(args.output_dir / "analysis_report.md", metrics, summary, false_positives)

    print(f"[OK] analysis: {args.output_dir}")
    print(summary.sort_values(["average_precision", "precision_at_10"]).head(args.worst_n)[
        [
            "query_id",
            "average_precision",
            "precision_at_10",
            "recall_at_10",
            "false_positives_at_10",
            "num_relevant",
        ]
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
