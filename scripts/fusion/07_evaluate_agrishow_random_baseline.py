from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import DATA_DIR, OUTPUTS_DIR
from random_baseline_lib import expected_random_ap, probability_perfect_topk
from retrieval_common import sha256_file
from run_manifest import write_run_manifest


DEFAULT_SEED = 12031681346522691727
METRICS = ("average_precision", "precision_at_5", "precision_at_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the frozen Agrishow results with reproducible random rankings."
        )
    )
    parser.add_argument(
        "--relevance-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_relevance.csv",
    )
    parser.add_argument(
        "--queries-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_queries.csv",
    )
    parser.add_argument(
        "--observed-metrics-csv",
        type=Path,
        default=OUTPUTS_DIR
        / "experiments"
        / "ex-023_agrishow_paired_fusion"
        / "fusion_metrics_per_query.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-025_agrishow_random_baseline",
    )
    parser.add_argument("--iterations", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def simulate_random_rankings(
    *, total: int, relevant: int, iterations: int, batch_size: int, seed: int
) -> dict[str, np.ndarray]:
    if iterations <= 0 or batch_size <= 0:
        raise ValueError("iterations and batch-size must be positive")
    rng = np.random.default_rng(seed)
    outputs = {
        "average_precision": np.empty(iterations, dtype=np.float64),
        "precision_at_5": np.empty(iterations, dtype=np.float64),
        "precision_at_10": np.empty(iterations, dtype=np.float64),
    }
    base = np.zeros(total, dtype=np.int8)
    base[:relevant] = 1
    ranks = np.arange(1, total + 1, dtype=np.float64)

    offset = 0
    while offset < iterations:
        count = min(batch_size, iterations - offset)
        order = np.argsort(rng.random((count, total)), axis=1)
        hits = base[order]
        cumulative_hits = np.cumsum(hits, axis=1)
        precision_at_rank = cumulative_hits / ranks[None, :]
        outputs["average_precision"][offset : offset + count] = (
            (precision_at_rank * hits).sum(axis=1) / relevant
        )
        outputs["precision_at_5"][offset : offset + count] = hits[:, :5].mean(axis=1)
        outputs["precision_at_10"][offset : offset + count] = hits[:, :10].mean(axis=1)
        offset += count
    return outputs


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    queries = pd.read_csv(args.queries_csv, dtype=str).fillna("")
    require(len(queries) == 1, f"Expected one frozen query, found {len(queries)}")
    query = queries.iloc[0]
    query_id = str(query["query_id"])
    source_image_id = str(query["source_image_id"])

    relevance = pd.read_csv(args.relevance_csv, dtype={"query_id": str, "image_id": str})
    relevance = relevance[relevance["query_id"].astype(str) == query_id].copy()
    require(len(relevance) == 124, f"Expected 124 relevance rows, found {len(relevance)}")
    require(relevance["image_id"].astype(str).nunique() == 124, "Relevance IDs are not unique")
    source_rows = relevance[relevance["image_id"].astype(str) == source_image_id]
    require(len(source_rows) == 1, "Frozen source is absent from relevance table")
    require(int(source_rows.iloc[0]["relevant"]) == 1, "Frozen source must be relevant")

    eligible = relevance[relevance["image_id"].astype(str) != source_image_id]
    total = len(eligible)
    relevant = int(eligible["relevant"].astype(int).sum())
    require(total == 123, f"Expected 123 eligible candidates, found {total}")
    require(relevant == 90, f"Expected 90 eligible relevant images, found {relevant}")

    observed = pd.read_csv(args.observed_metrics_csv)
    require(set(METRICS).issubset(observed.columns), "EX-023 metrics are incomplete")
    require(observed["method"].nunique() == 5, "Expected five frozen EX-023 methods")
    require(observed["query_id"].astype(str).eq(query_id).all(), "EX-023 query differs")
    require(observed["num_relevant"].astype(int).eq(relevant).all(), "EX-023 relevance differs")
    require(observed["ranking_size"].astype(int).eq(total).all(), "EX-023 ranking size differs")

    simulated = simulate_random_rankings(
        total=total,
        relevant=relevant,
        iterations=args.iterations,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    exact_expectations = {
        "average_precision": expected_random_ap(total, relevant),
        "precision_at_5": relevant / total,
        "precision_at_10": relevant / total,
    }
    summary_rows: list[dict[str, object]] = []
    for metric in METRICS:
        values = simulated[metric]
        summary_rows.append(
            {
                "metric": metric,
                "exact_random_expectation": exact_expectations[metric],
                "simulation_mean": float(values.mean()),
                "simulation_std": float(values.std(ddof=1)),
                "simulation_min": float(values.min()),
                "simulation_q025": float(np.quantile(values, 0.025)),
                "simulation_median": float(np.quantile(values, 0.5)),
                "simulation_q975": float(np.quantile(values, 0.975)),
                "simulation_max": float(values.max()),
                "iterations": args.iterations,
                "seed": args.seed,
            }
        )
    distribution_summary = pd.DataFrame(summary_rows)

    comparison_rows: list[dict[str, object]] = []
    for row in observed.itertuples(index=False):
        for metric in METRICS:
            value = float(getattr(row, metric))
            random_values = simulated[metric]
            exceedances = int(np.count_nonzero(random_values >= value - 1e-15))
            empirical_tail_p = (exceedances + 1) / (args.iterations + 1)
            exact_tail_p = ""
            if metric in {"precision_at_5", "precision_at_10"} and np.isclose(value, 1.0):
                k = 5 if metric.endswith("_5") else 10
                exact_tail_p = probability_perfect_topk(total, relevant, k)
            comparison_rows.append(
                {
                    "method": str(row.method),
                    "metric": metric,
                    "observed_value": value,
                    "random_expectation": exact_expectations[metric],
                    "delta_vs_random_expectation": value - exact_expectations[metric],
                    "random_percentile": 100.0 * float(np.mean(random_values <= value + 1e-15)),
                    "simulation_exceedances": exceedances,
                    "empirical_one_sided_p_smoothed": empirical_tail_p,
                    "exact_one_sided_p_if_available": exact_tail_p,
                }
            )
    comparison = pd.DataFrame(comparison_rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "random_distribution_summary.csv"
    comparison_path = output_dir / "observed_vs_random.csv"
    distribution_summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    manifest_path = write_run_manifest(
        output_dir / "random_baseline_manifest.json",
        script_path=Path(__file__),
        method="agrishow_2022_uniform_random_ranking_baseline",
        configuration={
            "seed": args.seed,
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "ranking_model": "uniform_random_permutation_without_replacement",
            "empirical_p_correction": "(exceedances + 1) / (iterations + 1)",
            "quantiles": [0.025, 0.5, 0.975],
        },
        inputs={
            "queries_csv": args.queries_csv,
            "relevance_csv": args.relevance_csv,
            "observed_metrics_csv": args.observed_metrics_csv,
        },
        result_files={
            "random_distribution_summary": summary_path,
            "observed_vs_random": comparison_path,
        },
        extra={
            "query_id": query_id,
            "source_image_id": source_image_id,
            "source_excluded": True,
            "eligible_candidates": total,
            "eligible_relevant": relevant,
            "relevant_prevalence": relevant / total,
            "input_hashes": {
                "queries_csv": sha256_file(args.queries_csv),
                "relevance_csv": sha256_file(args.relevance_csv),
                "observed_metrics_csv": sha256_file(args.observed_metrics_csv),
            },
        },
    )

    print("EX-025: RANDOM BASELINE APPROVED")
    print(f"eligible={total} relevant={relevant} prevalence={relevant / total:.6f}")
    print(distribution_summary.to_string(index=False))
    print(f"comparison={comparison_path}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
