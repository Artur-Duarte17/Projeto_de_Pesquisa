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

from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval.adapters.files import relevance_from_csv, resolve_stored_path
from retrieval.adapters.manifest import write_run_manifest
from retrieval.domain.fusion import classify_metric_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired Gallagher fusion outcomes.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=OUTPUTS_DIR
        / "final"
        / "gallagher"
        / "evaluation"
        / "fusion_topk_results.csv",
    )
    parser.add_argument(
        "--per-query-csv",
        type=Path,
        default=OUTPUTS_DIR
        / "final"
        / "gallagher"
        / "evaluation"
        / "fusion_metrics_per_query.csv",
    )
    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        default=OUTPUTS_DIR
        / "final"
        / "gallagher"
        / "evaluation"
        / "fusion_metrics.csv",
    )
    parser.add_argument(
        "--queries-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "gallagher_final" / "gallagher_fusion_queries.csv",
    )
    parser.add_argument(
        "--relevance-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "gallagher_final" / "gallagher_relevance.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "final" / "gallagher" / "error_analysis",
    )
    parser.add_argument("--primary-fusion", default="fusion_0p9_0p1")
    parser.add_argument("--visual-topk", type=int, default=5)
    return parser.parse_args()


def label_topk(results: pd.DataFrame, relevance: dict[str, set[str]]) -> pd.DataFrame:
    labeled = results.copy()
    labeled["relevant"] = [
        int(str(row.image_id) in relevance.get(str(row.query_id), set()))
        for row in labeled.itertuples(index=False)
    ]
    labeled["result_type"] = np.where(
        labeled["relevant"].astype(int) == 1,
        "true_positive",
        "false_positive",
    )
    return labeled


def build_deltas(per_query: pd.DataFrame) -> pd.DataFrame:
    face = per_query[per_query["method"] == "face_only"].set_index("query_id")
    if face.empty:
        raise ValueError("Missing face_only baseline")
    rows: list[dict[str, object]] = []
    for method in sorted(set(per_query["method"]) - {"face_only"}):
        current = per_query[per_query["method"] == method].set_index("query_id")
        if set(current.index) != set(face.index):
            raise ValueError(f"Query coverage differs for method: {method}")
        for query_id in face.index:
            ap_face = float(face.loc[query_id, "average_precision"])
            ap_method = float(current.loc[query_id, "average_precision"])
            delta_ap = ap_method - ap_face
            rows.append(
                {
                    "method": method,
                    "query_id": query_id,
                    "ap_face_only": ap_face,
                    "ap_method": ap_method,
                    "delta_ap": delta_ap,
                    "impact": classify_metric_delta(delta_ap),
                    "delta_precision_at_5": float(current.loc[query_id, "precision_at_5"])
                    - float(face.loc[query_id, "precision_at_5"]),
                    "delta_precision_at_10": float(current.loc[query_id, "precision_at_10"])
                    - float(face.loc[query_id, "precision_at_10"]),
                    "delta_recall_at_10": float(current.loc[query_id, "recall_at_10"])
                    - float(face.loc[query_id, "recall_at_10"]),
                }
            )
    return pd.DataFrame(rows)


def build_primary_summary(
    labeled: pd.DataFrame,
    deltas: pd.DataFrame,
    queries: pd.DataFrame,
    primary_method: str,
) -> pd.DataFrame:
    primary = deltas[deltas["method"] == primary_method].copy()
    if primary.empty:
        raise ValueError(f"Primary fusion method is absent: {primary_method}")

    rows: list[dict[str, object]] = []
    for delta in primary.itertuples(index=False):
        query_id = str(delta.query_id)
        face = labeled[
            (labeled["method"] == "face_only") & (labeled["query_id"] == query_id)
        ].sort_values("rank")
        fusion = labeled[
            (labeled["method"] == primary_method) & (labeled["query_id"] == query_id)
        ].sort_values("rank")
        face_ids = face["image_id"].astype(str).tolist()
        fusion_ids = fusion["image_id"].astype(str).tolist()
        query = queries[queries["query_id"] == query_id].iloc[0]
        rows.append(
            {
                **delta._asdict(),
                "target_label": str(query["target_label"]),
                "source_image_id": str(query["source_image_id"]),
                "source_image_name": str(query.get("source_image_name", "")),
                "face_true_positives_at_10": int(face["relevant"].sum()),
                "fusion_true_positives_at_10": int(fusion["relevant"].sum()),
                "top10_overlap": len(set(face_ids) & set(fusion_ids)),
                "face_first_false_positive_rank": int(
                    face.loc[face["relevant"] == 0, "rank"].min()
                )
                if (face["relevant"] == 0).any()
                else 0,
                "fusion_first_false_positive_rank": int(
                    fusion.loc[fusion["relevant"] == 0, "rank"].min()
                )
                if (fusion["relevant"] == 0).any()
                else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["delta_ap", "query_id"], ascending=[False, True])


def choose_cases(primary: pd.DataFrame) -> pd.DataFrame:
    improved = primary[primary["impact"] == "improved"]
    degraded = primary[primary["impact"] == "degraded"]
    unchanged = primary[primary["impact"] == "unchanged"]
    selections: list[dict[str, object]] = []
    if not improved.empty:
        row = improved.sort_values(["delta_ap", "query_id"], ascending=[False, True]).iloc[0]
        selections.append({"case": "largest_improvement", **row.to_dict()})
    if not unchanged.empty:
        row = unchanged.sort_values("query_id").iloc[0]
        selections.append({"case": "unchanged_example", **row.to_dict()})
    if not degraded.empty:
        row = degraded.sort_values(["delta_ap", "query_id"]).iloc[0]
        selections.append({"case": "largest_degradation", **row.to_dict()})
    return pd.DataFrame(selections)


def read_rgb(path_value: str) -> np.ndarray | None:
    image = cv2.imread(str(resolve_stored_path(path_value)))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_case(
    query: pd.Series,
    labeled: pd.DataFrame,
    primary_method: str,
    out_path: Path,
    visual_topk: int,
    case_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    query_id = str(query["query_id"])
    face = labeled[
        (labeled["method"] == "face_only") & (labeled["query_id"] == query_id)
    ].sort_values("rank").head(visual_topk)
    fusion = labeled[
        (labeled["method"] == primary_method) & (labeled["query_id"] == query_id)
    ].sort_values("rank").head(visual_topk)

    columns = max(visual_topk, 2)
    fig, axes = plt.subplots(3, columns, figsize=(3.5 * columns, 10.5))
    for axis in axes.reshape(-1):
        axis.axis("off")

    for column, (path, title) in enumerate(
        [
            (str(query["face_query_path"]), "Consulta: pessoa-alvo"),
            (str(query["global_query_path"]), "Consulta: fotografia completa"),
        ]
    ):
        image = read_rgb(path)
        if image is not None:
            axes[0, column].imshow(image)
        axes[0, column].set_title(title)

    for row_index, (rows, method_title) in enumerate(
        [(face, "Somente face"), (fusion, "Face + contexto 0,9/0,1")],
        start=1,
    ):
        for column, result in enumerate(rows.itertuples(index=False)):
            image = read_rgb(str(result.image_path))
            if image is not None:
                axes[row_index, column].imshow(image)
            color = "green" if int(result.relevant) else "red"
            for spine in axes[row_index, column].spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(4)
            label = "correta" if int(result.relevant) else "incorreta"
            axes[row_index, column].set_title(
                f"{method_title}\n#{int(result.rank)} {label} | {float(result.score):.3f}"
            )

    fig.suptitle(f"{case_label}: {query_id}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_aggregate(aggregate: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered_methods = [
        "face_only",
        "global_context_only",
        "fusion_0p9_0p1",
        "fusion_0p7_0p3",
        "fusion_0p5_0p5",
    ]
    table = aggregate.set_index("method").loc[ordered_methods]
    labels = ["Face", "Contexto", "0,9/0,1", "0,7/0,3", "0,5/0,5"]
    columns = ["precision_at_5", "precision_at_10", "recall_at_10", "mAP"]
    names = ["P@5", "P@10", "Recall@10", "mAP"]
    x = np.arange(len(labels))
    width = 0.19
    fig, axis = plt.subplots(figsize=(12, 6))
    for index, (column, name) in enumerate(zip(columns, names, strict=True)):
        axis.bar(x + (index - 1.5) * width, table[column], width, label=name)
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Métrica")
    axis.set_title("Comparação oficial — Gallagher pareado")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(
    out_path: Path,
    aggregate: pd.DataFrame,
    deltas: pd.DataFrame,
    primary: pd.DataFrame,
    cases: pd.DataFrame,
    primary_method: str,
) -> None:
    primary_counts = primary["impact"].value_counts().to_dict()
    face_map = float(aggregate.loc[aggregate["method"] == "face_only", "mAP"].iloc[0])
    fusion_map = float(aggregate.loc[aggregate["method"] == primary_method, "mAP"].iloc[0])
    lines = [
        "# Análise de erros da fusão pareada Gallagher",
        "",
        "## Conclusão principal",
        "",
        f"- mAP somente face: `{face_map:.6f}`",
        f"- mAP face/contexto 0,9/0,1: `{fusion_map:.6f}`",
        f"- diferença média: `{fusion_map - face_map:.6f}`",
        f"- consultas melhoradas: {primary_counts.get('improved', 0)}",
        f"- consultas inalteradas: {primary_counts.get('unchanged', 0)}",
        f"- consultas degradadas: {primary_counts.get('degraded', 0)}",
        "",
        "O contexto ajuda casos isolados, mas não melhora o resultado médio. Quanto maior o peso global, maior foi a degradação agregada.",
        "",
        "## Casos selecionados deterministicamente",
        "",
    ]
    for row in cases.itertuples(index=False):
        lines.append(
            f"- `{row.case}` — `{row.query_id}`: AP face `{row.ap_face_only:.6f}`, "
            f"AP fusão `{row.ap_method:.6f}`, diferença `{row.delta_ap:+.6f}`, "
            f"sobreposição Top-10 `{int(row.top10_overlap)}/10`."
        )
    lines.extend(["", "## Sensibilidade por método", ""])
    for method, group in deltas.groupby("method"):
        counts = group["impact"].value_counts().to_dict()
        lines.append(
            f"- `{method}`: melhora {counts.get('improved', 0)}, "
            f"empata {counts.get('unchanged', 0)}, piora {counts.get('degraded', 0)}, "
            f"diferença média de AP `{group['delta_ap'].mean():+.6f}`."
        )
    lines.extend(
        [
            "",
            "## Limites de interpretação",
            "",
            "- São 20 identidades selecionadas entre as mais frequentes, não todas as pessoas do acervo.",
            "- O cenário global pode coincidir com a pessoa em um evento, mas não representa identidade.",
            "- Os pesos foram declarados antes da execução; os casos positivos não autorizam escolher um peso retrospectivamente.",
            "- Gallagher não é uma validação agro. Uma avaliação agro autorizada seria outro estudo, fora do escopo deste artigo.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.visual_topk <= 0:
        raise ValueError("visual-topk must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results_csv, dtype={"query_id": str, "image_id": str})
    per_query = pd.read_csv(args.per_query_csv, dtype={"query_id": str})
    aggregate = pd.read_csv(args.aggregate_csv)
    queries = pd.read_csv(args.queries_csv, dtype=str)
    relevance = relevance_from_csv(args.relevance_csv)

    labeled = label_topk(results, relevance)
    deltas = build_deltas(per_query)
    primary = build_primary_summary(labeled, deltas, queries, args.primary_fusion)
    cases = choose_cases(primary)
    aggregate_comparison = aggregate.copy()
    face_map = float(aggregate.loc[aggregate["method"] == "face_only", "mAP"].iloc[0])
    aggregate_comparison["delta_map_vs_face"] = aggregate_comparison["mAP"] - face_map

    labeled_path = args.output_dir / "topk_labeled.csv"
    deltas_path = args.output_dir / "method_deltas.csv"
    primary_path = args.output_dir / "primary_fusion_query_analysis.csv"
    cases_path = args.output_dir / "selected_cases.csv"
    aggregate_path = args.output_dir / "aggregate_comparison.csv"
    report_path = args.output_dir / "analysis_report.md"
    aggregate_figure_path = args.output_dir / "figures" / "aggregate_metrics.png"
    labeled.to_csv(labeled_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    primary.to_csv(primary_path, index=False)
    cases.to_csv(cases_path, index=False)
    aggregate_comparison.to_csv(aggregate_path, index=False)
    write_report(report_path, aggregate, deltas, primary, cases, args.primary_fusion)
    plot_aggregate(aggregate, aggregate_figure_path)

    result_files: dict[str, Path] = {
        "labeled_topk": labeled_path,
        "method_deltas": deltas_path,
        "primary_query_analysis": primary_path,
        "selected_cases": cases_path,
        "aggregate_comparison": aggregate_path,
        "analysis_report": report_path,
        "aggregate_figure": aggregate_figure_path,
    }
    for case in cases.itertuples(index=False):
        query = queries[queries["query_id"] == str(case.query_id)].iloc[0]
        figure_path = args.output_dir / "figures" / f"{case.case}_{case.query_id}.png"
        plot_case(
            query,
            labeled,
            args.primary_fusion,
            figure_path,
            args.visual_topk,
            str(case.case),
        )
        result_files[f"case_{case.case}"] = figure_path

    manifest_path = write_run_manifest(
        args.output_dir / "error_analysis_manifest.json",
        script_path=Path(__file__),
        method="gallagher_paired_fusion_error_analysis",
        configuration={
            "baseline_method": "face_only",
            "primary_fusion": args.primary_fusion,
            "delta_tolerance": 1e-12,
            "visual_topk": args.visual_topk,
            "case_selection": "max_improvement_first_unchanged_max_degradation",
        },
        inputs={
            "results_csv": args.results_csv,
            "per_query_csv": args.per_query_csv,
            "aggregate_csv": args.aggregate_csv,
            "queries_csv": args.queries_csv,
            "relevance_csv": args.relevance_csv,
        },
        result_files=result_files,
        extra={
            "queries": int(primary["query_id"].nunique()),
            "improved": int((primary["impact"] == "improved").sum()),
            "unchanged": int((primary["impact"] == "unchanged").sum()),
            "degraded": int((primary["impact"] == "degraded").sum()),
        },
    )
    print(f"[OK] Error analysis: {args.output_dir}")
    print(f"[OK] Run manifest: {manifest_path}")
    print(cases[["case", "query_id", "delta_ap", "top10_overlap"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
