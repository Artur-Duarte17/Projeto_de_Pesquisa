from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from face_lib import build_face_app, load_face_index, query_embedding_from_image
from fusion_lib import fuse_cosine_score_matrices
from global_lib import load_global_index
from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval_common import average_precision, relevance_from_csv, resolve_stored_path
from run_manifest import write_run_manifest


WEIGHTS = [
    ("face_only", 1.0, 0.0),
    ("global_context_only", 0.0, 1.0),
    ("fusion_0p9_0p1", 0.9, 0.1),
    ("fusion_0p7_0p3", 0.7, 0.3),
    ("fusion_0p5_0p5", 0.5, 0.5),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit full Agrishow rankings and build the qualitative Top-5 figure."
    )
    parser.add_argument(
        "--queries-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_queries.csv",
    )
    parser.add_argument(
        "--relevance-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_relevance.csv",
    )
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=DATA_DIR
        / "raw"
        / "agrishow_2022"
        / "metadata"
        / "agrishow_2022_selected.csv",
    )
    parser.add_argument(
        "--face-index-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-022_agrishow_face_index",
    )
    parser.add_argument(
        "--global-index-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-022_agrishow_global_index",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-023_agrishow_paired_fusion",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-024_agrishow_analysis",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--det-size", type=int, default=640)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_face_box(image_path: Path, bbox: tuple[float, float, float, float] | None) -> Image.Image:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read figure image: {image_path}")
    if bbox is not None:
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        thickness = max(5, min(image.shape[:2]) // 350)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), thickness)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def place_contained(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    available_width = right - left
    available_height = bottom - top
    rendered = image.copy()
    rendered.thumbnail((available_width, available_height), Image.Resampling.LANCZOS)
    x = left + (available_width - rendered.width) // 2
    y = top + (available_height - rendered.height) // 2
    canvas.paste(rendered, (x, y))


def create_figure(
    query_path: Path,
    selected: pd.DataFrame,
    output_path: Path,
) -> None:
    panel_width = 1200
    panel_height = 800
    header_height = 72
    gap = 20
    canvas = Image.new(
        "RGB",
        (panel_width * 3 + gap * 4, panel_height * 2 + gap * 3),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(38, bold=True)
    small_font = load_font(30)

    panels: list[tuple[str, str, Image.Image]] = [
        (
            "Consulta facial",
            "recorte da fotografia-fonte",
            draw_face_box(query_path, None),
        )
    ]
    for row in selected.itertuples(index=False):
        bbox = (
            float(row.bbox_x1),
            float(row.bbox_y1),
            float(row.bbox_x2),
            float(row.bbox_y2),
        )
        panels.append(
            (
                f"{int(row.rank)}º resultado",
                f"relevante · escore {float(row.score):.4f}",
                draw_face_box(resolve_stored_path(str(row.image_path)), bbox),
            )
        )

    for panel_index, (title, subtitle, image) in enumerate(panels):
        row = panel_index // 3
        column = panel_index % 3
        left = gap + column * (panel_width + gap)
        top = gap + row * (panel_height + gap)
        right = left + panel_width
        bottom = top + panel_height
        draw.rectangle((left, top, right, bottom), fill=(245, 245, 245), outline=(90, 90, 90), width=3)
        draw.text((left + 20, top + 10), title, fill=(20, 20, 20), font=font)
        subtitle_width = draw.textbbox((0, 0), subtitle, font=small_font)[2]
        draw.text(
            (right - subtitle_width - 20, top + 20),
            subtitle,
            fill=(50, 50, 50),
            font=small_font,
        )
        place_contained(canvas, image, (left + 8, top + header_height, right - 8, bottom - 8))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")

    queries = pd.read_csv(args.queries_csv, dtype=str).fillna("")
    require(len(queries) == 1, f"Expected one query, found {len(queries)}")
    query = queries.iloc[0]
    query_id = str(query["query_id"])
    source_image_id = str(query["source_image_id"])
    query_path = resolve_stored_path(str(query["face_query_path"])).resolve()
    query_face_mode = str(query.get("query_face_mode", "largest") or "largest")
    query_face_index = int(query.get("query_face_index", 0) or 0)

    relevance = relevance_from_csv(args.relevance_csv)
    relevant_ids = set(relevance[query_id]) - {source_image_id}
    require(len(relevant_ids) == 90, f"Expected 90 eligible relevant images, found {len(relevant_ids)}")
    expected_topk = pd.read_csv(args.evaluation_dir / "fusion_topk_results.csv", dtype=str)
    expected_metrics = pd.read_csv(args.evaluation_dir / "fusion_metrics_per_query.csv")
    face_embeddings, face_metadata = load_face_index(args.face_index_dir)
    global_embeddings, global_metadata = load_global_index(args.global_index_dir)
    face_metadata = face_metadata.sort_values("embedding_row").reset_index(drop=True)
    global_metadata = global_metadata.sort_values("embedding_row").reset_index(drop=True)

    image_ids = global_metadata["image_id"].astype(str).tolist()
    image_id_to_code = {image_id: code for code, image_id in enumerate(image_ids)}
    require(len(image_id_to_code) == 124, "Expected 124 unique global image IDs")
    require(source_image_id in image_id_to_code, "Source image is absent from global index")
    face_image_codes = face_metadata["image_id"].astype(str).map(image_id_to_code).to_numpy(np.int64)

    face_app = build_face_app(args.device, args.det_size)
    query_face_embedding, query_bbox = query_embedding_from_image(
        query_path,
        face_app,
        query_face_mode=query_face_mode,
        query_face_index=query_face_index,
    )

    device = torch.device(args.device)
    face_gallery = functional.normalize(torch.from_numpy(face_embeddings).to(device), dim=1)
    global_gallery = functional.normalize(torch.from_numpy(global_embeddings).to(device), dim=1)
    face_query = functional.normalize(
        torch.from_numpy(np.asarray(query_face_embedding, dtype=np.float32)).to(device), dim=0
    )
    source_code = image_id_to_code[source_image_id]
    global_query = functional.normalize(
        torch.from_numpy(global_embeddings[source_code]).to(device), dim=0
    )
    synchronize(device)
    face_cosine = (face_gallery @ face_query).cpu().numpy()
    global_cosine = (global_gallery @ global_query).cpu().numpy()

    face_image_cosine = np.full(len(image_ids), -np.inf, dtype=np.float32)
    best_face_rows = np.full(len(image_ids), -1, dtype=np.int64)
    for face_row, image_code in enumerate(face_image_codes):
        score = float(face_cosine[face_row])
        if score > float(face_image_cosine[image_code]):
            face_image_cosine[image_code] = score
            best_face_rows[image_code] = face_row
    face_available = np.isfinite(face_image_cosine)

    full_rows: list[dict[str, object]] = []
    analysis_rows: list[dict[str, object]] = []
    face_ranking_ids: list[str] | None = None
    for method, face_weight, global_weight in WEIGHTS:
        fused, face_unit, global_unit = fuse_cosine_score_matrices(
            face_image_cosine[None, :],
            global_cosine[None, :],
            face_available[None, :],
            face_weight,
            global_weight,
        )
        scores = fused[0]
        scores[source_code] = -np.inf
        ranked_codes = [int(code) for code in np.argsort(-scores) if int(code) != source_code]
        ranked_ids = [image_ids[code] for code in ranked_codes]
        require(len(ranked_ids) == 123, f"{method} ranking does not contain 123 images")

        expected = expected_topk[expected_topk["method"] == method].copy()
        expected["rank"] = expected["rank"].astype(int)
        expected = expected.sort_values("rank")
        require(expected["image_id"].tolist() == ranked_ids[:10], f"{method} Top-10 differs from EX-023")
        require(
            np.allclose(expected["score"].astype(float), scores[ranked_codes[:10]], atol=1e-6),
            f"{method} Top-10 scores differ from EX-023",
        )
        ap = average_precision(ranked_ids, relevant_ids)
        expected_ap = float(
            expected_metrics.loc[expected_metrics["method"] == method, "average_precision"].iloc[0]
        )
        require(np.isclose(ap, expected_ap, atol=1e-12), f"{method} AP differs from EX-023")

        if method == "face_only":
            face_ranking_ids = ranked_ids
        require(face_ranking_ids is not None, "Face ranking must be evaluated first")
        face_positions = {image_id: rank for rank, image_id in enumerate(face_ranking_ids, start=1)}
        irrelevant_ranks = [rank for rank, image_id in enumerate(ranked_ids, start=1) if image_id not in relevant_ids]
        relevant_ranks = [rank for rank, image_id in enumerate(ranked_ids, start=1) if image_id in relevant_ids]
        analysis_rows.append(
            {
                "method": method,
                "average_precision": ap,
                "delta_ap_vs_face": ap - float(expected_metrics.loc[expected_metrics["method"] == "face_only", "average_precision"].iloc[0]),
                "first_irrelevant_rank": min(irrelevant_ranks),
                "last_relevant_rank": max(relevant_ranks),
                "irrelevant_before_last_relevant": sum(rank < max(relevant_ranks) for rank in irrelevant_ranks),
                "top10_overlap_with_face": len(set(ranked_ids[:10]) & set(face_ranking_ids[:10])),
            }
        )

        for rank, image_code in enumerate(ranked_codes, start=1):
            face_row = int(best_face_rows[image_code])
            face_meta = face_metadata.iloc[face_row] if face_row >= 0 else None
            image_id = image_ids[image_code]
            full_rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "query_id": query_id,
                    "image_id": image_id,
                    "image_path": str(global_metadata.iloc[image_code]["image_path"]),
                    "relevant": int(image_id in relevant_ids),
                    "score": float(scores[image_code]),
                    "face_score": float(face_unit[0, image_code]),
                    "global_score": float(global_unit[0, image_code]),
                    "rank_delta_vs_face": rank - face_positions.get(image_id, rank),
                    "matched_face_id": "" if face_meta is None else str(face_meta["face_id"]),
                    "bbox_x1": np.nan if face_meta is None else float(face_meta["bbox_x1"]),
                    "bbox_y1": np.nan if face_meta is None else float(face_meta["bbox_y1"]),
                    "bbox_x2": np.nan if face_meta is None else float(face_meta["bbox_x2"]),
                    "bbox_y2": np.nan if face_meta is None else float(face_meta["bbox_y2"]),
                }
            )

    full_ranking = pd.DataFrame(full_rows)
    analysis = pd.DataFrame(analysis_rows)
    selected = full_ranking[(full_ranking["method"] == "face_only") & (full_ranking["rank"] <= 5)].copy()
    require(selected["relevant"].astype(int).eq(1).all(), "A selected Top-5 panel is not relevant")
    require(not selected[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().any().any(), "Missing face box in figure")

    output_dir.mkdir(parents=True, exist_ok=False)
    full_ranking_path = output_dir / "full_rankings.csv"
    analysis_path = output_dir / "ranking_analysis.csv"
    selected_path = output_dir / "qualitative_selection.csv"
    figure_path = output_dir / "agrishow_face_top5.png"
    attribution_path = output_dir / "figure_attributions.csv"
    caption_path = output_dir / "figure_caption.txt"
    full_ranking.to_csv(full_ranking_path, index=False)
    analysis.to_csv(analysis_path, index=False)
    selected.to_csv(selected_path, index=False)
    create_figure(query_path, selected, figure_path)

    inventory = pd.read_csv(args.inventory_csv, dtype=str).fillna("").set_index("image_id")
    attribution_rows: list[dict[str, str]] = []
    figure_ids = [source_image_id] + selected["image_id"].astype(str).tolist()
    for panel_index, image_id in enumerate(figure_ids):
        row = inventory.loc[image_id]
        attribution_rows.append(
            {
                "panel": "consulta" if panel_index == 0 else f"resultado_{panel_index}",
                "image_id": image_id,
                "file_name": str(row["file_name"]),
                "photographer": str(row["photographer"]),
                "source": "Palácio do Planalto / Wikimedia Commons",
                "license": str(row["license"]),
                "license_url": str(row["license_url"]),
                "page_url": str(row["page_url"]),
                "modification": "recorte" if panel_index == 0 else "caixa do rosto e composição em painel",
            }
        )
    pd.DataFrame(attribution_rows).to_csv(attribution_path, index=False)
    caption_path.write_text(
        "Consulta facial e cinco primeiras fotografias recuperadas pelo método facial na "
        "coleção Agrishow 2022. As caixas amarelas indicam o rosto que produziu a maior "
        "similaridade em cada fotografia. A fotografia-fonte foi excluída do ranking. "
        "Imagens sob CC BY 2.0; créditos e ligações individuais constam na tabela de atribuições.",
        encoding="utf-8",
    )

    manifest_path = write_run_manifest(
        output_dir / "analysis_manifest.json",
        script_path=Path(__file__),
        method="agrishow_2022_qualitative_error_analysis",
        configuration={
            "device": args.device,
            "det_size": args.det_size,
            "weights": [[face, global_] for _, face, global_ in WEIGHTS],
            "query_face_mode": query_face_mode,
            "query_face_index": query_face_index,
            "figure_method": "face_only",
            "figure_topk": 5,
        },
        inputs={
            "queries_csv": args.queries_csv,
            "relevance_csv": args.relevance_csv,
            "inventory_csv": args.inventory_csv,
            "ex023_topk": args.evaluation_dir / "fusion_topk_results.csv",
            "ex023_metrics": args.evaluation_dir / "fusion_metrics_per_query.csv",
            "face_embeddings": args.face_index_dir / "face_embeddings.npy",
            "face_metadata": args.face_index_dir / "face_metadata.csv",
            "global_embeddings": args.global_index_dir / "global_embeddings.npy",
            "global_metadata": args.global_index_dir / "global_metadata.csv",
        },
        result_files={
            "full_rankings": full_ranking_path,
            "ranking_analysis": analysis_path,
            "qualitative_selection": selected_path,
            "figure": figure_path,
            "figure_attributions": attribution_path,
            "figure_caption": caption_path,
        },
        extra={
            "query_id": query_id,
            "query_bbox": query_bbox,
            "source_image_id": source_image_id,
            "source_excluded": True,
            "full_ranking_rows": len(full_ranking),
            "top10_reproduced_for_all_methods": True,
            "selected_image_ids": selected["image_id"].astype(str).tolist(),
        },
    )

    print("EX-024: ANALYSIS GENERATED")
    print(analysis.to_string(index=False))
    print(f"figure={figure_path}")
    print(f"attributions={attribution_path}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
