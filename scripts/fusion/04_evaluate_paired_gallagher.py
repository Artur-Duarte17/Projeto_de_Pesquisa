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

from face_lib import build_face_app, load_face_index, query_embedding_from_image
from fusion_lib import fuse_cosine_score_matrices
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


DEFAULT_WEIGHTS = [(1.0, 0.0), (0.0, 1.0), (0.9, 0.1), (0.7, 0.3), (0.5, 0.5)]


def parse_weight(value: str) -> tuple[float, float]:
    separator = "/" if "/" in value else "," if "," in value else None
    if separator is None:
        raise argparse.ArgumentTypeError("Use FACE/GLOBAL, for example 0.7/0.3")
    face_weight, global_weight = (float(part) for part in value.split(separator, 1))
    if face_weight < 0.0 or global_weight < 0.0:
        raise argparse.ArgumentTypeError("Weights must be non-negative")
    if not np.isclose(face_weight + global_weight, 1.0, atol=1e-9):
        raise argparse.ArgumentTypeError("Weights must sum to one")
    return face_weight, global_weight


def method_name(face_weight: float, global_weight: float) -> str:
    if face_weight == 1.0 and global_weight == 0.0:
        return "face_only"
    if face_weight == 0.0 and global_weight == 1.0:
        return "global_context_only"
    return f"fusion_{face_weight:.1f}_{global_weight:.1f}".replace(".", "p")


def rank_image_codes(
    scores: np.ndarray,
    image_ids: list[str],
    excluded_codes: set[int],
) -> list[int]:
    if scores.ndim != 1 or len(scores) != len(image_ids):
        raise ValueError("Scores and image IDs must describe the same one-dimensional gallery")
    if not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be finite before source exclusion")
    eligible_codes = [code for code in range(len(image_ids)) if code not in excluded_codes]
    return sorted(eligible_codes, key=lambda code: (-float(scores[code]), image_ids[code]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paired face/context retrieval.")
    parser.add_argument(
        "--queries-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "gallagher_fusion_queries.csv",
    )
    parser.add_argument(
        "--relevance-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "gallagher_relevance.csv",
    )
    parser.add_argument(
        "--face-index-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-010_gallagher_face_index",
    )
    parser.add_argument(
        "--global-index-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-015_gallagher_global_index",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-017_gallagher_paired_fusion",
    )
    parser.add_argument("--save-topk", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--weights-list", nargs="+", type=parse_weight, default=DEFAULT_WEIGHTS)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--save-visual-examples", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset-name", default="gallagher")
    parser.add_argument("--face-query-source", default="corrected_annotated_eye_crop")
    parser.add_argument(
        "--global-query-source",
        default="source_photo_descriptor_from_ex015",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    args = parse_args()
    if args.save_topk <= 0 or args.batch_size <= 0:
        raise ValueError("save-topk and batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")

    results_path = args.output_dir / "fusion_topk_results.csv"
    per_query_path = args.output_dir / "fusion_metrics_per_query.csv"
    metrics_path = args.output_dir / "fusion_metrics.csv"
    run_manifest_path = args.output_dir / "fusion_run_manifest.json"
    protected_outputs = [results_path, per_query_path, metrics_path, run_manifest_path]
    if args.save_visual_examples:
        protected_outputs.append(args.output_dir / "visual_examples" / "fusion_examples.png")
    existing_outputs = [path for path in protected_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        existing_list = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Evaluation outputs already exist: {existing_list}")

    queries = pd.read_csv(args.queries_csv, dtype=str)
    if args.max_queries is not None:
        queries = queries.head(args.max_queries).copy()
    if "target_label" not in queries.columns and "target_id" in queries.columns:
        queries["target_label"] = queries["target_id"]
    required_query_columns = {
        "query_id",
        "face_query_path",
        "global_query_path",
        "source_image_id",
        "target_label",
        "query_face_mode",
        "query_target_x",
        "query_target_y",
    }
    missing_query_columns = sorted(required_query_columns - set(queries.columns))
    if missing_query_columns:
        raise ValueError(f"Missing paired-query columns: {missing_query_columns}")
    if queries.empty or queries["query_id"].duplicated().any():
        raise ValueError("Queries must be non-empty and have unique IDs")

    face_embeddings, face_metadata = load_face_index(args.face_index_dir)
    global_embeddings, global_metadata = load_global_index(args.global_index_dir)
    face_metadata = face_metadata.sort_values("embedding_row").reset_index(drop=True)
    global_metadata = global_metadata.sort_values("embedding_row").reset_index(drop=True)
    if not np.array_equal(
        face_metadata["embedding_row"].to_numpy(np.int64), np.arange(len(face_metadata))
    ):
        raise ValueError("Face metadata is not aligned with the embedding matrix")
    if not np.array_equal(
        global_metadata["embedding_row"].to_numpy(np.int64), np.arange(len(global_metadata))
    ):
        raise ValueError("Global metadata is not aligned with the embedding matrix")

    image_ids = global_metadata["image_id"].astype(str).tolist()
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Global image IDs are not unique")
    image_id_to_code = {image_id: code for code, image_id in enumerate(image_ids)}
    face_image_ids = face_metadata["image_id"].astype(str)
    missing_face_images = sorted(set(face_image_ids) - set(image_ids))
    if missing_face_images:
        raise ValueError(f"Face images absent from global index: {missing_face_images[:3]}")
    face_image_codes_np = face_image_ids.map(image_id_to_code).to_numpy(np.int64)

    relevance = relevance_from_csv(args.relevance_csv)
    exclusions: dict[str, set[str]] = {}
    source_codes: list[int] = []
    for query in queries.itertuples(index=False):
        query_id = str(query.query_id)
        source_image_id = str(query.source_image_id)
        if source_image_id not in image_id_to_code:
            raise ValueError(f"Source image is absent from global index: {source_image_id}")
        excluded = {source_image_id}
        if hasattr(query, "exclude_image_ids") and pd.notna(query.exclude_image_ids):
            excluded.update(parse_image_ids(str(query.exclude_image_ids)))
        exclusions[query_id] = excluded
        source_codes.append(image_id_to_code[source_image_id])
    relevance = apply_relevance_exclusions(relevance, exclusions)

    face_app = build_face_app(device=args.device, det_size=args.det_size)
    query_face_embeddings: list[np.ndarray] = []
    query_bboxes: list[str] = []
    query_face_modes: list[str] = []
    query_face_indices: list[int] = []
    query_target_xs: list[float] = []
    query_target_ys: list[float] = []
    face_extraction_ms: list[float] = []
    for index, query in enumerate(queries.itertuples(index=False), start=1):
        query_face_mode = str(query.query_face_mode)
        if query_face_mode != "annotated_eye_midpoint":
            raise ValueError(
                f"Query {query.query_id} does not use annotated-eye target selection: "
                f"{query_face_mode}"
            )
        query_face_index_value = getattr(query, "query_face_index", 0)
        query_face_index = int(query_face_index_value) if pd.notna(query_face_index_value) else 0
        query_target_x = float(query.query_target_x)
        query_target_y = float(query.query_target_y)
        if not np.isfinite(query_target_x) or not np.isfinite(query_target_y):
            raise ValueError(f"Query {query.query_id} has a non-finite annotated target point")
        started = time.perf_counter()
        embedding, bbox = query_embedding_from_image(
            Path(str(query.face_query_path)),
            face_app,
            query_face_mode=query_face_mode,
            query_face_index=query_face_index,
            query_target_x=query_target_x,
            query_target_y=query_target_y,
        )
        face_extraction_ms.append((time.perf_counter() - started) * 1000.0)
        query_face_embeddings.append(np.asarray(embedding, dtype=np.float32))
        query_bboxes.append(str(bbox))
        query_face_modes.append(query_face_mode)
        query_face_indices.append(query_face_index)
        query_target_xs.append(query_target_x)
        query_target_ys.append(query_target_y)
        print(f"[INFO] extracted target faces: {index}/{len(queries)}")

    query_face_matrix = np.vstack(query_face_embeddings).astype(np.float32)
    query_global_matrix = np.asarray(global_embeddings[source_codes], dtype=np.float32)
    device = torch.device(args.device)
    face_gallery = functional.normalize(
        torch.from_numpy(np.asarray(face_embeddings, dtype=np.float32)).to(device), dim=1
    )
    global_gallery = functional.normalize(
        torch.from_numpy(np.asarray(global_embeddings, dtype=np.float32)).to(device), dim=1
    )
    face_queries = functional.normalize(torch.from_numpy(query_face_matrix).to(device), dim=1)
    global_queries = functional.normalize(torch.from_numpy(query_global_matrix).to(device), dim=1)
    face_image_codes = torch.from_numpy(face_image_codes_np).to(device)

    rankings_by_method: dict[str, dict[str, list[str]]] = {
        method_name(*weights): {} for weights in args.weights_list
    }
    backend_times: dict[str, dict[str, float]] = {
        method_name(*weights): {} for weights in args.weights_list
    }
    topk_rows: list[dict[str, object]] = []

    for batch_start in range(0, len(queries), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(queries))
        batch_count = batch_end - batch_start
        face_cosine = face_queries[batch_start:batch_end] @ face_gallery.T
        face_image_cosine = torch.full(
            (batch_count, len(image_ids)),
            -torch.inf,
            dtype=face_cosine.dtype,
            device=device,
        )
        face_image_cosine.scatter_reduce_(
            1,
            face_image_codes.unsqueeze(0).expand(batch_count, -1),
            face_cosine,
            reduce="amax",
            include_self=True,
        )
        global_cosine = global_queries[batch_start:batch_end] @ global_gallery.T
        face_cosine_np = face_image_cosine.cpu().numpy()
        global_cosine_np = global_cosine.cpu().numpy()
        face_available = np.isfinite(face_cosine_np)

        for face_weight, global_weight in args.weights_list:
            method = method_name(face_weight, global_weight)
            synchronize(device)
            started = time.perf_counter()
            fused_np, face_unit_np, global_unit_np = fuse_cosine_score_matrices(
                face_cosine_np,
                global_cosine_np,
                face_available,
                face_weight,
                global_weight,
            )
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
                    raise ValueError(f"No source exclusion resolved for {query.query_id}")
                batch_excluded_codes.append(excluded_codes)
            ranked_codes_batch = [
                rank_image_codes(fused_np[local_index], image_ids, excluded_codes)
                for local_index, excluded_codes in enumerate(batch_excluded_codes)
            ]
            synchronize(device)
            elapsed_per_query_ms = (time.perf_counter() - started) * 1000.0 / batch_count

            for local_index, query in enumerate(
                queries.iloc[batch_start:batch_end].itertuples(index=False)
            ):
                query_index = batch_start + local_index
                query_id = str(query.query_id)
                ranked_codes = ranked_codes_batch[local_index]
                rankings_by_method[method][query_id] = [image_ids[code] for code in ranked_codes]
                backend_times[method][query_id] = elapsed_per_query_ms
                for rank, image_code in enumerate(ranked_codes[: args.save_topk], start=1):
                    candidate = global_metadata.iloc[image_code]
                    topk_rows.append(
                        {
                            "method": method,
                            "face_weight": face_weight,
                            "global_weight": global_weight,
                            "query_id": query_id,
                            "face_query_path": str(query.face_query_path),
                            "global_query_path": str(query.global_query_path),
                            "source_image_id": str(query.source_image_id),
                            "target_label": str(query.target_label),
                            "query_bbox": query_bboxes[query_index],
                            "query_face_mode": query_face_modes[query_index],
                            "query_face_index": query_face_indices[query_index],
                            "query_target_x": query_target_xs[query_index],
                            "query_target_y": query_target_ys[query_index],
                            "rank": rank,
                            "image_id": str(candidate["image_id"]),
                            "image_path": str(candidate["image_path"]),
                            "score": float(fused_np[local_index, image_code]),
                            "face_score": float(face_unit_np[local_index, image_code]),
                            "global_score": float(global_unit_np[local_index, image_code]),
                            "face_query_extraction_ms": face_extraction_ms[query_index],
                            "retrieval_backend_ms": elapsed_per_query_ms,
                        }
                    )
        print(f"[INFO] evaluated paired queries: {batch_end}/{len(queries)}")

    topk = pd.DataFrame(topk_rows)
    per_query_frames: list[pd.DataFrame] = []
    aggregate_frames: list[pd.DataFrame] = []
    for face_weight, global_weight in args.weights_list:
        method = method_name(face_weight, global_weight)
        per_query = summarize_rankings(rankings_by_method[method], relevance, ks=(5, 10))
        per_query["method"] = method
        per_query["face_weight"] = face_weight
        per_query["global_weight"] = global_weight
        per_query["face_query_extraction_ms"] = per_query["query_id"].map(
            dict(zip(queries["query_id"].astype(str), face_extraction_ms, strict=True))
        )
        per_query["retrieval_backend_ms"] = per_query["query_id"].map(backend_times[method])
        per_query_frames.append(per_query)
        aggregate = aggregate_metrics(per_query, method)
        aggregate["face_weight"] = face_weight
        aggregate["global_weight"] = global_weight
        aggregate["face_query_extraction_ms"] = float(per_query["face_query_extraction_ms"].mean())
        aggregate["retrieval_backend_ms"] = float(per_query["retrieval_backend_ms"].mean())
        aggregate_frames.append(aggregate)

    per_query_out = pd.concat(per_query_frames, ignore_index=True)
    metrics_out = pd.concat(aggregate_frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topk.to_csv(results_path, index=False)
    per_query_out.to_csv(per_query_path, index=False)
    metrics_out.to_csv(metrics_path, index=False)

    result_files = {
        "saved_topk": results_path,
        "per_query_metrics": per_query_path,
        "aggregate_metrics": metrics_path,
    }
    if args.save_visual_examples:
        visual_method = "fusion_0p7_0p3"
        first_query_id = str(queries.iloc[0]["query_id"])
        visual_path = args.output_dir / "visual_examples" / "fusion_examples.png"
        save_visual_grid(
            topk[(topk["method"] == visual_method) & (topk["query_id"] == first_query_id)],
            visual_path,
            f"Paired {args.dataset_name} fusion: {first_query_id} ({visual_method})",
        )
        result_files["visual_examples"] = visual_path

    manifest_path = write_run_manifest(
        run_manifest_path,
        script_path=Path(__file__),
        method=f"{args.dataset_name}_paired_face_global_fusion",
        configuration={
            "save_topk": args.save_topk,
            "ranking_depth": "all_eligible_images",
            "metric_ks": [5, 10],
            "device": args.device,
            "det_size": args.det_size,
            "batch_size": args.batch_size,
            "weights_list": [list(weights) for weights in args.weights_list],
            "max_queries": args.max_queries,
            "dataset_name": args.dataset_name,
            "face_query_source": args.face_query_source,
            "global_query_source": args.global_query_source,
            "query_face_selection": "annotated-eye midpoint contained by detected bbox",
            "query_face_tie_break": "nearest bbox center, then smallest area, then bbox coordinates",
            "ranking_tie_break": "score descending, then image_id ascending",
            "missing_face_rule": "face unit score zero; global score remains available",
        },
        inputs={
            "queries_csv": args.queries_csv,
            "relevance_csv": args.relevance_csv,
            "face_embeddings": args.face_index_dir / "face_embeddings.npy",
            "face_metadata": args.face_index_dir / "face_metadata.csv",
            "face_index_manifest": args.face_index_dir / "face_index_manifest.json",
            "global_embeddings": args.global_index_dir / "global_embeddings.npy",
            "global_metadata": args.global_index_dir / "global_metadata.csv",
            "global_index_manifest": args.global_index_dir / "global_index_manifest.json",
        },
        result_files=result_files,
        extra={
            "failed_queries": 0,
            "queries": int(len(queries)),
            "eligible_gallery_images": int(len(global_metadata) - 1),
            "images_without_detected_faces": int(len(global_metadata) - face_metadata["image_id"].nunique()),
            "dataset_name": args.dataset_name,
        },
    )
    print(f"[OK] Fusion metrics: {metrics_path}")
    print(f"[OK] Run manifest: {manifest_path}")
    print(metrics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
