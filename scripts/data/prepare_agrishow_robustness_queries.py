from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from face_lib import build_face_app, face_area, sorted_faces
from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval_common import rel_to_root, resolve_stored_path, sha256_file
from run_manifest import write_run_manifest
from scripts.data.prepare_agrishow_face_query import (
    bbox_iou,
    clamp_bbox,
    draw_review,
    expanded_square_bbox,
)


DEFAULT_SEED = 1161755629553718132
TARGET_ID = "agrishow2022_person_01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select nine additional Agrishow sources before retrieval, create numbered "
            "face reviews, and optionally freeze ten paired queries."
        )
    )
    parser.add_argument(
        "--base-queries-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_queries.csv",
    )
    parser.add_argument(
        "--base-relevance-csv",
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
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-026_agrishow_robustness_protocol",
    )
    parser.add_argument(
        "--queries-output-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_robustness_queries.csv",
    )
    parser.add_argument(
        "--relevance-output-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_robustness_relevance.csv",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=DATA_DIR / "query" / "agrishow_2022_robustness",
    )
    parser.add_argument(
        "--target-indices-csv",
        type=Path,
        default=None,
        help="CSV with query_id and manually confirmed target_face_index.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--additional-queries", type=int, default=9)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--crop-scale", type=float, default=2.0)
    return parser.parse_args()


def selection_key(seed: int, image_id: str) -> str:
    payload = f"{seed}|{TARGET_ID}|{image_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def candidate_rows(image: np.ndarray, faces: list) -> list[dict[str, object]]:
    height, width = image.shape[:2]
    rows: list[dict[str, object]] = []
    for index, face in enumerate(faces):
        x1, y1, x2, y2 = clamp_bbox(
            tuple(float(value) for value in face.bbox), width, height
        )
        rows.append(
            {
                "face_index": index,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_width": x2 - x1,
                "bbox_height": y2 - y1,
                "detection_score": round(float(getattr(face, "det_score", 0.0)), 8),
            }
        )
    return rows


def make_contact_sheet(items: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    columns = 3
    panel_width, panel_height = 1200, 900
    rows = int(np.ceil(len(items) / columns))
    sheet = np.full((rows * panel_height, columns * panel_width, 3), 245, dtype=np.uint8)
    for panel_index, (query_id, image) in enumerate(items):
        row, column = divmod(panel_index, columns)
        available_height = panel_height - 70
        scale = min(panel_width / image.shape[1], available_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        x = column * panel_width + (panel_width - resized.shape[1]) // 2
        y = row * panel_height + 70 + (available_height - resized.shape[0]) // 2
        sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(
            sheet,
            query_id,
            (column * panel_width + 20, row * panel_height + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Could not write contact sheet: {output_path}")


def select_sources(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_queries = pd.read_csv(args.base_queries_csv, dtype=str).fillna("")
    base_relevance = pd.read_csv(args.base_relevance_csv, dtype=str).fillna("")
    inventory = pd.read_csv(args.inventory_csv, dtype=str).fillna("")
    require(len(base_queries) == 1, "Expected exactly one original Agrishow query")
    require(len(base_relevance) == 124, "Expected 124 original relevance rows")
    require(inventory["image_id"].nunique() == 124, "Expected 124 unique inventory IDs")

    original_source = str(base_queries.iloc[0]["source_image_id"])
    present_ids = base_relevance.loc[
        base_relevance["relevant"].astype(int).eq(1), "image_id"
    ].astype(str)
    require(len(present_ids) == 91, f"Expected 91 present images, found {len(present_ids)}")
    candidates = [image_id for image_id in present_ids if image_id != original_source]
    require(len(candidates) == 90, "Expected 90 candidates after excluding original source")

    ordered = sorted(candidates, key=lambda image_id: selection_key(args.seed, image_id))
    selected_ids = ordered[: args.additional_queries]
    inventory_by_id = inventory.set_index("image_id", drop=False)
    rows: list[dict[str, object]] = []
    for offset, image_id in enumerate(selected_ids, start=2):
        item = inventory_by_id.loc[image_id]
        image_path = resolve_stored_path(str(item["image_path"])).resolve()
        rows.append(
            {
                "query_id": f"agrishow2022_query_{offset:02d}",
                "target_id": TARGET_ID,
                "source_image_id": image_id,
                "source_file_name": str(item["file_name"]),
                "source_image_path": rel_to_root(image_path),
                "source_image_sha256": sha256_file(image_path),
                "selection_seed": args.seed,
                "selection_key": selection_key(args.seed, image_id),
                "selection_rule": "lowest_sha256(seed,target_id,image_id)",
                "candidate_count": len(candidates),
            }
        )
    return pd.DataFrame(rows), base_queries, base_relevance


def prepare_reviews(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    selection, _, _ = select_sources(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = output_dir / "selected_sources.csv"
    selection.to_csv(selection_path, index=False)

    app = build_face_app(args.device, args.det_size)
    face_dir = output_dir / "face_reviews"
    face_dir.mkdir()
    contact_items: list[tuple[str, np.ndarray]] = []
    face_counts: dict[str, int] = {}
    review_files: dict[str, Path] = {}
    candidate_files: dict[str, Path] = {}
    for row in selection.itertuples(index=False):
        query_id = str(row.query_id)
        source_path = resolve_stored_path(str(row.source_image_path)).resolve()
        image = cv2.imread(str(source_path))
        if image is None:
            raise FileNotFoundError(f"Could not read source image: {source_path}")
        faces = sorted_faces(app.get(image))
        require(bool(faces), f"No face detected in selected source: {query_id}")
        candidates = candidate_rows(image, faces)
        candidates_path = face_dir / f"{query_id}_candidates.csv"
        review_path = face_dir / f"{query_id}_review.jpg"
        pd.DataFrame(candidates).to_csv(candidates_path, index=False)
        review = draw_review(image, candidates)
        if not cv2.imwrite(str(review_path), review, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"Could not write review image: {review_path}")
        contact_items.append((query_id, review))
        face_counts[query_id] = len(candidates)
        review_files[query_id] = review_path
        candidate_files[query_id] = candidates_path

    contact_path = output_dir / "face_review_contact_sheet.jpg"
    make_contact_sheet(contact_items, contact_path)
    manifest_path = write_run_manifest(
        output_dir / "selection_review_manifest.json",
        script_path=Path(__file__),
        method="agrishow_2022_pre_retrieval_robustness_query_selection",
        configuration={
            "seed": args.seed,
            "selection_rule": "lowest_sha256(seed,target_id,image_id)",
            "candidate_population": "frozen present images excluding original source",
            "additional_queries": args.additional_queries,
            "device": args.device,
            "det_size": args.det_size,
            "retrieval_executed": False,
        },
        inputs={
            "base_queries_csv": args.base_queries_csv,
            "base_relevance_csv": args.base_relevance_csv,
            "inventory_csv": args.inventory_csv,
        },
        result_files={
            "selected_sources": selection_path,
            "face_review_contact_sheet": contact_path,
            **{f"review_{key}": value for key, value in review_files.items()},
            **{f"candidates_{key}": value for key, value in candidate_files.items()},
        },
        extra={
            "selected_source_ids": selection["source_image_id"].astype(str).tolist(),
            "face_counts": face_counts,
            "selection_before_retrieval": True,
            "redraw_based_on_results_allowed": False,
        },
    )
    print("EX-026: SOURCES SELECTED BEFORE RETRIEVAL")
    print(selection[["query_id", "source_image_id"]].to_string(index=False))
    print(f"contact_sheet={contact_path}")
    print(f"manifest={manifest_path}")
    print("status=awaiting_manual_target_face_indices")
    return 0


def finalize_queries(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    selection_path = output_dir / "selected_sources.csv"
    manifest_path = output_dir / "selection_review_manifest.json"
    require(selection_path.is_file() and manifest_path.is_file(), "Selection review is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["git"]["dirty"] is False, "Selection review was not created from a clean tree")
    require(
        sha256_file(selection_path) == manifest["results"]["selected_sources"]["sha256"],
        "Selected sources differ from the pre-retrieval manifest",
    )

    selection = pd.read_csv(selection_path, dtype=str).fillna("")
    target_indices = pd.read_csv(args.target_indices_csv, dtype=str).fillna("")
    require(
        set(target_indices.columns) >= {"query_id", "target_face_index"},
        "Target-index CSV must contain query_id and target_face_index",
    )
    require(target_indices["query_id"].is_unique, "Target-index query IDs are not unique")
    require(
        set(target_indices["query_id"]) == set(selection["query_id"]),
        "Target indices must cover exactly the nine selected queries",
    )
    index_by_query = target_indices.set_index("query_id")["target_face_index"].astype(int)

    if args.queries_output_csv.exists() or args.relevance_output_csv.exists():
        raise FileExistsError("Frozen robustness query or relevance CSV already exists")
    query_dir = args.query_dir.resolve()
    if query_dir.exists():
        raise FileExistsError(f"Robustness query directory already exists: {query_dir}")
    query_dir.mkdir(parents=True, exist_ok=False)

    app = build_face_app(args.device, args.det_size)
    query_rows = pd.read_csv(args.base_queries_csv, dtype=str).fillna("").to_dict("records")
    crop_files: dict[str, Path] = {}
    audit_rows: list[dict[str, object]] = []
    for row in selection.itertuples(index=False):
        query_id = str(row.query_id)
        source_path = resolve_stored_path(str(row.source_image_path)).resolve()
        require(sha256_file(source_path) == str(row.source_image_sha256), f"Source hash mismatch: {query_id}")
        image = cv2.imread(str(source_path))
        if image is None:
            raise FileNotFoundError(f"Could not read source image: {source_path}")
        faces = sorted_faces(app.get(image))
        target_face_index = int(index_by_query.loc[query_id])
        require(0 <= target_face_index < len(faces), f"Invalid face index for {query_id}")
        candidates = candidate_rows(image, faces)
        selected = candidates[target_face_index]
        detector_bbox = tuple(int(selected[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
        height, width = image.shape[:2]
        crop_bbox = expanded_square_bbox(detector_bbox, width, height, args.crop_scale)
        x1, y1, x2, y2 = crop_bbox
        crop = image[y1:y2, x1:x2]
        query_path = query_dir / f"{query_id}_face.png"
        if not cv2.imwrite(str(query_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"Could not write query crop: {query_path}")
        crop_faces = app.get(crop)
        require(bool(crop_faces), f"No face found in frozen crop: {query_id}")
        largest = max(crop_faces, key=face_area)
        query_face_bbox = clamp_bbox(
            tuple(float(value) for value in largest.bbox), crop.shape[1], crop.shape[0]
        )
        expected_bbox = (
            detector_bbox[0] - x1,
            detector_bbox[1] - y1,
            detector_bbox[2] - x1,
            detector_bbox[3] - y1,
        )
        target_iou = bbox_iou(query_face_bbox, expected_bbox)
        require(target_iou >= 0.5, f"Largest crop face differs from selected target: {query_id}")
        query_rows.append(
            {
                "query_id": query_id,
                "target_id": TARGET_ID,
                "face_query_path": rel_to_root(query_path),
                "global_query_path": rel_to_root(source_path),
                "source_image_id": str(row.source_image_id),
                "exclude_image_ids": str(row.source_image_id),
                "target_face_index": target_face_index,
                "query_face_mode": "largest",
                "query_face_index": 0,
                "detector_bbox": ",".join(str(value) for value in detector_bbox),
                "crop_bbox": ",".join(str(value) for value in crop_bbox),
                "crop_scale": args.crop_scale,
                "face_query_sha256": sha256_file(query_path),
                "source_image_sha256": str(row.source_image_sha256),
                "selection_method": "manual_visual_confirmation_from_numbered_review",
            }
        )
        crop_files[query_id] = query_path
        audit_rows.append(
            {
                "query_id": query_id,
                "source_image_id": str(row.source_image_id),
                "target_face_index": target_face_index,
                "faces_in_source": len(faces),
                "faces_in_crop": len(crop_faces),
                "target_iou": target_iou,
                "crop_width": crop.shape[1],
                "crop_height": crop.shape[0],
                "face_query_sha256": sha256_file(query_path),
            }
        )

    queries = pd.DataFrame(query_rows)
    require(len(queries) == 10 and queries["query_id"].is_unique, "Expected ten unique queries")
    args.queries_output_csv.parent.mkdir(parents=True, exist_ok=True)
    queries.to_csv(args.queries_output_csv, index=False)

    base_relevance = pd.read_csv(args.base_relevance_csv, dtype=str).fillna("")
    relevance_frames: list[pd.DataFrame] = []
    for query_id in queries["query_id"].astype(str):
        frame = base_relevance.copy()
        frame["query_id"] = query_id
        relevance_frames.append(frame)
    robustness_relevance = pd.concat(relevance_frames, ignore_index=True)
    require(len(robustness_relevance) == 1240, "Expected 1,240 robustness relevance rows")
    robustness_relevance.to_csv(args.relevance_output_csv, index=False)

    audit_path = output_dir / "frozen_query_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    final_manifest_path = write_run_manifest(
        output_dir / "robustness_query_manifest.json",
        script_path=Path(__file__),
        method="agrishow_2022_ten_query_protocol_freeze",
        configuration={
            "queries": 10,
            "original_queries": 1,
            "additional_queries": 9,
            "source_selection_seed": args.seed,
            "target_selection": "manual_visual_confirmation_from_numbered_review",
            "crop_scale": args.crop_scale,
            "device": args.device,
            "det_size": args.det_size,
            "retrieval_executed": False,
        },
        inputs={
            "selection_review_manifest": manifest_path,
            "selected_sources": selection_path,
            "target_indices_csv": args.target_indices_csv,
            "base_queries_csv": args.base_queries_csv,
            "base_relevance_csv": args.base_relevance_csv,
        },
        result_files={
            "robustness_queries": args.queries_output_csv,
            "robustness_relevance": args.relevance_output_csv,
            "frozen_query_audit": audit_path,
            **{f"crop_{key}": value for key, value in crop_files.items()},
        },
        extra={
            "source_image_ids": queries["source_image_id"].astype(str).tolist(),
            "source_ids_unique": queries["source_image_id"].nunique() == 10,
            "selection_before_retrieval": True,
            "redraw_based_on_results_allowed": False,
        },
    )
    print("EX-026: TEN QUERY PAIRS FROZEN")
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print(f"queries={args.queries_output_csv.resolve()}")
    print(f"relevance={args.relevance_output_csv.resolve()}")
    print(f"manifest={final_manifest_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.additional_queries != 9:
        raise ValueError("The frozen robustness protocol requires exactly nine additional queries")
    if args.target_indices_csv is None:
        return prepare_reviews(args)
    return finalize_queries(args)


if __name__ == "__main__":
    raise SystemExit(main())
