from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
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


QUERY_ID = "agrishow2022_stress_hat_shadow_01"
TARGET_ID = "agrishow2022_person_01"
SOURCE_IMAGE_ID = "agrishow2022_52030041698"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the predefined Agrishow hat-and-shadow stress query."
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
        "--base-relevance-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_relevance.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-027_agrishow_hat_shadow_protocol",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=DATA_DIR / "query" / "agrishow_2022_stress",
    )
    parser.add_argument(
        "--queries-output-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_hat_shadow_query.csv",
    )
    parser.add_argument(
        "--relevance-output-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_hat_shadow_relevance.csv",
    )
    parser.add_argument("--target-face-index", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--crop-scale", type=float, default=2.0)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_source(args: argparse.Namespace) -> tuple[pd.Series, Path, pd.DataFrame]:
    inventory = pd.read_csv(args.inventory_csv, dtype=str).fillna("")
    source_rows = inventory[inventory["image_id"] == SOURCE_IMAGE_ID]
    require(len(source_rows) == 1, "Predefined stress source is absent from inventory")
    source = source_rows.iloc[0]
    source_path = resolve_stored_path(str(source["image_path"])).resolve()
    require(source_path.is_file(), f"Stress source file is missing: {source_path}")
    relevance = pd.read_csv(args.base_relevance_csv, dtype=str).fillna("")
    label = relevance[relevance["image_id"] == SOURCE_IMAGE_ID]
    require(len(label) == 1 and int(label.iloc[0]["relevant"]) == 1, "Stress source is not relevant")
    return source, source_path, relevance


def detect_candidates(image, faces: list) -> list[dict[str, object]]:
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


def prepare_review(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Stress protocol output already exists: {output_dir}")
    source, source_path, _ = load_source(args)
    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Could not read stress source: {source_path}")
    app = build_face_app(args.device, args.det_size)
    faces = sorted_faces(app.get(image))
    require(bool(faces), "No face detected in stress source")
    candidates = detect_candidates(image, faces)

    output_dir.mkdir(parents=True, exist_ok=False)
    candidates_path = output_dir / "face_candidates.csv"
    review_path = output_dir / "face_candidates_review.jpg"
    pd.DataFrame(candidates).to_csv(candidates_path, index=False)
    if not cv2.imwrite(
        str(review_path), draw_review(image, candidates), [cv2.IMWRITE_JPEG_QUALITY, 94]
    ):
        raise RuntimeError(f"Could not write stress review image: {review_path}")
    manifest_path = write_run_manifest(
        output_dir / "stress_selection_manifest.json",
        script_path=Path(__file__),
        method="agrishow_2022_predefined_hat_shadow_stress_selection",
        configuration={
            "query_id": QUERY_ID,
            "target_id": TARGET_ID,
            "source_image_id": SOURCE_IMAGE_ID,
            "selection_method": "manual_hat_shadow_case_named_before_retrieval",
            "device": args.device,
            "det_size": args.det_size,
            "retrieval_executed": False,
        },
        inputs={
            "inventory_csv": args.inventory_csv,
            "base_relevance_csv": args.base_relevance_csv,
            "source_image": source_path,
        },
        result_files={
            "face_candidates": candidates_path,
            "numbered_review": review_path,
        },
        extra={
            "face_count": len(faces),
            "source_image_sha256": sha256_file(source_path),
            "commons_page_url": str(source["page_url"]),
            "selection_before_retrieval": True,
        },
    )
    print("EX-027: HAT-SHADOW SOURCE FROZEN BEFORE RETRIEVAL")
    print(f"faces={len(faces)}")
    print(f"review={review_path}")
    print(f"manifest={manifest_path}")
    print("status=awaiting_manual_target_face_index")
    return 0


def finalize(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    selection_manifest_path = output_dir / "stress_selection_manifest.json"
    candidates_path = output_dir / "face_candidates.csv"
    require(selection_manifest_path.is_file() and candidates_path.is_file(), "Stress review is incomplete")
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    require(selection_manifest["git"]["dirty"] is False, "Stress selection tree was dirty")
    require(
        sha256_file(candidates_path) == selection_manifest["results"]["face_candidates"]["sha256"],
        "Stress face candidates differ from selection manifest",
    )
    source, source_path, base_relevance = load_source(args)
    require(
        sha256_file(source_path) == selection_manifest["extra"]["source_image_sha256"],
        "Stress source hash differs from selection manifest",
    )
    if args.queries_output_csv.exists() or args.relevance_output_csv.exists():
        raise FileExistsError("Frozen stress query or relevance CSV already exists")
    query_dir = args.query_dir.resolve()
    if query_dir.exists():
        raise FileExistsError(f"Stress query directory already exists: {query_dir}")

    image = cv2.imread(str(source_path))
    app = build_face_app(args.device, args.det_size)
    faces = sorted_faces(app.get(image))
    candidates = detect_candidates(image, faces)
    target_index = int(args.target_face_index)
    require(0 <= target_index < len(faces), "Stress target-face index is invalid")
    selected = candidates[target_index]
    detector_bbox = tuple(
        int(selected[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    )
    height, width = image.shape[:2]
    crop_bbox = expanded_square_bbox(detector_bbox, width, height, args.crop_scale)
    x1, y1, x2, y2 = crop_bbox
    crop = image[y1:y2, x1:x2]
    query_dir.mkdir(parents=True, exist_ok=False)
    query_path = query_dir / f"{QUERY_ID}_face.png"
    if not cv2.imwrite(str(query_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"Could not write stress query crop: {query_path}")
    crop_faces = app.get(crop)
    require(bool(crop_faces), "No face detected in stress query crop")
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
    require(target_iou >= 0.5, "Largest crop face differs from manually selected stress target")

    query_row = {
        "query_id": QUERY_ID,
        "target_id": TARGET_ID,
        "face_query_path": rel_to_root(query_path),
        "global_query_path": rel_to_root(source_path),
        "source_image_id": SOURCE_IMAGE_ID,
        "exclude_image_ids": SOURCE_IMAGE_ID,
        "target_face_index": target_index,
        "query_face_mode": "largest",
        "query_face_index": 0,
        "detector_bbox": ",".join(str(value) for value in detector_bbox),
        "crop_bbox": ",".join(str(value) for value in crop_bbox),
        "crop_scale": args.crop_scale,
        "face_query_sha256": sha256_file(query_path),
        "source_image_sha256": sha256_file(source_path),
        "selection_method": "manual_hat_shadow_case_and_numbered_face_confirmation",
    }
    args.queries_output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([query_row]).to_csv(args.queries_output_csv, index=False)
    stress_relevance = base_relevance.copy()
    stress_relevance["query_id"] = QUERY_ID
    stress_relevance.to_csv(args.relevance_output_csv, index=False)

    manifest_path = write_run_manifest(
        output_dir / "stress_query_manifest.json",
        script_path=Path(__file__),
        method="agrishow_2022_hat_shadow_query_freeze",
        configuration={
            "target_face_index": target_index,
            "target_selection": "manual_visual_confirmation_from_numbered_review",
            "crop_scale": args.crop_scale,
            "device": args.device,
            "det_size": args.det_size,
            "retrieval_executed": False,
        },
        inputs={
            "selection_manifest": selection_manifest_path,
            "face_candidates": candidates_path,
            "base_relevance_csv": args.base_relevance_csv,
        },
        result_files={
            "stress_query_csv": args.queries_output_csv,
            "stress_relevance_csv": args.relevance_output_csv,
            "face_query": query_path,
        },
        extra={
            "query_id": QUERY_ID,
            "source_image_id": SOURCE_IMAGE_ID,
            "source_excluded": True,
            "faces_in_source": len(faces),
            "faces_in_crop": len(crop_faces),
            "query_face_target_iou": target_iou,
            "selection_before_retrieval": True,
        },
    )
    print("EX-027: HAT-SHADOW QUERY FROZEN")
    print(f"target_face_index={target_index}")
    print(f"target_iou={target_iou:.6f}")
    print(f"face_query={query_path}")
    print(f"manifest={manifest_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.target_face_index is None:
        return prepare_review(args)
    return finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
