from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from face_lib import build_face_app, face_area, sorted_faces
from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval_common import rel_to_root, sha256_file


DEFAULT_SELECTION_MANIFEST = (
    OUTPUTS_DIR
    / "experiments"
    / "ex-021_agrishow_query_selection"
    / "query_selection_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    OUTPUTS_DIR / "experiments" / "ex-021_agrishow_query_selection"
)
DEFAULT_QUERY_DIR = DATA_DIR / "query" / "agrishow_2022"
DEFAULT_QUERY_CSV = DATA_DIR / "evaluation" / "agrishow_2022_queries.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and number every face in the frozen Agrishow source; "
            "optionally freeze a lossless crop selected by its review index."
        )
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query-dir", type=Path, default=DEFAULT_QUERY_DIR)
    parser.add_argument("--query-csv", type=Path, default=DEFAULT_QUERY_CSV)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument(
        "--target-face-index",
        type=int,
        default=None,
        help="Zero-based number drawn in the review image.",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=2.0,
        help="Final crop width and height relative to the detected face box.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clamp_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(1, min(width, int(round(x2)))),
        max(1, min(height, int(round(y2)))),
    )


def expanded_square_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    scale: float,
) -> tuple[int, int, int, int]:
    if scale < 1.0:
        raise ValueError("crop-scale must be at least 1.0")
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * scale
    out_x1 = int(round(cx - side / 2.0))
    out_y1 = int(round(cy - side / 2.0))
    out_x2 = int(round(cx + side / 2.0))
    out_y2 = int(round(cy + side / 2.0))

    if out_x1 < 0:
        out_x2 -= out_x1
        out_x1 = 0
    if out_y1 < 0:
        out_y2 -= out_y1
        out_y1 = 0
    if out_x2 > width:
        out_x1 -= out_x2 - width
        out_x2 = width
    if out_y2 > height:
        out_y1 -= out_y2 - height
        out_y2 = height

    out_x1 = max(0, out_x1)
    out_y1 = max(0, out_y1)
    if out_x2 <= out_x1 or out_y2 <= out_y1:
        raise RuntimeError("Expanded crop is empty after clipping to the image.")
    return out_x1, out_y1, out_x2, out_y2


def bbox_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    intersection_width = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def draw_review(image, candidates: list[dict]) -> object:
    review = image.copy()
    font_scale = max(1.0, min(image.shape[:2]) / 1600.0)
    thickness = max(3, int(round(font_scale * 3)))
    for candidate in candidates:
        index = int(candidate["face_index"])
        x1, y1, x2, y2 = (
            int(candidate["bbox_x1"]),
            int(candidate["bbox_y1"]),
            int(candidate["bbox_x2"]),
            int(candidate["bbox_y2"]),
        )
        color = (0, 255, 255)
        cv2.rectangle(review, (x1, y1), (x2, y2), color, thickness)
        label = str(index)
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y1 = max(0, y1 - label_h - baseline - 12)
        label_y2 = label_y1 + label_h + baseline + 12
        cv2.rectangle(
            review,
            (x1, label_y1),
            (x1 + label_w + 20, label_y2),
            color,
            -1,
        )
        cv2.putText(
            review,
            label,
            (x1 + 10, label_y2 - baseline - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return review


def write_candidates_csv(path: Path, candidates: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)


def write_query_csv(path: Path, row: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def require_recorded_hash(path_value: str, expected_sha256: str, label: str) -> None:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen {label}: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Frozen {label} hash mismatch: expected {expected_sha256}, "
            f"found {actual_sha256}."
        )


def validate_review_manifest(review_manifest_path: Path) -> dict:
    review = read_json(review_manifest_path)
    require_recorded_hash(
        review["candidates_csv_path"],
        review["candidates_csv_sha256"],
        "candidate CSV",
    )
    require_recorded_hash(
        review["review_image_path"],
        review["review_image_sha256"],
        "numbered review image",
    )
    return review


def validate_frozen_query(final_manifest_path: Path) -> dict:
    frozen = read_json(final_manifest_path)
    if frozen.get("status") != "query_pair_frozen":
        raise RuntimeError("Existing EX-021 query pair is not marked as frozen.")
    require_recorded_hash(
        frozen["face_query_path"],
        frozen["face_query_sha256"],
        "face query",
    )
    require_recorded_hash(
        frozen["query_csv_path"],
        frozen["query_csv_sha256"],
        "paired-query CSV",
    )
    review_manifest_path = Path(frozen["review_manifest_path"])
    require_recorded_hash(
        str(review_manifest_path),
        frozen["review_manifest_sha256"],
        "review manifest",
    )
    validate_review_manifest(review_manifest_path)
    return frozen


def main() -> int:
    args = parse_args()
    selection_path = args.selection_manifest.resolve()
    selection = read_json(selection_path)
    if selection.get("status") != "source_selected_and_frozen":
        raise RuntimeError("The EX-021 random source selection is not frozen.")

    source_path = Path(selection["source_image_path"]).resolve()
    if sha256_file(source_path) != selection["source_image_sha256"]:
        raise RuntimeError("Source image hash differs from the frozen selection.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "face_candidates.csv"
    review_path = output_dir / "face_candidates_review.jpg"
    review_manifest_path = output_dir / "face_candidate_review_manifest.json"
    final_manifest_path = output_dir / "face_query_manifest.json"
    query_dir = args.query_dir.resolve()
    query_path = query_dir / "agrishow2022_query_01_face.png"
    query_csv_path = args.query_csv.resolve()

    if final_manifest_path.exists() and not args.overwrite:
        frozen = validate_frozen_query(final_manifest_path)
        print("EX-021: FROZEN QUERY PAIR PRESERVED AND VALIDATED")
        print(f"target_face_index={frozen['target_face_index']}")
        print(f"face_query={frozen['face_query_path']}")
        print(f"manifest={final_manifest_path}")
        return 0
    if args.target_face_index is None and review_manifest_path.exists() and not args.overwrite:
        review = validate_review_manifest(review_manifest_path)
        print("EX-021: EXISTING FACE REVIEW PRESERVED AND VALIDATED")
        print(f"faces={review['face_count']}")
        print(f"review={review['review_image_path']}")
        return 0
    protected_outputs = (query_path, query_csv_path, final_manifest_path)
    if not args.overwrite and any(path.exists() for path in protected_outputs):
        existing = ", ".join(str(path) for path in protected_outputs if path.exists())
        raise FileExistsError(f"Frozen query output already exists: {existing}")

    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Could not read source image: {source_path}")

    app = build_face_app(args.device, args.det_size)
    faces = sorted_faces(app.get(image))
    if not faces:
        raise RuntimeError("No faces were detected in the selected source image.")

    height, width = image.shape[:2]
    candidates: list[dict] = []
    for index, face in enumerate(faces):
        x1, y1, x2, y2 = clamp_bbox(tuple(float(v) for v in face.bbox), width, height)
        candidates.append(
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

    write_candidates_csv(candidates_path, candidates)
    if not cv2.imwrite(str(review_path), draw_review(image, candidates), [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Could not write review image: {review_path}")

    review_manifest = {
        "execution_id": "EX-021",
        "stage": "face_candidate_review",
        "status": "candidates_detected",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_image_id": selection["source_image_id"],
        "source_image_path": str(source_path),
        "source_image_sha256": selection["source_image_sha256"],
        "image_width": width,
        "image_height": height,
        "device_requested": args.device,
        "detector": "insightface_buffalo_l",
        "det_size": args.det_size,
        "face_count": len(candidates),
        "candidate_order": "top_to_bottom_then_left_to_right",
        "candidates_csv_path": str(candidates_path),
        "candidates_csv_sha256": sha256_file(candidates_path),
        "review_image_path": str(review_path),
        "review_image_sha256": sha256_file(review_path),
    }
    write_json(review_manifest_path, review_manifest)

    print("EX-021: FACE CANDIDATES DETECTED")
    print(f"faces={len(candidates)}")
    print(f"review={review_path}")
    print(f"candidates={candidates_path}")

    if args.target_face_index is None:
        print("status=awaiting_target_face_index")
        return 0
    if args.target_face_index < 0 or args.target_face_index >= len(faces):
        raise ValueError(
            f"target-face-index must be between 0 and {len(faces) - 1}."
        )

    selected = candidates[args.target_face_index]
    detector_bbox = (
        int(selected["bbox_x1"]),
        int(selected["bbox_y1"]),
        int(selected["bbox_x2"]),
        int(selected["bbox_y2"]),
    )
    crop_bbox = expanded_square_bbox(detector_bbox, width, height, args.crop_scale)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        raise RuntimeError("Target crop is empty.")
    query_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(query_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"Could not write lossless query crop: {query_path}")

    crop_faces = app.get(crop)
    if not crop_faces:
        raise RuntimeError("The selected lossless crop contains no detectable face.")
    query_face = max(crop_faces, key=face_area)
    query_face_bbox = clamp_bbox(
        tuple(float(value) for value in query_face.bbox),
        crop.shape[1],
        crop.shape[0],
    )
    expected_target_bbox = (
        detector_bbox[0] - crop_x1,
        detector_bbox[1] - crop_y1,
        detector_bbox[2] - crop_x1,
        detector_bbox[3] - crop_y1,
    )
    target_iou = bbox_iou(query_face_bbox, expected_target_bbox)
    if target_iou < 0.5:
        raise RuntimeError(
            "Largest face in the crop does not match the manually selected target "
            f"(IoU={target_iou:.4f})."
        )

    query_row: dict[str, str | int | float] = {
        "query_id": "agrishow2022_query_01",
        "target_id": selection["target_id"],
        "face_query_path": rel_to_root(query_path),
        "global_query_path": rel_to_root(source_path),
        "source_image_id": selection["source_image_id"],
        "exclude_image_ids": selection["source_image_id"],
        "target_face_index": args.target_face_index,
        "query_face_mode": "largest",
        "query_face_index": 0,
        "detector_bbox": ",".join(str(value) for value in detector_bbox),
        "crop_bbox": ",".join(str(value) for value in crop_bbox),
        "crop_scale": args.crop_scale,
        "face_query_sha256": sha256_file(query_path),
        "source_image_sha256": selection["source_image_sha256"],
        "selection_method": "manual_visual_confirmation_from_numbered_review",
    }
    write_query_csv(query_csv_path, query_row)

    final_manifest = {
        "execution_id": "EX-021",
        "stage": "paired_query_freeze",
        "status": "query_pair_frozen",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_selection_manifest_path": str(selection_path),
        "parent_selection_manifest_sha256": sha256_file(selection_path),
        "source_image_id": selection["source_image_id"],
        "source_image_path": str(source_path),
        "source_image_sha256": selection["source_image_sha256"],
        "exclude_image_ids": selection["exclude_image_ids"],
        "target_face_index": args.target_face_index,
        "target_selection_method": "manual_visual_confirmation_from_numbered_review",
        "detector": "insightface_buffalo_l",
        "det_size": args.det_size,
        "device_requested": args.device,
        "detector_bbox": list(detector_bbox),
        "crop_bbox": list(crop_bbox),
        "crop_scale": args.crop_scale,
        "crop_encoding": "lossless_png_from_source_pixels",
        "query_face_mode": "largest",
        "query_face_bbox": list(query_face_bbox),
        "query_face_target_iou": target_iou,
        "face_query_path": str(query_path),
        "face_query_sha256": sha256_file(query_path),
        "face_query_width": int(crop.shape[1]),
        "face_query_height": int(crop.shape[0]),
        "faces_detected_in_query_crop": len(crop_faces),
        "global_query_path": str(source_path),
        "query_csv_path": str(query_csv_path),
        "query_csv_sha256": sha256_file(query_csv_path),
        "review_manifest_path": str(review_manifest_path),
        "review_manifest_sha256": sha256_file(review_manifest_path),
        "retrieval_executed": False,
    }
    write_json(final_manifest_path, final_manifest)

    print("EX-021: PAIRED QUERY FROZEN")
    print(f"target_face_index={args.target_face_index}")
    print(f"detector_bbox={detector_bbox}")
    print(f"crop_bbox={crop_bbox}")
    print(f"face_query={query_path}")
    print(f"query_csv={query_csv_path}")
    print(f"manifest={final_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
