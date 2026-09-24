from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import DATA_DIR, OUTPUTS_DIR
from face_lib import bbox_string, build_face_app, pick_query_face
from retrieval_common import rel_to_root, resolve_stored_path, sha256_file
from run_manifest import write_run_manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prepare Gallagher face queries and relevance CSVs for evaluation."
    )
    ap.add_argument(
        "--annotations-csv",
        type=Path,
        default=DATA_DIR / "raw" / "gallagher" / "metadata" / "face_annotations.csv",
    )
    ap.add_argument(
        "--index-dir",
        type=Path,
        default=OUTPUTS_DIR / "final" / "gallagher" / "face_index",
    )
    ap.add_argument(
        "--gallery-index-dir",
        type=Path,
        default=OUTPUTS_DIR / "final" / "gallagher" / "global_index",
        help="Global gallery whose photographs define Gallagher eligibility.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "evaluation" / "gallagher_final",
    )
    ap.add_argument(
        "--query-crop-dir",
        type=Path,
        default=DATA_DIR / "query" / "gallagher_final",
    )
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        default=OUTPUTS_DIR / "final" / "gallagher" / "face_protocol",
    )
    ap.add_argument("--max-identities", type=int, default=20)
    ap.add_argument("--queries-per-identity", type=int, default=1)
    ap.add_argument("--min-relevant", type=int, default=3)
    ap.add_argument("--crop-scale", type=float, default=6.0)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--skip-detection-validation", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def basename(path_value: str) -> str:
    return Path(str(path_value).replace("\\", "/")).name


def build_image_map(gallery_metadata: pd.DataFrame) -> dict[str, str]:
    required = {"image_id", "image_path"}
    missing = sorted(required - set(gallery_metadata.columns))
    if missing:
        raise ValueError(f"Missing gallery-metadata columns: {missing}")
    image_rows = gallery_metadata[["image_id", "image_path"]].drop_duplicates()
    image_rows = image_rows.assign(
        image_name=image_rows["image_path"].astype(str).map(basename)
    )
    duplicate_names = sorted(
        image_rows.loc[image_rows["image_name"].duplicated(keep=False), "image_name"].unique()
    )
    if duplicate_names:
        raise ValueError(f"Ambiguous gallery basenames: {duplicate_names[:3]}")
    out: dict[str, str] = {}
    for row in image_rows.itertuples(index=False):
        out[str(row.image_name)] = str(row.image_id)
    return out


def map_annotations_to_gallery(
    annotations: pd.DataFrame,
    gallery_metadata: pd.DataFrame,
) -> pd.DataFrame:
    required = {"image_name", "identity"}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"Missing Gallagher annotation columns: {missing}")
    mapped = annotations.copy()
    mapped["image_name"] = mapped["image_name"].astype(str)
    mapped["identity"] = mapped["identity"].astype(str)
    mapped["image_id"] = mapped["image_name"].map(build_image_map(gallery_metadata))
    return mapped.dropna(subset=["image_id"]).copy()


def build_image_ids_by_identity(annotations: pd.DataFrame) -> dict[str, set[str]]:
    image_ids_by_identity: dict[str, set[str]] = {}
    for identity, group in annotations.groupby("identity"):
        image_ids_by_identity[str(identity)] = set(group["image_id"].astype(str).tolist())
    return image_ids_by_identity


def relevant_image_ids_for_query(
    image_ids_by_identity: dict[str, set[str]],
    identity: str,
    source_image_id: str,
) -> list[str]:
    return sorted(image_ids_by_identity[str(identity)] - {str(source_image_id)})


def prepare_crop_from_eyes(
    image_path: Path,
    left_eye_x: float,
    left_eye_y: float,
    right_eye_x: float,
    right_eye_y: float,
    out_path: Path,
    scale: float,
    overwrite: bool,
) -> tuple[int, int, int, int] | None:
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    eye_dx = right_eye_x - left_eye_x
    eye_dy = right_eye_y - left_eye_y
    eye_dist = max((eye_dx**2 + eye_dy**2) ** 0.5, 1.0)
    cx = (left_eye_x + right_eye_x) / 2.0
    cy = (left_eye_y + right_eye_y) / 2.0

    half = max(80.0, eye_dist * scale / 2.0)
    x1 = int(max(0, cx - half))
    x2 = int(min(w, cx + half))
    y1 = int(max(0, cy - half * 0.9))
    y2 = int(min(h, cy + half * 1.25))

    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if overwrite or not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), crop):
            return None
    return x1, y1, x2, y2


def annotated_target_bbox(
    crop_path: Path,
    app,
    target_x: float,
    target_y: float,
) -> str | None:
    img = cv2.imread(str(crop_path))
    if img is None:
        return None
    face = pick_query_face(
        app.get(img),
        mode="annotated_eye_midpoint",
        target_x=target_x,
        target_y=target_y,
    )
    return None if face is None else bbox_string(face)


def main() -> int:
    args = parse_args()
    queries_path = args.output_dir / "gallagher_face_queries.csv"
    relevance_path = args.output_dir / "gallagher_relevance.csv"
    crops_manifest_path = args.output_dir / "gallagher_query_crops_manifest.csv"
    skipped_path = args.output_dir / "gallagher_prepare_skipped.csv"
    manifest_path = args.manifest_dir / "gallagher_protocol_manifest.json"
    protected_outputs = [
        queries_path,
        relevance_path,
        crops_manifest_path,
        skipped_path,
        manifest_path,
    ]
    existing_outputs = [path for path in protected_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        existing_list = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Corrected protocol outputs already exist: {existing_list}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.query_crop_dir.mkdir(parents=True, exist_ok=True)

    raw_annotations = pd.read_csv(args.annotations_csv)
    face_metadata_path = args.index_dir / "face_metadata.csv"
    gallery_metadata_path = args.gallery_index_dir / "global_metadata.csv"
    if not face_metadata_path.exists():
        raise FileNotFoundError(f"Missing face metadata: {face_metadata_path}")
    if not gallery_metadata_path.exists():
        raise FileNotFoundError(f"Missing global gallery metadata: {gallery_metadata_path}")
    face_metadata = pd.read_csv(face_metadata_path)
    gallery_metadata = pd.read_csv(gallery_metadata_path)
    gallery_image_ids = set(gallery_metadata["image_id"].astype(str))
    face_image_ids = set(face_metadata["image_id"].astype(str))
    face_images_outside_gallery = sorted(face_image_ids - gallery_image_ids)
    if face_images_outside_gallery:
        raise ValueError(
            "Face-index images are absent from the eligible global gallery: "
            f"{face_images_outside_gallery[:3]}"
        )

    annotations = map_annotations_to_gallery(raw_annotations, gallery_metadata)
    unmapped_annotation_rows = int(len(raw_annotations) - len(annotations))
    image_ids_by_identity = build_image_ids_by_identity(annotations)

    identity_counts = (
        annotations[["identity", "image_id"]]
        .drop_duplicates()
        .groupby("identity")
        .size()
        .sort_values(ascending=False, kind="mergesort")
    )

    query_rows: list[dict[str, object]] = []
    relevance_rows: list[dict[str, str | int]] = []
    crop_rows: list[dict[str, object]] = []
    skipped = []
    app = None if args.skip_detection_validation else build_face_app(args.device, args.det_size)

    selected_identities = [
        str(identity)
        for identity, count in identity_counts.items()
        if int(count) >= args.min_relevant + 1
    ][: args.max_identities]
    if len(selected_identities) != args.max_identities:
        raise RuntimeError(
            f"Expected {args.max_identities} eligible identities, found {len(selected_identities)}"
        )

    for identity in selected_identities:
        candidates = (
            annotations[annotations["identity"] == identity]
            .sort_values(["image_name", "face_index"], kind="mergesort")
            .copy()
        )
        candidates["_image_relevant_count"] = candidates["image_id"].map(
            lambda image_id: len(image_ids_by_identity[identity] - {str(image_id)})
        )
        candidates = candidates[candidates["_image_relevant_count"] >= args.min_relevant]
        made_for_identity = 0

        for row in candidates.itertuples(index=False):
            if made_for_identity >= args.queries_per_identity:
                break
            source_image_id = str(row.image_id)
            image_name = str(row.image_name)
            face_index = int(row.face_index)
            query_id = f"gallagher_id{identity}_q{made_for_identity + 1}"
            crop_path = args.query_crop_dir / f"{query_id}_{Path(image_name).stem}_f{face_index}.jpg"
            left_eye_x = float(row.left_eye_x)
            left_eye_y = float(row.left_eye_y)
            right_eye_x = float(row.right_eye_x)
            right_eye_y = float(row.right_eye_y)
            crop_bounds = prepare_crop_from_eyes(
                resolve_stored_path(str(row.image_path)),
                left_eye_x,
                left_eye_y,
                right_eye_x,
                right_eye_y,
                crop_path,
                args.crop_scale,
                args.overwrite,
            )
            if crop_bounds is None:
                skipped.append({"identity": identity, "image_name": image_name, "reason": "crop_failed"})
                continue
            crop_x1, crop_y1, _, _ = crop_bounds
            query_target_x = (left_eye_x + right_eye_x) / 2.0 - crop_x1
            query_target_y = (left_eye_y + right_eye_y) / 2.0 - crop_y1
            validation_bbox = ""
            if app is not None:
                validation_bbox = annotated_target_bbox(
                    crop_path,
                    app,
                    query_target_x,
                    query_target_y,
                ) or ""
                if not validation_bbox:
                    skipped.append(
                        {
                            "identity": identity,
                            "image_name": image_name,
                            "reason": "annotated_target_not_detected_in_crop",
                        }
                    )
                    continue

            query_rows.append(
                {
                    "query_id": query_id,
                    "query_path": rel_to_root(crop_path),
                    "target_label": identity,
                    "query_type": "face",
                    "source_image_id": source_image_id,
                    "source_image_name": image_name,
                    "source_face_index": str(face_index),
                    "query_face_mode": "annotated_eye_midpoint",
                    "query_target_x": query_target_x,
                    "query_target_y": query_target_y,
                    "validation_bbox": validation_bbox,
                }
            )
            crop_rows.append(
                {
                    "query_id": query_id,
                    "query_path": rel_to_root(crop_path),
                    "bytes": crop_path.stat().st_size,
                    "sha256": sha256_file(crop_path),
                    "query_face_mode": "annotated_eye_midpoint",
                    "query_target_x": query_target_x,
                    "query_target_y": query_target_y,
                    "validation_bbox": validation_bbox,
                }
            )

            for image_id in relevant_image_ids_for_query(
                image_ids_by_identity,
                identity,
                source_image_id,
            ):
                relevance_rows.append({"query_id": query_id, "image_id": image_id, "relevant": 1})
            made_for_identity += 1

    expected_queries = args.max_identities * args.queries_per_identity
    if len(query_rows) != expected_queries:
        raise RuntimeError(
            f"Expected {expected_queries} annotated-target queries, generated {len(query_rows)}"
        )

    pd.DataFrame(query_rows).to_csv(queries_path, index=False)
    pd.DataFrame(relevance_rows).to_csv(relevance_path, index=False)
    pd.DataFrame(crop_rows).to_csv(crops_manifest_path, index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(skipped_path, index=False)
    elif skipped_path.exists():
        skipped_path.unlink()

    result_files = {
        "queries": queries_path,
        "relevance": relevance_path,
        "query_crops_manifest": crops_manifest_path,
    }
    if skipped_path.exists():
        result_files["skipped"] = skipped_path
    manifest_path = write_run_manifest(
        manifest_path,
        script_path=Path(__file__),
        method="gallagher_face_protocol",
        configuration={
            "max_identities": args.max_identities,
            "queries_per_identity": args.queries_per_identity,
            "min_relevant": args.min_relevant,
            "crop_scale": args.crop_scale,
            "device": args.device,
            "det_size": args.det_size,
            "detection_validation": not args.skip_detection_validation,
            "selection": "identity frequency descending, then image name and face index",
            "relevance_source": "all annotated photographs in the eligible global gallery",
            "query_face_selection": "detected bbox containing the annotated-eye midpoint",
            "query_face_tie_break": "nearest bbox center, then smallest area, then bbox coordinates",
        },
        inputs={
            "annotations": args.annotations_csv,
            "face_metadata": face_metadata_path,
            "gallery_metadata": gallery_metadata_path,
        },
        result_files=result_files,
        extra={
            "mapped_annotation_rows": int(len(annotations)),
            "unmapped_gallery_annotation_rows": unmapped_annotation_rows,
            "eligible_gallery_images": int(len(gallery_image_ids)),
            "gallery_images_without_detected_faces": int(
                len(gallery_image_ids - face_image_ids)
            ),
            "selected_identities": int(len(selected_identities)),
            "expected_queries": int(expected_queries),
            "queries": int(len(query_rows)),
            "relevance_rows": int(len(relevance_rows)),
            "skipped_crops": int(len(skipped)),
        },
    )

    print(f"[OK] queries: {queries_path}")
    print(f"[OK] relevance: {relevance_path}")
    print(f"[OK] crop hashes: {crops_manifest_path}")
    print(f"[OK] query crops: {args.query_crop_dir}")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] identities selected: {len(selected_identities)}")
    print(f"[OK] queries generated: {len(query_rows)}")
    print(f"[OK] relevance rows: {len(relevance_rows)}")
    if skipped:
        print(f"[WARN] skipped crops: {len(skipped)}")
    return 0 if query_rows and relevance_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
