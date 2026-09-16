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
from face_lib import build_face_app
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
    ap.add_argument("--index-dir", type=Path, default=OUTPUTS_DIR / "face_index_gallagher")
    ap.add_argument("--output-dir", type=Path, default=DATA_DIR / "evaluation")
    ap.add_argument("--query-crop-dir", type=Path, default=DATA_DIR / "query" / "gallagher")
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-011_gallagher_protocol",
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


def build_image_map(face_metadata: pd.DataFrame) -> dict[str, str]:
    image_rows = face_metadata[["image_id", "image_path"]].drop_duplicates()
    out: dict[str, str] = {}
    for row in image_rows.itertuples(index=False):
        out[basename(str(row.image_path))] = str(row.image_id)
    return out


def crop_face_from_eyes(
    image_path: Path,
    left_eye_x: float,
    left_eye_y: float,
    right_eye_x: float,
    right_eye_y: float,
    out_path: Path,
    scale: float,
) -> bool:
    img = cv2.imread(str(image_path))
    if img is None:
        return False
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
        return False
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), crop))


def crop_has_detectable_face(crop_path: Path, app) -> bool:
    img = cv2.imread(str(crop_path))
    if img is None:
        return False
    return len(app.get(img)) > 0


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.query_crop_dir.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(args.annotations_csv)
    face_metadata_path = args.index_dir / "face_metadata.csv"
    if not face_metadata_path.exists():
        raise FileNotFoundError(f"Missing face metadata: {face_metadata_path}")
    face_metadata = pd.read_csv(face_metadata_path)

    image_name_to_id = build_image_map(face_metadata)
    annotations["image_name"] = annotations["image_name"].astype(str)
    annotations["identity"] = annotations["identity"].astype(str)
    annotations["image_id"] = annotations["image_name"].map(image_name_to_id)
    annotations = annotations.dropna(subset=["image_id"]).copy()

    image_ids_by_identity: dict[str, set[str]] = {}
    for identity, group in annotations.groupby("identity"):
        image_ids_by_identity[str(identity)] = set(group["image_id"].astype(str).tolist())

    identity_counts = (
        annotations[["identity", "image_id"]]
        .drop_duplicates()
        .groupby("identity")
        .size()
        .sort_values(ascending=False, kind="mergesort")
    )

    query_rows: list[dict[str, str]] = []
    relevance_rows: list[dict[str, str | int]] = []
    crop_rows: list[dict[str, str | int]] = []
    skipped = []
    app = None if args.skip_detection_validation else build_face_app(args.device, args.det_size)

    selected_identities = [
        str(identity)
        for identity, count in identity_counts.items()
        if int(count) >= args.min_relevant + 1
    ][: args.max_identities]

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
            if crop_path.exists() and not args.overwrite:
                crop_ok = True
            else:
                crop_ok = crop_face_from_eyes(
                    resolve_stored_path(str(row.image_path)),
                    float(row.left_eye_x),
                    float(row.left_eye_y),
                    float(row.right_eye_x),
                    float(row.right_eye_y),
                    crop_path,
                    args.crop_scale,
                )
            if not crop_ok:
                skipped.append({"identity": identity, "image_name": image_name, "reason": "crop_failed"})
                continue
            if app is not None and not crop_has_detectable_face(crop_path, app):
                skipped.append(
                    {"identity": identity, "image_name": image_name, "reason": "no_face_in_crop"}
                )
                continue

            query_rows.append(
                {
                    "query_id": query_id,
                    "query_path": str(crop_path.relative_to(ROOT)).replace("\\", "/"),
                    "target_label": identity,
                    "query_type": "face",
                    "source_image_id": source_image_id,
                    "source_image_name": image_name,
                    "source_face_index": str(face_index),
                }
            )
            crop_rows.append(
                {
                    "query_id": query_id,
                    "query_path": rel_to_root(crop_path),
                    "bytes": crop_path.stat().st_size,
                    "sha256": sha256_file(crop_path),
                }
            )

            for image_id in sorted(image_ids_by_identity[identity] - {source_image_id}):
                relevance_rows.append({"query_id": query_id, "image_id": image_id, "relevant": 1})
            made_for_identity += 1

    queries_path = args.output_dir / "gallagher_face_queries.csv"
    relevance_path = args.output_dir / "gallagher_relevance.csv"
    crops_manifest_path = args.output_dir / "gallagher_query_crops_manifest.csv"
    skipped_path = args.output_dir / "gallagher_prepare_skipped.csv"
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
        args.manifest_dir / "gallagher_protocol_manifest.json",
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
        },
        inputs={
            "annotations": args.annotations_csv,
            "face_metadata": face_metadata_path,
        },
        result_files=result_files,
        extra={
            "mapped_annotation_rows": int(len(annotations)),
            "selected_identities": int(len(selected_identities)),
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
