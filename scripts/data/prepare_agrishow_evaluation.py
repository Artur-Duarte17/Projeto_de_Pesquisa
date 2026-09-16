from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval_common import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen Agrishow relevance table for paired evaluation."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DATA_DIR / "annotations" / "agrishow_2022_presence_review.csv",
    )
    parser.add_argument(
        "--annotations-manifest",
        type=Path,
        default=OUTPUTS_DIR
        / "experiments"
        / "ex-020_agrishow_annotation_review"
        / "annotations_frozen_manifest.json",
    )
    parser.add_argument(
        "--queries-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_queries.csv",
    )
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=OUTPUTS_DIR
        / "experiments"
        / "ex-021_agrishow_query_selection"
        / "face_query_manifest.json",
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
        "--relevance-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "agrishow_2022_relevance.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-023_agrishow_protocol",
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_existing(manifest_path: Path) -> dict:
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "relevance_frozen", "Existing relevance is not frozen")
    relevance_path = Path(manifest["relevance_csv_path"])
    require(relevance_path.is_file(), f"Missing frozen relevance: {relevance_path}")
    require(
        sha256_file(relevance_path) == manifest["relevance_csv_sha256"],
        "Frozen relevance hash mismatch",
    )
    return manifest


def main() -> int:
    args = parse_args()
    annotations_path = args.annotations.resolve()
    annotations_manifest_path = args.annotations_manifest.resolve()
    queries_path = args.queries_csv.resolve()
    query_manifest_path = args.query_manifest.resolve()
    face_dir = args.face_index_dir.resolve()
    global_dir = args.global_index_dir.resolve()
    relevance_path = args.relevance_csv.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "relevance_manifest.json"

    if manifest_path.exists() and not args.overwrite:
        existing = validate_existing(manifest_path)
        print("EX-023: FROZEN RELEVANCE PRESERVED AND VALIDATED")
        print(f"relevant_after_source_exclusion={existing['relevant_after_source_exclusion']}")
        print(f"manifest={manifest_path}")
        return 0
    if relevance_path.exists() and not args.overwrite:
        raise FileExistsError(f"Relevance CSV already exists without a frozen manifest: {relevance_path}")

    annotations_manifest = read_json(annotations_manifest_path)
    query_manifest = read_json(query_manifest_path)
    require(annotations_manifest.get("status") == "approved_and_frozen", "EX-020 is not frozen")
    require(query_manifest.get("status") == "query_pair_frozen", "EX-021 is not frozen")
    require(
        sha256_file(annotations_path) == annotations_manifest["annotations_sha256"],
        "EX-020 annotation hash mismatch",
    )
    require(
        sha256_file(queries_path) == query_manifest["query_csv_sha256"],
        "EX-021 query CSV hash mismatch",
    )

    annotations = pd.read_csv(annotations_path, dtype=str).fillna("")
    queries = pd.read_csv(queries_path, dtype=str).fillna("")
    face_metadata = pd.read_csv(face_dir / "face_metadata.csv", dtype={"image_id": str})
    global_metadata = pd.read_csv(global_dir / "global_metadata.csv", dtype={"image_id": str})
    require(len(annotations) == 124, f"Expected 124 annotations, found {len(annotations)}")
    require(len(queries) == 1, f"Expected one frozen query, found {len(queries)}")
    require(annotations["image_id"].is_unique, "Annotation image IDs are not unique")
    require(global_metadata["image_id"].is_unique, "Global image IDs are not unique")
    annotation_ids = set(annotations["image_id"])
    global_ids = set(global_metadata["image_id"])
    require(annotation_ids == global_ids, "Annotation IDs differ from the global gallery")

    query = queries.iloc[0]
    query_id = str(query["query_id"])
    target_id = str(query["target_id"])
    source_image_id = str(query["source_image_id"])
    require(source_image_id == query_manifest["source_image_id"], "Query source mismatch")
    require(source_image_id in annotation_ids, "Query source is absent from annotations")
    require(set(annotations["target_id"]) == {target_id}, "Annotation target differs from query")
    require(set(annotations["presence"]) <= {"present", "absent"}, "Unresolved annotations remain")

    relevance = pd.DataFrame(
        {
            "query_id": query_id,
            "image_id": annotations["image_id"].astype(str),
            "relevant": (annotations["presence"] == "present").astype(int),
        }
    )
    relevant_ids = set(relevance.loc[relevance["relevant"] == 1, "image_id"])
    require(len(relevant_ids) == 91, f"Expected 91 relevant images, found {len(relevant_ids)}")
    require(source_image_id in relevant_ids, "Frozen source is not marked relevant")

    face_ids = set(face_metadata["image_id"].astype(str))
    missing_face_ids = sorted(global_ids - face_ids)
    relevant_missing_face_ids = sorted(relevant_ids - face_ids)
    require(
        not relevant_missing_face_ids,
        f"Relevant images without a detected face: {relevant_missing_face_ids}",
    )

    relevance_path.parent.mkdir(parents=True, exist_ok=True)
    relevance.to_csv(relevance_path, index=False)
    manifest = {
        "execution_id": "EX-023",
        "stage": "relevance_preparation",
        "status": "relevance_frozen",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_id": query_id,
        "target_id": target_id,
        "source_image_id": source_image_id,
        "annotations_path": str(annotations_path),
        "annotations_sha256": sha256_file(annotations_path),
        "queries_csv_path": str(queries_path),
        "queries_csv_sha256": sha256_file(queries_path),
        "relevance_csv_path": str(relevance_path),
        "relevance_csv_sha256": sha256_file(relevance_path),
        "gallery_images": len(global_ids),
        "eligible_gallery_images": len(global_ids) - 1,
        "relevant_before_source_exclusion": len(relevant_ids),
        "relevant_after_source_exclusion": len(relevant_ids - {source_image_id}),
        "images_without_detected_face": len(missing_face_ids),
        "image_ids_without_detected_face": missing_face_ids,
        "relevant_images_without_detected_face": len(relevant_missing_face_ids),
        "retrieval_executed": False,
    }
    write_json(manifest_path, manifest)

    print("EX-023: RELEVANCE PREPARED AND FROZEN")
    print(f"gallery_images={len(global_ids)} eligible_after_exclusion={len(global_ids) - 1}")
    print(f"relevant_before_exclusion={len(relevant_ids)}")
    print(f"relevant_after_exclusion={len(relevant_ids - {source_image_id})}")
    print(f"relevant_without_detected_face={len(relevant_missing_face_ids)}")
    print(f"relevance={relevance_path}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
