from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval.adapters.files import resolve_stored_path, sha256_file
from retrieval.adapters.manifest import write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair Gallagher face crops with source photographs.")
    parser.add_argument(
        "--face-queries-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "gallagher_final" / "gallagher_face_queries.csv",
    )
    parser.add_argument(
        "--relevance-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "gallagher_final" / "gallagher_relevance.csv",
    )
    parser.add_argument(
        "--face-index-dir",
        type=Path,
        default=OUTPUTS_DIR / "final" / "gallagher" / "face_index",
    )
    parser.add_argument(
        "--global-index-dir",
        type=Path,
        default=OUTPUTS_DIR / "final" / "gallagher" / "global_index",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DATA_DIR / "evaluation" / "gallagher_final" / "gallagher_fusion_queries.csv",
    )
    parser.add_argument(
        "--query-manifest-csv",
        type=Path,
        default=DATA_DIR
        / "evaluation"
        / "gallagher_final"
        / "gallagher_fusion_query_manifest.csv",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=OUTPUTS_DIR / "final" / "gallagher" / "paired_protocol",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_manifest_path = args.manifest_dir / "paired_protocol_manifest.json"
    for output_path in (args.output_csv, args.query_manifest_csv, protocol_manifest_path):
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output_path}")

    face_queries = pd.read_csv(args.face_queries_csv, dtype=str)
    required = {
        "query_id",
        "query_path",
        "target_label",
        "source_image_id",
        "source_image_name",
        "source_face_index",
        "query_face_mode",
        "query_target_x",
        "query_target_y",
    }
    missing = sorted(required - set(face_queries.columns))
    if missing:
        raise ValueError(f"Missing face-query columns: {missing}")
    if face_queries.empty or face_queries["query_id"].duplicated().any():
        raise ValueError("Face queries must be non-empty and have unique query IDs")

    face_metadata_path = args.face_index_dir / "face_metadata.csv"
    global_metadata_path = args.global_index_dir / "global_metadata.csv"
    face_metadata = pd.read_csv(face_metadata_path, dtype=str)
    global_metadata = pd.read_csv(global_metadata_path, dtype=str)
    face_image_ids = set(face_metadata["image_id"].astype(str))
    global_paths = (
        global_metadata[["image_id", "image_path"]]
        .drop_duplicates("image_id")
        .set_index("image_id")["image_path"]
        .astype(str)
        .to_dict()
    )

    relevance = pd.read_csv(args.relevance_csv, dtype={"query_id": str, "image_id": str})
    relevant_query_ids = set(relevance.loc[relevance["relevant"].astype(int) == 1, "query_id"])

    paired_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for query in face_queries.itertuples(index=False):
        query_id = str(query.query_id)
        source_image_id = str(query.source_image_id)
        query_face_mode = str(query.query_face_mode)
        if query_face_mode != "annotated_eye_midpoint":
            raise ValueError(
                f"Query {query_id} does not use annotated-eye target selection: {query_face_mode}"
            )
        if source_image_id not in face_image_ids:
            raise ValueError(f"Source is absent from the face index: {source_image_id}")
        if source_image_id not in global_paths:
            raise ValueError(f"Source is absent from the global index: {source_image_id}")
        if query_id not in relevant_query_ids:
            raise ValueError(f"Query has no relevance entries: {query_id}")

        face_query_path = str(query.query_path)
        global_query_path = str(global_paths[source_image_id])
        resolved_face = resolve_stored_path(face_query_path)
        resolved_global = resolve_stored_path(global_query_path)
        if not resolved_face.is_file() or not resolved_global.is_file():
            raise FileNotFoundError(f"Missing paired input for {query_id}")

        paired_rows.append(
            {
                "query_id": query_id,
                "face_query_path": face_query_path,
                "global_query_path": global_query_path,
                "target_label": str(query.target_label),
                "query_type": "face_global_paired",
                "source_image_id": source_image_id,
                "exclude_image_ids": source_image_id,
                "source_image_name": str(query.source_image_name),
                "source_face_index": int(query.source_face_index),
                "query_face_mode": query_face_mode,
                "query_target_x": float(query.query_target_x),
                "query_target_y": float(query.query_target_y),
            }
        )
        manifest_rows.append(
            {
                "query_id": query_id,
                "target_label": str(query.target_label),
                "source_image_id": source_image_id,
                "face_query_path": face_query_path,
                "face_query_sha256": sha256_file(resolved_face),
                "global_query_path": global_query_path,
                "global_query_sha256": sha256_file(resolved_global),
                "query_face_mode": query_face_mode,
                "query_target_x": float(query.query_target_x),
                "query_target_y": float(query.query_target_y),
            }
        )

    paired = pd.DataFrame(paired_rows)
    query_manifest = pd.DataFrame(manifest_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_csv, index=False)
    query_manifest.to_csv(args.query_manifest_csv, index=False)

    manifest_path = write_run_manifest(
        protocol_manifest_path,
        script_path=Path(__file__),
        method="gallagher_paired_face_global_protocol",
        configuration={
            "selection_source": "corrected_gallagher_queries",
            "query_face_selection": "detected bbox containing the annotated-eye midpoint",
        },
        inputs={
            "face_queries_csv": args.face_queries_csv,
            "relevance_csv": args.relevance_csv,
            "face_metadata": face_metadata_path,
            "global_metadata": global_metadata_path,
            "face_index_manifest": args.face_index_dir / "face_index_manifest.json",
            "global_index_manifest": args.global_index_dir / "global_index_manifest.json",
        },
        result_files={
            "paired_queries": args.output_csv,
            "paired_query_hashes": args.query_manifest_csv,
        },
        extra={
            "queries": int(len(paired)),
            "unique_sources": int(paired["source_image_id"].nunique()),
            "unique_target_identities": int(paired["target_label"].nunique()),
        },
    )
    print(f"[OK] paired queries: {len(paired)}")
    print(f"[OK] {args.output_csv}")
    print(f"[OK] {args.query_manifest_csv}")
    print(f"[OK] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
