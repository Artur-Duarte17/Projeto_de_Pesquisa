from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval.adapters.files import resolve_stored_path
from retrieval.adapters.manifest import write_run_manifest


QUERY_COLUMNS = (
    "query_id",
    "query_path",
    "target_label",
    "query_type",
    "source_image_id",
    "exclude_image_ids",
)
RELEVANCE_COLUMNS = ("query_id", "image_id", "relevant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze deterministic LFW face-retrieval queries and relevance."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=OUTPUTS_DIR / "final" / "lfw" / "face_index",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "evaluation" / "lfw_final",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=OUTPUTS_DIR / "final" / "lfw" / "protocol",
    )
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--min-images", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selection_key(seed: int, identity: str, image_id: str) -> str:
    payload = f"{seed}\0{identity}\0{image_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def unique_indexed_images(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"image_id", "image_path", "sha256", "identity"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Missing columns in face metadata: {missing}")

    images = metadata[list(sorted(required))].copy()
    for column in required:
        if images[column].isna().any():
            raise ValueError(f"Null values found in required column: {column}")
        images[column] = images[column].astype(str)

    conflicting = images.groupby("image_id").agg(
        identities=("identity", "nunique"),
        paths=("image_path", "nunique"),
        hashes=("sha256", "nunique"),
    )
    conflicts = conflicting[
        (conflicting["identities"] != 1)
        | (conflicting["paths"] != 1)
        | (conflicting["hashes"] != 1)
    ]
    if not conflicts.empty:
        raise ValueError(f"Conflicting metadata for {len(conflicts)} image IDs")

    return (
        images.sort_values(["identity", "image_path", "image_id"])
        .drop_duplicates(subset=["image_id"], keep="first")
        .reset_index(drop=True)
    )


def build_protocol(
    images: pd.DataFrame,
    *,
    seed: int,
    min_images: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if min_images < 2:
        raise ValueError("min-images must be at least 2")

    query_rows: list[dict[str, str]] = []
    relevance_rows: list[dict[str, str | int]] = []

    for identity, group in images.groupby("identity", sort=True):
        group = group.sort_values(["image_path", "image_id"]).copy()
        if len(group) < min_images:
            continue

        group["_selection_key"] = group["image_id"].map(
            lambda image_id: selection_key(seed, str(identity), str(image_id))
        )
        query = group.sort_values(["_selection_key", "image_id"]).iloc[0]
        query_id = f"lfw_{identity}"
        query_path = str(query["image_path"]).replace("\\", "/")
        if not resolve_stored_path(query_path).is_file():
            raise FileNotFoundError(f"Query image does not exist: {query_path}")

        source_image_id = str(query["image_id"])
        query_rows.append(
            {
                "query_id": query_id,
                "query_path": query_path,
                "target_label": str(identity),
                "query_type": "face",
                "source_image_id": source_image_id,
                "exclude_image_ids": "",
            }
        )

        relevant_ids = sorted(set(group["image_id"].astype(str)) - {source_image_id})
        for image_id in relevant_ids:
            relevance_rows.append(
                {"query_id": query_id, "image_id": image_id, "relevant": 1}
            )

    queries = pd.DataFrame(query_rows, columns=QUERY_COLUMNS)
    relevance = pd.DataFrame(relevance_rows, columns=RELEVANCE_COLUMNS)
    if queries.empty or relevance.empty:
        raise RuntimeError("The LFW protocol produced no eligible queries or relevance rows")
    return queries, relevance


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    metadata_path = args.index_dir / "face_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing face metadata: {metadata_path}")

    queries_path = args.output_dir / "lfw_face_queries.csv"
    relevance_path = args.output_dir / "lfw_face_relevance.csv"
    if not args.overwrite:
        existing = [path for path in (queries_path, relevance_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing files: {existing}")

    metadata = pd.read_csv(metadata_path)
    images = unique_indexed_images(metadata)
    queries, relevance = build_protocol(
        images,
        seed=args.seed,
        min_images=args.min_images,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(queries, queries_path)
    write_csv_atomic(relevance, relevance_path)

    manifest_path = write_run_manifest(
        args.manifest_dir / "lfw_protocol_manifest.json",
        script_path=Path(__file__),
        method="lfw_face_protocol",
        configuration={
            "seed": args.seed,
            "min_images": args.min_images,
            "queries_per_identity": 1,
            "selection": "minimum_sha256(seed, identity, image_id)",
        },
        inputs={"face_metadata": metadata_path},
        result_files={"queries": queries_path, "relevance": relevance_path},
        extra={
            "indexed_images": int(len(images)),
            "indexed_identities": int(images["identity"].nunique()),
            "eligible_identities": int(len(queries)),
            "queries": int(len(queries)),
            "relevance_rows": int(len(relevance)),
        },
    )

    print(f"[OK] queries: {queries_path}")
    print(f"[OK] relevance: {relevance_path}")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] indexed images: {len(images)}")
    print(f"[OK] indexed identities: {images['identity'].nunique()}")
    print(f"[OK] eligible identities and queries: {len(queries)}")
    print(f"[OK] relevance rows: {len(relevance)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
