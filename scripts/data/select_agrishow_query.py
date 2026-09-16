from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from project_paths import DATA_DIR, OUTPUTS_DIR

TARGET_ID = "agrishow2022_person_01"
EXPECTED_PRESENT = 91


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and freeze one Agrishow 2022 query source before retrieval."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DATA_DIR
        / "annotations"
        / "agrishow_2022_presence_review.csv",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DATA_DIR
        / "raw"
        / "agrishow_2022"
        / "metadata"
        / "agrishow_2022_selected.csv",
    )
    parser.add_argument(
        "--frozen-annotations-manifest",
        type=Path,
        default=OUTPUTS_DIR
        / "experiments"
        / "ex-020_agrishow_annotation_review"
        / "annotations_frozen_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-021_agrishow_query_selection",
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=DATA_DIR
        / "annotations"
        / "agrishow_2022_query_selection.csv",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def selection_key(seed: int, image_id: str) -> str:
    payload = f"{seed}\0{TARGET_ID}\0{image_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_selection_csv(path: Path, row: dict[str, str | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    annotations_path = args.annotations.resolve()
    inventory_path = args.inventory.resolve()
    frozen_manifest_path = args.frozen_annotations_manifest.resolve()
    output_dir = args.output_dir.resolve()
    selection_csv_path = args.selection_csv.resolve()
    selection_manifest_path = output_dir / "query_selection_manifest.json"

    if selection_manifest_path.exists() and not args.overwrite:
        existing = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
        print("EX-021: EXISTING SELECTION PRESERVED")
        print(f"source_image_id={existing['source_image_id']}")
        print(f"seed={existing['seed']}")
        print(f"manifest={selection_manifest_path}")
        return 0

    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    if frozen_manifest.get("status") != "approved_and_frozen":
        raise RuntimeError("EX-020 annotations are not approved and frozen.")
    annotations_sha256 = sha256_file(annotations_path)
    if annotations_sha256 != frozen_manifest.get("annotations_sha256"):
        raise RuntimeError("Annotation hash differs from the frozen EX-020 manifest.")

    annotations = read_csv(annotations_path)
    inventory = read_csv(inventory_path)
    inventory_by_id = {row["image_id"]: row for row in inventory}
    candidates = sorted(
        row["image_id"]
        for row in annotations
        if row["target_id"] == TARGET_ID and row["presence"] == "present"
    )
    if len(candidates) != EXPECTED_PRESENT or len(set(candidates)) != EXPECTED_PRESENT:
        raise RuntimeError(
            f"Expected {EXPECTED_PRESENT} unique present candidates, found {len(candidates)}."
        )
    if not set(candidates).issubset(inventory_by_id):
        raise RuntimeError("One or more candidates are absent from the inventory.")

    seed = args.seed if args.seed is not None else secrets.randbits(64)
    ranked = sorted(candidates, key=lambda image_id: selection_key(seed, image_id))
    source_image_id = ranked[0]
    source = inventory_by_id[source_image_id]
    source_path = (
        DATA_DIR / "raw" / "agrishow_2022" / "images" / source["file_name"]
    ).resolve()
    source_sha256 = sha256_file(source_path)
    selected_key = selection_key(seed, source_image_id)

    selection_row: dict[str, str | int] = {
        "query_id": "agrishow2022_query_01",
        "target_id": TARGET_ID,
        "source_image_id": source_image_id,
        "source_file_name": source["file_name"],
        "source_image_path": str(source_path),
        "source_image_sha256": source_sha256,
        "exclude_image_ids": source_image_id,
        "selection_seed": seed,
        "selection_key": selected_key,
        "selection_rule": "minimum_sha256(seed, target_id, image_id)",
        "candidate_count": len(candidates),
        "commons_page_url": source["page_url"],
        "photographer": source["photographer"],
        "license": source["license"],
        "face_query_status": "pending_manual_target_crop",
    }
    write_selection_csv(selection_csv_path, selection_row)

    manifest = {
        "execution_id": "EX-021",
        "status": "source_selected_and_frozen",
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_id": TARGET_ID,
        "candidate_rule": "presence == present in frozen EX-020 annotations",
        "candidate_count": len(candidates),
        "seed": seed,
        "selection_rule": "minimum_sha256(seed, target_id, image_id)",
        "selected_key": selected_key,
        "source_image_id": source_image_id,
        "source_file_name": source["file_name"],
        "source_image_path": str(source_path),
        "source_image_sha256": source_sha256,
        "exclude_image_ids": [source_image_id],
        "annotations_sha256": annotations_sha256,
        "inventory_sha256": sha256_file(inventory_path),
        "selection_csv_path": str(selection_csv_path),
        "selection_csv_sha256": sha256_file(selection_csv_path),
        "commons_page_url": source["page_url"],
        "photographer": source["photographer"],
        "license": source["license"],
        "face_query_status": "pending_manual_target_crop",
        "selection_before_retrieval": True,
        "redraw_based_on_results_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("EX-021: RANDOM SOURCE SELECTED AND FROZEN")
    print(f"candidates={len(candidates)}")
    print(f"seed={seed}")
    print(f"source_image_id={source_image_id}")
    print(f"source_image_sha256={source_sha256}")
    print(f"selection_csv={selection_csv_path}")
    print(f"manifest={selection_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
