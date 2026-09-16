from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from project_paths import DATA_DIR, OUTPUTS_DIR

TARGET_ID = "agrishow2022_person_01"
EXPECTED_RECORDS = 124
ALLOWED_PRESENCE = {"present", "absent"}
ALLOWED_REVIEW_STATUS = {"first_review_completed", "second_review_completed"}
REQUIRED_COLUMNS = {
    "image_id",
    "file_name",
    "target_id",
    "presence",
    "review_status",
    "notes",
    "commons_page_url",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the Agrishow 2022 manual annotations."
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
        "--annotations",
        type=Path,
        default=DATA_DIR
        / "annotations"
        / "agrishow_2022_presence_review.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR
        / "experiments"
        / "ex-020_agrishow_annotation_review",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def validate_annotations(
    inventory_rows: list[dict[str, str]],
    annotation_columns: list[str],
    annotation_rows: list[dict[str, str]],
) -> dict[str, int]:
    missing_columns = sorted(REQUIRED_COLUMNS - set(annotation_columns))
    if missing_columns:
        raise RuntimeError(f"Missing annotation columns: {missing_columns}")
    if len(inventory_rows) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS} inventory records, found {len(inventory_rows)}."
        )
    if len(annotation_rows) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS} annotations, found {len(annotation_rows)}."
        )

    inventory_by_id = {row["image_id"]: row for row in inventory_rows}
    annotations_by_id = {row["image_id"]: row for row in annotation_rows}
    if len(inventory_by_id) != EXPECTED_RECORDS:
        raise RuntimeError("Inventory contains duplicate image IDs.")
    if len(annotations_by_id) != EXPECTED_RECORDS:
        raise RuntimeError("Annotations contain duplicate image IDs.")
    if set(inventory_by_id) != set(annotations_by_id):
        missing = sorted(set(inventory_by_id) - set(annotations_by_id))
        extra = sorted(set(annotations_by_id) - set(inventory_by_id))
        raise RuntimeError(
            f"Annotation IDs do not match inventory: missing={missing} extra={extra}"
        )

    counts = {"present": 0, "absent": 0, "second_review_completed": 0}
    for image_id, row in annotations_by_id.items():
        inventory_row = inventory_by_id[image_id]
        if row["file_name"] != inventory_row["file_name"]:
            raise RuntimeError(f"File name mismatch for {image_id}.")
        if row["commons_page_url"] != inventory_row["page_url"]:
            raise RuntimeError(f"Source URL mismatch for {image_id}.")
        if row["target_id"] != TARGET_ID:
            raise RuntimeError(f"Unexpected target ID for {image_id}.")
        if row["presence"] not in ALLOWED_PRESENCE:
            raise RuntimeError(
                f"Unresolved or invalid presence for {image_id}: {row['presence']!r}"
            )
        if row["review_status"] not in ALLOWED_REVIEW_STATUS:
            raise RuntimeError(
                f"Invalid review status for {image_id}: {row['review_status']!r}"
            )
        counts[row["presence"]] += 1
        if row["review_status"] == "second_review_completed":
            counts["second_review_completed"] += 1
    return counts


def main() -> int:
    args = parse_args()
    inventory_path = args.inventory.resolve()
    annotations_path = args.annotations.resolve()
    output_dir = args.output_dir.resolve()

    _, inventory_rows = read_csv(inventory_path)
    annotation_columns, annotation_rows = read_csv(annotations_path)
    counts = validate_annotations(
        inventory_rows,
        annotation_columns,
        annotation_rows,
    )

    manifest = {
        "execution_id": "EX-020",
        "status": "approved_and_frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_id": TARGET_ID,
        "records": len(annotation_rows),
        "present": counts["present"],
        "absent": counts["absent"],
        "uncertain": 0,
        "second_review_completed": counts["second_review_completed"],
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "annotations_path": str(annotations_path),
        "annotations_sha256": sha256_file(annotations_path),
        "ground_truth_independent_of_model_scores": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "annotations_frozen_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("EX-020: APPROVED AND FROZEN")
    print(
        f"records={manifest['records']} present={manifest['present']} "
        f"absent={manifest['absent']} uncertain={manifest['uncertain']}"
    )
    print(f"second_review_completed={manifest['second_review_completed']}")
    print(f"annotations_sha256={manifest['annotations_sha256']}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
