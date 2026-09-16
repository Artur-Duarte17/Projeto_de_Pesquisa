from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from project_paths import DATA_DIR

API_URL = "https://commons.wikimedia.org/w/api.php"
CATEGORY = "Category:Agrishow (2022)"
CATEGORY_URL = "https://commons.wikimedia.org/wiki/Category:Agrishow_(2022)"
EXPECTED_LICENSE = "CC BY 2.0"
USER_AGENT = "AcademicCBIRDatasetPreparation/1.0 (Wikimedia Commons API)"
DERIVED_TITLE_PATTERN = re.compile(r"\(cropped\)", re.IGNORECASE)
PHOTO_ID_PATTERN = re.compile(r"\((\d{8,})\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and optionally download the open Agrishow 2022 collection."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "raw" / "agrishow_2022",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download original images. Without this flag, only metadata is written.",
    )
    parser.add_argument("--include-derived", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def request_json(params: dict[str, str], timeout: int) -> dict:
    url = API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt, delay in enumerate((0, 2, 5, 10, 20), start=1):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
    raise RuntimeError("Wikimedia API request failed after retries.")


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(value).split())


def metadata_value(metadata: dict, key: str) -> str:
    return strip_html(metadata.get(key, {}).get("value"))


def list_category_titles(timeout: int) -> list[str]:
    payload = request_json(
        {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": CATEGORY,
            "cmtype": "file",
            "cmlimit": "500",
            "format": "json",
            "formatversion": "2",
        },
        timeout,
    )
    return sorted(item["title"] for item in payload["query"]["categorymembers"])


def fetch_metadata(titles: list[str], timeout: int) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for start in range(0, len(titles), 40):
        chunk = titles[start : start + 40]
        payload = request_json(
            {
                "action": "query",
                "titles": "|".join(chunk),
                "prop": "imageinfo",
                "iiprop": "url|size|sha1|mime|extmetadata",
                "format": "json",
                "formatversion": "2",
            },
            timeout,
        )
        for page in payload["query"]["pages"]:
            info = page.get("imageinfo", [None])[0]
            if info is None:
                raise RuntimeError(f"Missing image information for {page['title']}")
            metadata = info.get("extmetadata", {})
            title = page["title"]
            photo_id_match = PHOTO_ID_PATTERN.search(title)
            photo_id = photo_id_match.group(1) if photo_id_match else str(page["pageid"])
            is_derived = bool(DERIVED_TITLE_PATTERN.search(title))
            extension = Path(urllib.parse.urlparse(info["url"]).path).suffix.lower() or ".jpg"
            description = metadata_value(metadata, "ImageDescription")
            photographer_match = re.search(r"Foto:\s*([^\r\n]+)", description)
            records.append(
                {
                    "image_id": (
                        f"agrishow2022_{photo_id}_derived_{page['pageid']}"
                        if is_derived
                        else f"agrishow2022_{photo_id}"
                    ),
                    "commons_page_id": int(page["pageid"]),
                    "file_name": f"{photo_id}{extension}",
                    "commons_title": title,
                    "page_url": "https://commons.wikimedia.org/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_"), safe=":_()"),
                    "original_url": info["url"],
                    "width": int(info["width"]),
                    "height": int(info["height"]),
                    "size_bytes": int(info["size"]),
                    "sha1": info["sha1"].lower(),
                    "mime": info["mime"],
                    "description": description,
                    "date": metadata_value(metadata, "DateTimeOriginal")
                    or metadata_value(metadata, "DateTime"),
                    "artist": metadata_value(metadata, "Artist"),
                    "photographer": photographer_match.group(1).strip()
                    if photographer_match
                    else "",
                    "credit": metadata_value(metadata, "Credit"),
                    "source": metadata_value(metadata, "Source"),
                    "license": metadata_value(metadata, "LicenseShortName"),
                    "license_url": metadata_value(metadata, "LicenseUrl"),
                    "usage_terms": metadata_value(metadata, "UsageTerms"),
                    "is_derived": int(is_derived),
                }
            )
    return sorted(records, key=lambda row: str(row["commons_title"]))


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: dict[str, str | int]) -> None:
    if record["license"] != EXPECTED_LICENSE:
        raise RuntimeError(
            f"Unexpected license for {record['commons_title']}: {record['license']!r}"
        )
    if not str(record["sha1"]):
        raise RuntimeError(f"Missing SHA-1 for {record['commons_title']}")
    if int(record["size_bytes"]) <= 0:
        raise RuntimeError(f"Invalid size for {record['commons_title']}")


def download_image(
    record: dict[str, str | int],
    image_dir: Path,
    timeout: int,
    overwrite: bool,
) -> dict[str, str]:
    destination = image_dir / str(record["file_name"])
    temporary = destination.with_suffix(destination.suffix + ".part")
    expected_sha1 = str(record["sha1"])
    expected_size = int(record["size_bytes"])

    if destination.exists() and not overwrite:
        if destination.stat().st_size == expected_size and sha1_file(destination) == expected_sha1:
            return {"image_id": str(record["image_id"]), "status": "cached", "error": ""}
        return {
            "image_id": str(record["image_id"]),
            "status": "failed",
            "error": f"Existing file failed validation: {destination}",
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for delay in (0, 2, 5, 10, 20):
        if delay:
            time.sleep(delay)
        if temporary.exists():
            temporary.unlink()
        request = urllib.request.Request(
            str(record["original_url"]), headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open(
                "wb"
            ) as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
            if temporary.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Size mismatch for {destination.name}: "
                    f"{temporary.stat().st_size} != {expected_size}"
                )
            actual_sha1 = sha1_file(temporary)
            if actual_sha1 != expected_sha1:
                raise RuntimeError(
                    f"SHA-1 mismatch for {destination.name}: {actual_sha1} != {expected_sha1}"
                )
            temporary.replace(destination)
            return {
                "image_id": str(record["image_id"]),
                "status": "downloaded",
                "error": "",
            }
        except Exception as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
    return {
        "image_id": str(record["image_id"]),
        "status": "failed",
        "error": repr(last_error),
    }


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    metadata_dir = output_dir / "metadata"
    image_dir = output_dir / "images"

    titles = list_category_titles(args.timeout)
    records = fetch_metadata(titles, args.timeout)
    for record in records:
        validate_record(record)

    derived = [record for record in records if int(record["is_derived"]) == 1]
    selected = records if args.include_derived else [
        record for record in records if int(record["is_derived"]) == 0
    ]
    if args.max_images is not None:
        selected = selected[: args.max_images]
    for record in selected:
        record["image_path"] = str(
            (image_dir / str(record["file_name"])).relative_to(ROOT)
        ).replace("\\", "/")

    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metadata_dir / "commons_inventory_all.csv", records)
    write_csv(metadata_dir / "agrishow_2022_selected.csv", selected)
    summary = {
        "category": CATEGORY,
        "category_url": CATEGORY_URL,
        "files_in_category": len(records),
        "derived_files_excluded": 0 if args.include_derived else len(derived),
        "selected_files": len(selected),
        "expected_license": EXPECTED_LICENSE,
        "selected_original_bytes": sum(int(record["size_bytes"]) for record in selected),
        "images_downloaded": bool(args.download_images),
        "selection_rule": "Exclude titles containing '(cropped)' unless --include-derived is used.",
    }

    failures: list[dict[str, str]] = []
    if args.download_images:
        image_dir.mkdir(parents=True, exist_ok=True)
        workers = max(1, args.workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    download_image,
                    record,
                    image_dir,
                    args.timeout,
                    args.overwrite,
                ): record
                for record in selected
            }
            for number, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result["status"] == "failed":
                    failures.append(result)
                print(
                    f"[{number:03d}/{len(selected):03d}] {result['status']}: {result['image_id']}"
                    + (f" - {result['error']}" if result["error"] else ""),
                    flush=True,
                )
        summary["download_failures"] = len(failures)
        if failures:
            with (metadata_dir / "download_failures.json").open("w", encoding="utf-8") as stream:
                json.dump(failures, stream, ensure_ascii=False, indent=2)
        else:
            failures_path = metadata_dir / "download_failures.json"
            if failures_path.exists():
                failures_path.unlink()

    with (metadata_dir / "dataset_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(f"[OK] category files audited: {len(records)}")
    print(f"[OK] derived files excluded: {summary['derived_files_excluded']}")
    print(f"[OK] selected files: {len(selected)}")
    print(f"[OK] license: {EXPECTED_LICENSE} for every audited file")
    print(f"[OK] metadata: {metadata_dir}")
    if args.download_images:
        print(f"[OK] images: {image_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
