from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from project_paths import DATA_DIR

DATASET_PAGE = "http://chenlab.ece.cornell.edu/people/Andy/GallagherDataset.html"
URLS_URL = "http://chenlab.ece.cornell.edu/people/Andy/GallagherDataset.txt"
GT_URL = "http://chenlab.ece.cornell.edu/people/Andy/GallagherDatasetGT.txt"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Download and prepare the Gallagher Collection Person Dataset."
    )
    ap.add_argument("--output-dir", type=Path, default=DATA_DIR / "raw" / "gallagher")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def download_file(url: str, dest: Path, timeout: int, overwrite: bool = False) -> bool:
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        dest.write_bytes(response.read())
    return True


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def parse_ground_truth(gt_path: Path, image_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_image_count: dict[str, int] = {}
    for line in read_lines(gt_path):
        parts = line.split("\t")
        if len(parts) != 6:
            raise ValueError(f"Invalid ground truth row: {line}")
        image_name, lex, ley, rex, rey, identity = parts
        face_index = per_image_count.get(image_name, 0)
        per_image_count[image_name] = face_index + 1
        rows.append(
            {
                "image_name": image_name,
                "image_path": str((image_dir / image_name).relative_to(ROOT)).replace("\\", "/"),
                "face_index": str(face_index),
                "left_eye_x": lex,
                "left_eye_y": ley,
                "right_eye_x": rex,
                "right_eye_y": rey,
                "identity": identity,
                "source": "gallagher",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def download_image(row: dict[str, str], timeout: int, overwrite: bool, sleep: float) -> dict[str, str]:
    dest = ROOT / row["image_path"]
    try:
        changed = download_file(row["url"], dest, timeout, overwrite=overwrite)
        if sleep > 0:
            time.sleep(sleep)
        return {
            "image_name": row["image_name"],
            "url": row["url"],
            "status": "downloaded" if changed else "cached",
            "error": "",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "image_name": row["image_name"],
            "url": row["url"],
            "status": "failed",
            "error": repr(exc),
        }


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    meta_dir = out_dir / "metadata"
    image_dir = out_dir / "images"
    meta_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    urls_path = meta_dir / "GallagherDataset.txt"
    gt_path = meta_dir / "GallagherDatasetGT.txt"
    print(f"[INFO] Dataset page: {DATASET_PAGE}")
    print("[INFO] Terms: non-commercial research only; do not redistribute images.")
    download_file(URLS_URL, urls_path, args.timeout, overwrite=args.overwrite)
    download_file(GT_URL, gt_path, args.timeout, overwrite=args.overwrite)

    urls = unique_urls(read_lines(urls_path))
    if args.max_images is not None:
        urls = urls[: args.max_images]

    url_rows = [
        {
            "image_name": Path(url).name,
            "url": url,
            "image_path": str((image_dir / Path(url).name).relative_to(ROOT)).replace("\\", "/"),
        }
        for url in urls
    ]
    write_csv(meta_dir / "image_urls.csv", url_rows, ["image_name", "url", "image_path"])

    annotations = parse_ground_truth(gt_path, image_dir)
    write_csv(
        meta_dir / "face_annotations.csv",
        annotations,
        [
            "image_name",
            "image_path",
            "face_index",
            "left_eye_x",
            "left_eye_y",
            "right_eye_x",
            "right_eye_y",
            "identity",
            "source",
        ],
    )

    if args.metadata_only:
        print(f"[OK] metadata: {meta_dir}")
        print(f"[OK] unique image URLs: {len(urls)}")
        print(f"[OK] face annotations: {len(annotations)}")
        return 0

    ok = 0
    failed: list[dict[str, str]] = []
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_image, row, args.timeout, args.overwrite, args.sleep)
            for row in url_rows
        ]
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["status"] == "failed":
                failed.append(
                    {
                        "image_name": result["image_name"],
                        "url": result["url"],
                        "error": result["error"],
                    }
                )
                print(
                    f"[{i:03d}/{len(url_rows):03d}] failed: {result['image_name']} - {result['error']}",
                    flush=True,
                )
            else:
                ok += 1
                print(
                    f"[{i:03d}/{len(url_rows):03d}] {result['status']}: {result['image_name']}",
                    flush=True,
                )

    if failed:
        write_csv(meta_dir / "download_failures.csv", failed, ["image_name", "url", "error"])

    print(f"[OK] image URLs listed: {len(url_rows)}")
    print(f"[OK] images available locally: {ok}")
    print(f"[OK] failed downloads: {len(failed)}")
    print(f"[OK] annotations: {meta_dir / 'face_annotations.csv'}")
    print(f"[OK] images: {image_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
