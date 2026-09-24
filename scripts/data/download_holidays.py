from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import DATA_DIR
from retrieval.adapters.files import stable_image_id

DATASET_PAGE = "https://thoth.inrialpes.fr/~jegou/data.php.html"
ARCHIVES = [
    ("jpg1.tar.gz", "ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz"),
    ("jpg2.tar.gz", "ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg2.tar.gz"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Download and prepare the INRIA Holidays dataset.")
    ap.add_argument("--output-dir", type=Path, default=DATA_DIR / "raw" / "holidays")
    ap.add_argument("--source", choices=["kaggle", "official"], default="kaggle")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-images", type=int, default=None)
    return ap.parse_args()


def find_kaggle_dataset() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install kagglehub to use --source kaggle: pip install kagglehub") from exc
    return Path(kagglehub.dataset_download("vadimshabashov/inria-holidays"))


def run_curl(url: str, dest: Path) -> None:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl was not found in PATH.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [curl, "-L", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o", str(dest), url]
    subprocess.run(cmd, check=True)


def download_with_urllib(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response, dest.open("ab") as f:
        shutil.copyfileobj(response, f)


def download_archive(url: str, dest: Path, overwrite: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        print(f"[OK] cached archive: {dest}")
        return
    if overwrite and dest.exists():
        dest.unlink()
    try:
        run_curl(url, dest)
    except Exception as exc:
        print(f"[WARN] curl failed, trying urllib: {exc}")
        download_with_urllib(url, dest)


def safe_extract_images(archive_path: Path, image_dir: Path, overwrite: bool) -> int:
    image_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name.lower().endswith((".jpg", ".jpeg")):
                continue
            dest = image_dir / name
            if dest.exists() and dest.stat().st_size > 0 and not overwrite:
                extracted += 1
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, dest.open("wb") as f:
                shutil.copyfileobj(src, f)
            extracted += 1
    return extracted


def copy_images(src_dir: Path, image_dir: Path, overwrite: bool) -> int:
    image_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(src_dir.rglob("*")):
        if not src.is_file() or not src.name.lower().endswith((".jpg", ".jpeg")):
            continue
        dest = image_dir / src.name
        if dest.exists() and dest.stat().st_size > 0 and not overwrite:
            copied += 1
            continue
        shutil.copy2(src, dest)
        copied += 1
    return copied


def group_id_from_name(path: Path) -> str:
    return f"{int(path.stem) // 100:04d}"


def is_query(path: Path) -> bool:
    return int(path.stem) % 100 == 0


def write_csv(path: Path, rows: list[dict[str, str | int]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_protocol(image_dir: Path, output_dir: Path, max_images: int | None = None) -> None:
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.JPG"))
    images = sorted({p for p in images})
    if max_images is not None:
        images = images[:max_images]
    image_set = {p.name for p in images}

    metadata_rows: list[dict[str, str | int]] = []
    for p in images:
        metadata_rows.append(
            {
                "image_id": stable_image_id(p, image_dir),
                "image_name": p.name,
                "image_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "group_id": group_id_from_name(p),
                "is_query": int(is_query(p)),
            }
        )

    by_group: dict[str, list[Path]] = {}
    for p in images:
        by_group.setdefault(group_id_from_name(p), []).append(p)

    queries_rows: list[dict[str, str]] = []
    relevance_rows: list[dict[str, str | int]] = []
    for group_id, group_images in sorted(by_group.items()):
        query_candidates = [p for p in group_images if is_query(p)]
        if not query_candidates:
            continue
        query = query_candidates[0]
        if query.name not in image_set:
            continue
        query_id = f"holidays_{group_id}"
        queries_rows.append(
            {
                    "query_id": query_id,
                    "query_path": str(query.relative_to(ROOT)).replace("\\", "/"),
                    "target_label": group_id,
                "query_type": "global",
            }
        )
        for p in group_images:
            if p.name == query.name:
                continue
            relevance_rows.append(
                {
                    "query_id": query_id,
                    "image_id": stable_image_id(p, image_dir),
                    "relevant": 1,
                }
            )

    meta_dir = output_dir / "metadata"
    write_csv(
        meta_dir / "holidays_metadata.csv",
        metadata_rows,
        ["image_id", "image_name", "image_path", "group_id", "is_query"],
    )
    write_csv(
        DATA_DIR / "evaluation" / "holidays_global_queries.csv",
        queries_rows,
        ["query_id", "query_path", "target_label", "query_type"],
    )
    write_csv(
        DATA_DIR / "evaluation" / "holidays_relevance.csv",
        relevance_rows,
        ["query_id", "image_id", "relevant"],
    )

    print(f"[OK] images prepared: {len(images)}")
    print(f"[OK] groups: {len(by_group)}")
    print(f"[OK] queries: {len(queries_rows)}")
    print(f"[OK] relevance rows: {len(relevance_rows)}")
    print(f"[OK] {meta_dir / 'holidays_metadata.csv'}")
    print(f"[OK] {DATA_DIR / 'evaluation' / 'holidays_global_queries.csv'}")
    print(f"[OK] {DATA_DIR / 'evaluation' / 'holidays_relevance.csv'}")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    archives_dir = out_dir / "archives"
    image_dir = out_dir / "images"
    archives_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Dataset page: {DATASET_PAGE}")
    print("[INFO] Official protocol: first image in each group is query; others are relevant.")

    if args.source == "kaggle":
        if args.skip_download:
            kaggle_root = Path.home() / ".cache" / "kagglehub" / "datasets" / "vadimshabashov" / "inria-holidays" / "versions" / "1"
        else:
            print("[INFO] downloading via Kaggle mirror")
            kaggle_root = find_kaggle_dataset()
        src_images = kaggle_root / "images"
        if not src_images.exists():
            raise FileNotFoundError(f"Missing Kaggle images directory: {src_images}")
        print(f"[INFO] copying images from {src_images}")
        count = copy_images(src_images, image_dir, args.overwrite)
        print(f"[OK] Kaggle images seen/copied: {count}")
    elif not args.skip_download:
        for filename, url in ARCHIVES:
            print(f"[INFO] downloading {filename}")
            download_archive(url, archives_dir / filename, args.overwrite)

    if args.source == "official" and not args.skip_extract:
        for filename, _url in ARCHIVES:
            archive_path = archives_dir / filename
            if not archive_path.exists():
                raise FileNotFoundError(f"Missing archive: {archive_path}")
            print(f"[INFO] extracting {filename}")
            count = safe_extract_images(archive_path, image_dir, args.overwrite)
            print(f"[OK] {filename}: {count} images seen/extracted")

    prepare_protocol(image_dir, out_dir, max_images=args.max_images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
