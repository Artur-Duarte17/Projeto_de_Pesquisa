from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

import requests

from widerface_eval_lib import load_difficulty_indices, parse_wider_annotations


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_http(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
    temporary.replace(destination)


def download_archives(source_manifest: dict, download_dir: Path) -> dict[str, Path]:
    download_dir.mkdir(parents=True, exist_ok=True)
    archives = source_manifest["archives"]
    paths = {name: download_dir / item["filename"] for name, item in archives.items()}
    if not paths["validation_images"].exists():
        import gdown

        output = gdown.download(
            id=archives["validation_images"]["google_drive_file_id"],
            output=str(paths["validation_images"]),
            quiet=False,
        )
        if output is None:
            raise RuntimeError("Google Drive did not return the WIDER validation archive")
    for name in ("annotations", "evaluation_tools"):
        if not paths[name].exists():
            _download_http(archives[name]["url"], paths[name])
    return paths


def validate_archive(path: Path, expected: dict) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    sha256 = file_hash(path, "sha256")
    md5 = file_hash(path, "md5")
    if size != int(expected["bytes"]):
        raise ValueError(f"Unexpected size for {path.name}: {size}")
    if sha256.lower() != expected["sha256"].lower():
        raise ValueError(f"SHA-256 mismatch for {path.name}: {sha256}")
    if expected.get("md5") and md5.lower() != expected["md5"].lower():
        raise ValueError(f"MD5 mismatch for {path.name}: {md5}")
    return {"path": str(path.resolve()), "bytes": size, "md5": md5, "sha256": sha256}


def inspect_zip(path: Path, forbidden_markers: list[str]) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        names = [item.filename for item in archive.infolist()]
    for name in names:
        normalized = name.replace("\\", "/")
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"Unsafe ZIP member: {name}")
        if any(marker.lower() in normalized.lower() for marker in forbidden_markers):
            raise ValueError(f"Forbidden train/test member in archive: {name}")
    return names


def _safe_extract(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def validate_extracted(dataset_root: Path, expected_images: int) -> dict[str, object]:
    images_root = dataset_root / "WIDER_val" / "images"
    annotations_path = dataset_root / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    ground_truth_dir = dataset_root / "eval_tools" / "ground_truth"
    image_paths = sorted(path for path in images_root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg"})
    if len(image_paths) != expected_images:
        raise ValueError(f"Expected {expected_images} validation images; found {len(image_paths)}")
    unexpected = [path for path in dataset_root.rglob("*") if "WIDER_train" in str(path) or "WIDER_test" in str(path)]
    if unexpected:
        raise ValueError(f"Train/test files found: {unexpected[0]}")
    annotations = parse_wider_annotations(annotations_path)
    image_keys = {path.relative_to(images_root).as_posix() for path in image_paths}
    if len(annotations) != expected_images or set(annotations) != image_keys:
        missing = sorted(set(annotations) - image_keys)
        extra = sorted(image_keys - set(annotations))
        raise ValueError(f"Image/annotation mismatch; missing={missing[:3]} extra={extra[:3]}")
    difficulty_counts: dict[str, int] = {}
    for level in ("easy", "medium", "hard"):
        path = ground_truth_dir / f"wider_{level}_val.mat"
        keep = load_difficulty_indices(path)
        if set(keep) != image_keys:
            raise ValueError(f"{level} difficulty file does not cover exactly the validation images")
        for key, indices in keep.items():
            if any(index < 0 or index >= len(annotations[key].boxes) for index in indices):
                raise ValueError(f"Invalid {level} ground-truth index for {key}")
        difficulty_counts[level] = sum(len(indices) for indices in keep.values())
    return {
        "images": len(image_paths),
        "annotated_images": len(annotations),
        "faces": sum(len(item.boxes) for item in annotations.values()),
        "difficulty_faces": difficulty_counts,
        "images_root": str(images_root.resolve()),
        "annotations_path": str(annotations_path.resolve()),
        "ground_truth_dir": str(ground_truth_dir.resolve()),
    }


def prepare_dataset(
    source_manifest_path: Path,
    download_dir: Path,
    dataset_root: Path,
    audit_path: Path,
) -> dict[str, object]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    archive_paths = download_archives(source_manifest, download_dir)
    archive_audit: dict[str, object] = {}
    member_counts: dict[str, int] = {}
    for name, path in archive_paths.items():
        archive_audit[name] = validate_archive(path, source_manifest["archives"][name])
        members = inspect_zip(path, source_manifest["forbidden_archive_markers"])
        member_counts[name] = len(members)
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty dataset directory: {dataset_root.resolve()}"
        )
    for path in archive_paths.values():
        _safe_extract(path, dataset_root)
    extracted = validate_extracted(dataset_root, int(source_manifest["expected_validation_images"]))
    notice = dataset_root / "WIDER_FACE_LICENSE_NOTICE.txt"
    notice.write_text(
        "WIDER FACE official validation split only.\n"
        f"License recorded by the official project page: {source_manifest['license']}.\n"
        f"Official project: {source_manifest['official_project_url']}\n"
        "No WIDER FACE image may be committed or included in a public artifact from this project.\n",
        encoding="utf-8",
    )
    audit = {
        "source_manifest": source_manifest,
        "archives": archive_audit,
        "archive_member_counts": member_counts,
        "extracted": extracted,
        "license_notice": str(notice.resolve()),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return audit
