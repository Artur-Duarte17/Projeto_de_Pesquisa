from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import OUTPUTS_DIR
from retrieval.adapters.manifest import write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create face and global indexes for a folder of photographs."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--collection-name", default=None, help="Display name in the local application")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "collections" / "default",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_commands(
    *,
    python_executable: str,
    input_dir: Path,
    output_dir: Path,
    device: str,
    det_size: int,
    batch_size: int,
    workers: int,
    max_images: int | None,
) -> list[list[str]]:
    face_command = [
        python_executable,
        str(ROOT / "scripts" / "face" / "01_index_faces.py"),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir / "face"),
        "--device",
        device,
        "--det-size",
        str(det_size),
        "--no-identity-from-parent",
    ]
    global_command = [
        python_executable,
        str(ROOT / "scripts" / "global" / "01_index_global_resnet.py"),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir / "global"),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--workers",
        str(workers),
        "--no-label-from-parent",
    ]
    if max_images is not None:
        face_command.extend(["--max-images", str(max_images)])
        global_command.extend(["--max-images", str(max_images)])
    return [face_command, global_command]


def protected_outputs(output_dir: Path) -> list[Path]:
    return [
        output_dir / "face" / "face_embeddings.npy",
        output_dir / "face" / "face_metadata.csv",
        output_dir / "global" / "global_embeddings.npy",
        output_dir / "global" / "global_metadata.csv",
        output_dir / "collection_manifest.json",
    ]


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.det_size <= 0 or args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("det-size, batch-size and workers must be positive")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("max-images must be positive")
    try:
        output_dir.relative_to(input_dir)
    except ValueError:
        pass
    else:
        raise ValueError("Output directory cannot be inside the input photo directory")

    existing = [path for path in protected_outputs(output_dir) if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Collection outputs already exist; use --overwrite: {joined}")

    output_dir.mkdir(parents=True, exist_ok=True)
    commands = build_commands(
        python_executable=sys.executable,
        input_dir=input_dir,
        output_dir=output_dir,
        device=args.device,
        det_size=args.det_size,
        batch_size=args.batch_size,
        workers=args.workers,
        max_images=args.max_images,
    )
    for command in commands:
        print(f"[RUN] {subprocess.list2cmdline(command)}")
        subprocess.run(command, cwd=ROOT, check=True)

    face_manifest = output_dir / "face" / "face_index_manifest.json"
    global_manifest = output_dir / "global" / "global_index_manifest.json"
    manifest_path = write_run_manifest(
        output_dir / "collection_manifest.json",
        script_path=Path(__file__),
        method="photo_collection_preparation",
        configuration={
            "device": args.device,
            "det_size": args.det_size,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "max_images": args.max_images,
            "identity_labels_stored": False,
            "collection_name": args.collection_name,
        },
        inputs={"input_directory": input_dir},
        result_files={
            "face_index_manifest": face_manifest,
            "global_index_manifest": global_manifest,
        },
        extra={
            "face_index_directory": str(output_dir / "face"),
            "global_index_directory": str(output_dir / "global"),
        },
    )
    print(f"[OK] Collection prepared: {output_dir}")
    print(f"[OK] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
