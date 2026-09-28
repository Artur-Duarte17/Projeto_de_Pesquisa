from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval.adapters.global_model import (
    build_resnet50_feature_extractor,
    describe_torch_device,
    extract_global_embeddings_batch,
)
from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval.adapters.files import (
    list_images,
    load_inventory_image_ids,
    parent_label,
    rel_to_root,
    sha256_file,
    stable_image_id,
)
from retrieval.adapters.image_io import open_image, oriented_rgb
from retrieval.adapters.manifest import write_run_manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Index whole-image ResNet50 embeddings.")
    ap.add_argument("--input-dir", type=Path, default=DATA_DIR / "raw" / "holidays")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "global_index")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="index")
    ap.add_argument("--no-label-from-parent", action="store_true")
    ap.add_argument(
        "--inventory-csv",
        type=Path,
        default=None,
        help="Optional CSV whose image_id values replace generated path hashes.",
    )
    return ap.parse_args()


def prepare_image(image_path: Path, transform) -> dict[str, object]:
    try:
        with open_image(image_path) as image:
            rgb = oriented_rgb(image, image_path)
            width, height = rgb.size
            tensor = transform(rgb)
        return {
            "image_path": image_path,
            "tensor": tensor,
            "width": int(width),
            "height": int(height),
            "sha256": sha256_file(image_path),
            "error": None,
        }
    except Exception as exc:
        return {
            "image_path": image_path,
            "tensor": None,
            "width": None,
            "height": None,
            "sha256": None,
            "error": repr(exc),
        }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    input_dir = args.input_dir.resolve()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir, max_images=args.max_images)
    if not image_paths:
        print(f"[ERRO] No images found in {input_dir}")
        return 1
    inventory_path = args.inventory_csv.resolve() if args.inventory_csv else None
    inventory_ids = (
        load_inventory_image_ids(inventory_path, input_dir, image_paths)
        if inventory_path
        else None
    )

    extractor, transform, device = build_resnet50_feature_extractor(
        device=args.device,
        weights_name=args.weights,
    )
    print(f"[INFO] Torch device: {describe_torch_device(args.device, device)}")

    rows = []
    embs = []
    failures = []
    progress = tqdm(total=len(image_paths), desc="Indexing global embeddings", unit="img")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for batch_start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[batch_start : batch_start + args.batch_size]
            prepared = list(executor.map(lambda path: prepare_image(path, transform), batch_paths))
            valid = [item for item in prepared if item["error"] is None]
            for item in prepared:
                if item["error"] is not None:
                    failures.append(
                        {
                            "image_path": rel_to_root(Path(item["image_path"])),
                            "error": str(item["error"]),
                        }
                    )

            if valid:
                batch_embeddings = extract_global_embeddings_batch(
                    [item["tensor"] for item in valid],
                    extractor,
                    device,
                )

                for item, embedding in zip(valid, batch_embeddings, strict=True):
                    img_path = Path(item["image_path"])
                    image_id = (
                        inventory_ids[img_path.resolve()]
                        if inventory_ids is not None
                        else stable_image_id(img_path, input_dir)
                    )
                    label = "" if args.no_label_from_parent else parent_label(img_path)
                    rows.append(
                        {
                            "image_id": image_id,
                            "image_path": rel_to_root(img_path),
                            "sha256": str(item["sha256"]),
                            "embedding_row": len(embs),
                            "width": int(item["width"]),
                            "height": int(item["height"]),
                            "label_or_group": label,
                            "split": args.split,
                        }
                    )
                    embs.append(embedding)
            progress.update(len(batch_paths))
    progress.close()

    if not embs:
        print("[ERRO] No global embeddings were extracted.")
        return 1

    E = np.vstack(embs).astype(np.float32)
    meta = pd.DataFrame(rows)
    embeddings_path = out_dir / "global_embeddings.npy"
    metadata_path = out_dir / "global_metadata.csv"
    failures_path = out_dir / "global_failures.csv"
    np.save(embeddings_path, E)
    meta.to_csv(metadata_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)
    elif failures_path.exists():
        failures_path.unlink()

    result_files = {"embeddings": embeddings_path, "metadata": metadata_path}
    if failures_path.exists():
        result_files["failures"] = failures_path
    manifest_path = write_run_manifest(
        out_dir / "global_index_manifest.json",
        script_path=Path(__file__),
        method="global_resnet50_index",
        configuration={
            "max_images": args.max_images,
            "device": args.device,
            "weights": args.weights,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "split": args.split,
            "label_from_parent": not args.no_label_from_parent,
            "image_id_source": "inventory_csv" if inventory_path else "stable_path_hash",
        },
        inputs={"input_directory": input_dir, "inventory_csv": inventory_path},
        result_files=result_files,
        extra={"images_scanned": len(image_paths), "embeddings": len(meta), "failures": len(failures)},
    )

    print(f"[OK] images scanned: {len(image_paths)}")
    print(f"[OK] global embeddings: {len(meta)}")
    print(f"[OK] failures: {len(failures)}")
    print(f"[OK] {embeddings_path}")
    print(f"[OK] {metadata_path}")
    print(f"[OK] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
