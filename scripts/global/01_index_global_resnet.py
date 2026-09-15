from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from global_lib import build_resnet50_feature_extractor, describe_torch_device, extract_global_embedding
from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval_common import (
    list_images,
    parent_label,
    read_image_size,
    rel_to_root,
    sha256_file,
    stable_image_id,
)
from run_manifest import write_run_manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Index whole-image ResNet50 embeddings.")
    ap.add_argument("--input-dir", type=Path, default=DATA_DIR / "raw" / "holidays")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "global_index")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--split", default="index")
    ap.add_argument("--no-label-from-parent", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir, max_images=args.max_images)
    if not image_paths:
        print(f"[ERRO] No images found in {input_dir}")
        return 1

    extractor, transform, device = build_resnet50_feature_extractor(
        device=args.device,
        weights_name=args.weights,
    )
    print(f"[INFO] Torch device: {describe_torch_device(args.device, device)}")

    rows = []
    embs = []
    failures = []
    for img_path in tqdm(image_paths, desc="Indexing global embeddings", unit="img"):
        try:
            emb = extract_global_embedding(img_path, extractor, transform, device)
        except Exception as exc:
            failures.append({"image_path": rel_to_root(img_path), "error": repr(exc)})
            continue
        image_id = stable_image_id(img_path, input_dir)
        width, height = read_image_size(img_path)
        label = "" if args.no_label_from_parent else parent_label(img_path)
        rows.append(
            {
                "image_id": image_id,
                "image_path": rel_to_root(img_path),
                "sha256": sha256_file(img_path),
                "embedding_row": len(embs),
                "width": width,
                "height": height,
                "label_or_group": label,
                "split": args.split,
            }
        )
        embs.append(emb.astype(np.float32))

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
            "split": args.split,
            "label_from_parent": not args.no_label_from_parent,
        },
        inputs={"input_directory": input_dir},
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
