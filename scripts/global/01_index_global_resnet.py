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

from global_lib import build_resnet50_feature_extractor, extract_global_embedding
from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval_common import list_images, parent_label, read_image_size, rel_to_root, stable_image_id


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
    np.save(out_dir / "global_embeddings.npy", E)
    meta.to_csv(out_dir / "global_metadata.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "global_failures.csv", index=False)

    print(f"[OK] images scanned: {len(image_paths)}")
    print(f"[OK] global embeddings: {len(meta)}")
    print(f"[OK] failures: {len(failures)}")
    print(f"[OK] {out_dir / 'global_embeddings.npy'}")
    print(f"[OK] {out_dir / 'global_metadata.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
