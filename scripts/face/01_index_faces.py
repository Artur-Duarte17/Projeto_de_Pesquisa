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

from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval.adapters.face_model import build_face_app
from retrieval.adapters.files import (
    list_images,
    load_inventory_image_ids,
    parent_label,
    rel_to_root,
    sha256_file,
    stable_image_id,
)
from retrieval.adapters.image_io import read_image_bgr
from retrieval.adapters.manifest import write_run_manifest
from retrieval.domain.face_policy import embedding_from_face, sorted_faces


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Index all detected faces from an image folder.")
    ap.add_argument("--input-dir", type=Path, default=DATA_DIR / "raw" / "lfw")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "face_index")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--split", default="index")
    ap.add_argument("--no-identity-from-parent", action="store_true")
    ap.add_argument(
        "--inventory-csv",
        type=Path,
        default=None,
        help="Optional CSV whose image_id values replace generated path hashes.",
    )
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
    inventory_path = args.inventory_csv.resolve() if args.inventory_csv else None
    inventory_ids = (
        load_inventory_image_ids(inventory_path, input_dir, image_paths)
        if inventory_path
        else None
    )

    app = build_face_app(device=args.device, det_size=args.det_size)

    rows = []
    embs = []
    n_read_fail = 0
    n_no_face = 0
    n_no_emb = 0

    for img_path in tqdm(image_paths, desc="Indexing faces", unit="img"):
        img = read_image_bgr(img_path)
        if img is None:
            n_read_fail += 1
            continue
        h, w = img.shape[:2]
        image_id = (
            inventory_ids[img_path.resolve()]
            if inventory_ids is not None
            else stable_image_id(img_path, input_dir)
        )
        image_sha256 = sha256_file(img_path)
        identity = "" if args.no_identity_from_parent else parent_label(img_path)

        faces = sorted_faces(app.get(img))
        if not faces:
            n_no_face += 1
            continue

        for face_index, face in enumerate(faces):
            emb = embedding_from_face(face)
            if emb is None:
                n_no_emb += 1
                continue
            row_idx = len(embs)
            x1, y1, x2, y2 = [float(v) for v in face.bbox]
            face_id = f"{image_id}_f{face_index:02d}"
            rows.append(
                {
                    "face_id": face_id,
                    "image_id": image_id,
                    "image_path": rel_to_root(img_path),
                    "sha256": image_sha256,
                    "face_index": face_index,
                    "bbox_x1": round(x1, 3),
                    "bbox_y1": round(y1, 3),
                    "bbox_x2": round(x2, 3),
                    "bbox_y2": round(y2, 3),
                    "det_score": float(getattr(face, "det_score", np.nan)),
                    "embedding_row": row_idx,
                    "identity": identity,
                    "split": args.split,
                    "width": int(w),
                    "height": int(h),
                }
            )
            embs.append(emb.astype(np.float32))

    if not embs:
        print("[ERRO] No face embeddings were extracted.")
        print(f"[INFO] read_fail={n_read_fail} no_face={n_no_face} no_emb={n_no_emb}")
        return 1

    E = np.vstack(embs).astype(np.float32)
    meta = pd.DataFrame(rows)
    embeddings_path = out_dir / "face_embeddings.npy"
    metadata_path = out_dir / "face_metadata.csv"
    np.save(embeddings_path, E)
    meta.to_csv(metadata_path, index=False)
    manifest_path = write_run_manifest(
        out_dir / "face_index_manifest.json",
        script_path=Path(__file__),
        method="face_index",
        configuration={
            "max_images": args.max_images,
            "device": args.device,
            "det_size": args.det_size,
            "split": args.split,
            "identity_from_parent": not args.no_identity_from_parent,
            "image_id_source": "inventory_csv" if inventory_path else "stable_path_hash",
        },
        inputs={"input_directory": input_dir, "inventory_csv": inventory_path},
        result_files={"embeddings": embeddings_path, "metadata": metadata_path},
        extra={
            "images_scanned": len(image_paths),
            "face_embeddings": len(meta),
            "read_failures": n_read_fail,
            "images_without_face": n_no_face,
            "faces_without_embedding": n_no_emb,
        },
    )

    print(f"[OK] images scanned: {len(image_paths)}")
    print(f"[OK] face embeddings: {len(meta)}")
    print(f"[OK] read_fail={n_read_fail} no_face={n_no_face} no_emb={n_no_emb}")
    print(f"[OK] {embeddings_path}")
    print(f"[OK] {metadata_path}")
    print(f"[OK] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
