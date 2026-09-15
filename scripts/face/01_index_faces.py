from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from face_lib import build_face_app, embedding_from_face, sorted_faces
from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval_common import list_images, parent_label, rel_to_root, stable_image_id


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Index all detected faces from an image folder.")
    ap.add_argument("--input-dir", type=Path, default=DATA_DIR / "raw" / "lfw")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "face_index")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--split", default="index")
    ap.add_argument("--no-identity-from-parent", action="store_true")
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

    app = build_face_app(device=args.device, det_size=args.det_size)

    rows = []
    embs = []
    n_read_fail = 0
    n_no_face = 0
    n_no_emb = 0

    for img_path in tqdm(image_paths, desc="Indexing faces", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            n_read_fail += 1
            continue
        h, w = img.shape[:2]
        image_id = stable_image_id(img_path, input_dir)
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
    np.save(out_dir / "face_embeddings.npy", E)
    meta.to_csv(out_dir / "face_metadata.csv", index=False)

    print(f"[OK] images scanned: {len(image_paths)}")
    print(f"[OK] face embeddings: {len(meta)}")
    print(f"[OK] read_fail={n_read_fail} no_face={n_no_face} no_emb={n_no_emb}")
    print(f"[OK] {out_dir / 'face_embeddings.npy'}")
    print(f"[OK] {out_dir / 'face_metadata.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
