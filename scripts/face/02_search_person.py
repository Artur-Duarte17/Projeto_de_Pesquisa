from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from face_lib import (
    build_face_app,
    load_face_index,
    query_embedding_from_image,
    query_id_from_path,
    search_face_index,
)
from project_paths import OUTPUTS_DIR
from retrieval_common import now_ms, save_visual_grid


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search full photos containing a reference person.")
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, default=OUTPUTS_DIR / "face_index")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "face_index")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--query-face-mode", choices=["largest", "index"], default="largest")
    ap.add_argument("--query-face-index", type=int, default=0)
    ap.add_argument("--save-visual", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, metadata = load_face_index(args.index_dir)
    app = build_face_app(device=args.device, det_size=args.det_size)

    t0 = now_ms()
    q, query_bbox = query_embedding_from_image(
        args.query,
        app,
        query_face_mode=args.query_face_mode,
        query_face_index=args.query_face_index,
    )
    results = search_face_index(
        q,
        embeddings,
        metadata,
        topk=args.topk,
        threshold=args.threshold,
        query_path=args.query,
    )
    elapsed = now_ms() - t0

    query_id = query_id_from_path(args.query)
    results.insert(0, "query_id", query_id)
    results.insert(1, "query_path", str(args.query))
    results.insert(2, "query_bbox", query_bbox)
    results["query_time_ms"] = elapsed

    out_csv = args.output_dir / "topk_results.csv"
    results.to_csv(out_csv, index=False)
    if args.save_visual:
        save_visual_grid(results, args.output_dir / "topk_results.png", f"Face search: {query_id}")

    pd.set_option("display.max_colwidth", 120)
    print(f"[OK] Results: {out_csv}")
    print(f"[OK] Query time ms: {elapsed:.2f}")
    print(results[["rank", "score", "image_path", "matched_face_id"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
