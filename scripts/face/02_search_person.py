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
from run_manifest import write_run_manifest


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
    ap.add_argument("--exclude-image-id", action="append", default=[])
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
        exclude_image_ids=set(args.exclude_image_id),
    )
    elapsed = now_ms() - t0

    query_id = query_id_from_path(args.query)
    results.insert(0, "query_id", query_id)
    results.insert(1, "query_path", str(args.query))
    results.insert(2, "query_bbox", query_bbox)
    results["query_time_ms"] = elapsed

    out_csv = args.output_dir / "topk_results.csv"
    results.to_csv(out_csv, index=False)
    visual_path = args.output_dir / "topk_results.png"
    if args.save_visual:
        save_visual_grid(results, visual_path, f"Face search: {query_id}")
    result_files = {"topk": out_csv}
    if args.save_visual and visual_path.exists():
        result_files["visual"] = visual_path
    manifest_path = write_run_manifest(
        args.output_dir / "face_search_manifest.json",
        script_path=Path(__file__),
        method="face_search",
        configuration={
            "topk": args.topk,
            "threshold": args.threshold,
            "device": args.device,
            "det_size": args.det_size,
            "query_face_mode": args.query_face_mode,
            "query_face_index": args.query_face_index,
            "exclude_image_ids": sorted(set(args.exclude_image_id)),
        },
        inputs={
            "query": args.query,
            "face_embeddings": args.index_dir / "face_embeddings.npy",
            "face_metadata": args.index_dir / "face_metadata.csv",
        },
        result_files=result_files,
        extra={"query_bbox": query_bbox, "query_time_ms": elapsed},
    )

    pd.set_option("display.max_colwidth", 120)
    print(f"[OK] Results: {out_csv}")
    print(f"[OK] Query time ms: {elapsed:.2f}")
    print(f"[OK] Run manifest: {manifest_path}")
    print(results[["rank", "score", "image_path", "matched_face_id"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
