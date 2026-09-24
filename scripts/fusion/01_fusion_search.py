from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from face_lib import build_face_app, load_face_index, query_embedding_from_image
from fusion_lib import search_fusion
from global_lib import (
    build_resnet50_feature_extractor,
    extract_global_embedding,
    load_global_index,
)
from project_paths import OUTPUTS_DIR
from retrieval_common import now_ms, save_visual_grid
from run_manifest import write_run_manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search with a weighted fusion of face and global scores.")
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--face-index-dir", type=Path, default=OUTPUTS_DIR / "face_index")
    ap.add_argument("--global-index-dir", type=Path, default=OUTPUTS_DIR / "global_index")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "fusion")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=-1.0)
    ap.add_argument("--face-weight", type=float, default=0.7)
    ap.add_argument("--global-weight", type=float, default=0.3)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--query-face-mode", choices=["largest", "index"], default="largest")
    ap.add_argument("--query-face-index", type=int, default=0)
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--max-images", type=int, default=None, help="Reserved for CLI compatibility.")
    ap.add_argument("--exclude-image-id", action="append", default=[])
    ap.add_argument("--save-visual", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    face_E, face_meta = load_face_index(args.face_index_dir)
    global_E, global_meta = load_global_index(args.global_index_dir)
    face_app = build_face_app(device=args.device, det_size=args.det_size)
    extractor, transform, device = build_resnet50_feature_extractor(args.device, args.weights)

    t0 = now_ms()
    face_q, query_bbox = query_embedding_from_image(
        args.query,
        face_app,
        query_face_mode=args.query_face_mode,
        query_face_index=args.query_face_index,
    )
    global_q = extract_global_embedding(args.query, extractor, transform, device)
    results = search_fusion(
        args.query,
        face_q,
        global_q,
        face_E,
        face_meta,
        global_E,
        global_meta,
        topk=args.topk,
        face_weight=args.face_weight,
        global_weight=args.global_weight,
        threshold=args.threshold,
        exclude_image_ids=set(args.exclude_image_id),
    )
    elapsed = now_ms() - t0

    query_id = args.query.stem
    results.insert(0, "query_id", query_id)
    results.insert(1, "query_path", str(args.query))
    results.insert(2, "query_bbox", query_bbox)
    results["query_time_ms"] = elapsed
    out_csv = args.output_dir / "fusion_results.csv"
    results.to_csv(out_csv, index=False)
    visual_path = args.output_dir / "fusion_results.png"
    if args.save_visual:
        save_visual_grid(results, visual_path, f"Fusion search: {query_id}")
    result_files = {"topk": out_csv}
    if args.save_visual and visual_path.exists():
        result_files["visual"] = visual_path
    manifest_path = write_run_manifest(
        args.output_dir / "fusion_search_manifest.json",
        script_path=Path(__file__),
        method="fusion_search",
        configuration={
            "topk": args.topk,
            "threshold": args.threshold,
            "face_weight": args.face_weight,
            "global_weight": args.global_weight,
            "device": args.device,
            "det_size": args.det_size,
            "query_face_mode": args.query_face_mode,
            "query_face_index": args.query_face_index,
            "weights": args.weights,
            "exclude_image_ids": sorted(set(args.exclude_image_id)),
        },
        inputs={
            "query": args.query,
            "face_embeddings": args.face_index_dir / "face_embeddings.npy",
            "face_metadata": args.face_index_dir / "face_metadata.csv",
            "global_embeddings": args.global_index_dir / "global_embeddings.npy",
            "global_metadata": args.global_index_dir / "global_metadata.csv",
        },
        result_files=result_files,
        extra={"query_bbox": query_bbox, "query_time_ms": elapsed},
    )

    print(f"[OK] Results: {out_csv}")
    print(f"[OK] Query time ms: {elapsed:.2f}")
    print(f"[OK] Run manifest: {manifest_path}")
    print(results[["rank", "score", "face_score", "global_score", "image_path"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
