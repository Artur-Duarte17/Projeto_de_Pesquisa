from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from global_lib import (
    build_resnet50_feature_extractor,
    describe_torch_device,
    extract_global_embedding,
    load_global_index,
    search_global_index,
)
from project_paths import OUTPUTS_DIR
from retrieval_common import now_ms, save_visual_grid


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search visually similar whole images.")
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, default=OUTPUTS_DIR / "global_index")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "global_index")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=-1.0, help="Reserved for CLI compatibility.")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--max-images", type=int, default=None, help="Reserved for CLI compatibility.")
    ap.add_argument("--save-visual", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, metadata = load_global_index(args.index_dir)
    extractor, transform, device = build_resnet50_feature_extractor(args.device, args.weights)
    print(f"[INFO] Torch device: {describe_torch_device(args.device, device)}")

    t0 = now_ms()
    q = extract_global_embedding(args.query, extractor, transform, device)
    results = search_global_index(q, embeddings, metadata, topk=args.topk, query_path=args.query)
    elapsed = now_ms() - t0

    query_id = args.query.stem
    results.insert(0, "query_id", query_id)
    results.insert(1, "query_path", str(args.query))
    results["query_time_ms"] = elapsed
    out_csv = args.output_dir / "global_topk_results.csv"
    results.to_csv(out_csv, index=False)
    if args.save_visual:
        save_visual_grid(results, args.output_dir / "global_topk_results.png", f"Global search: {query_id}")

    print(f"[OK] Results: {out_csv}")
    print(f"[OK] Query time ms: {elapsed:.2f}")
    print(results[["rank", "score", "image_path"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
