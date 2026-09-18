from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval_common import sha256_file
from run_manifest import write_run_manifest
from widerface_audit import audit_sample, fixed_sample
from widerface_data import prepare_dataset
from widerface_eval_lib import (
    evaluate_subset,
    evaluate_wider_official,
    load_difficulty_indices,
    parse_wider_annotations,
    predictions_from_flat_arrays,
    recall_by_categories,
    valid_boxes,
)


EXPECTED_IMAGES = 3226
ALLOWED_PREEXISTING_DIRTY = {
    "scripts/docs/build_final_documents.py",
    "scripts/docs/audit_references.py",
    "scripts/docs/download_references.py",
    "scripts/docs/generate_qualitative_figures.py",
    "scripts/docs/update_reference_links.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen EX-029 to EX-031 WIDER FACE validation cycle."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / ".insightface" / "models" / "buffalo_l" / "det_10g.onnx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "ex-031_widerface_validation",
    )
    return parser.parse_args()


def _git_status() -> tuple[str, list[str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    lines = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    paths = [line[3:].replace("\\", "/") for line in lines if len(line) >= 4]
    unexpected = [path for path in paths if path not in ALLOWED_PREEXISTING_DIRTY]
    if unexpected:
        raise RuntimeError(f"Unexpected dirty paths before experiment: {unexpected}")
    return commit, paths


def _load_detector(model_path: Path):
    import onnxruntime as ort
    import torch  # Loads the CUDA/cuDNN DLLs bundled with the approved environment.
    from insightface.model_zoo.scrfd import SCRFD

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    session = ort.InferenceSession(
        str(model_path), providers=["CUDAExecutionProvider"]
    )
    detector = SCRFD(model_file=str(model_path), session=session)
    if detector.taskname != "detection":
        raise RuntimeError(f"Expected a detection model; got {detector.taskname}")
    detector.prepare(ctx_id=0, input_size=(640, 640))
    providers = detector.session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA is mandatory; active providers are {providers}")
    return detector


def _model_audit(model_path: Path, detector) -> dict[str, object]:
    graph = onnx.load(str(model_path), load_external_data=False).graph
    outputs = [output.name for output in graph.output]
    forbidden = [name for name in outputs if "embed" in name.lower() or "recogn" in name.lower()]
    if forbidden:
        raise RuntimeError(f"Detector graph unexpectedly exposes embedding outputs: {forbidden}")
    return {
        "path": str(model_path.resolve()),
        "bytes": model_path.stat().st_size,
        "sha256": sha256_file(model_path),
        "runtime_class": f"{type(detector).__module__}.{type(detector).__name__}",
        "taskname": detector.taskname,
        "providers": detector.session.get_providers(),
        "provider_options": detector.session.get_provider_options(),
        "onnx_outputs": outputs,
        "embedding_outputs": forbidden,
    }


def _detect(detector, image: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    detections, _landmarks = detector.detect(
        image, threshold=threshold, input_size=(640, 640), max_num=0
    )
    detections = np.asarray(detections, dtype=np.float32).reshape((-1, 5))
    return detections[:, :4], detections[:, 4]


def _equivalence_check(detector, images_root: Path, keys: list[str]) -> dict[str, object]:
    maximum_box_difference = 0.0
    maximum_score_difference = 0.0
    counts: dict[str, int] = {}
    for key in keys:
        image = cv2.imread(str(images_root / key))
        if image is None:
            raise FileNotFoundError(images_root / key)
        floor_boxes, floor_scores = _detect(detector, image, 0.02)
        selected = floor_scores >= 0.50
        derived_boxes = floor_boxes[selected]
        derived_scores = floor_scores[selected]
        direct_boxes, direct_scores = _detect(detector, image, 0.50)
        if derived_boxes.shape != direct_boxes.shape or derived_scores.shape != direct_scores.shape:
            raise AssertionError(f"Threshold equivalence count mismatch for {key}")
        if len(direct_boxes):
            maximum_box_difference = max(
                maximum_box_difference, float(np.max(np.abs(derived_boxes - direct_boxes)))
            )
            maximum_score_difference = max(
                maximum_score_difference, float(np.max(np.abs(derived_scores - direct_scores)))
            )
        if not np.allclose(derived_boxes, direct_boxes, rtol=1e-5, atol=1e-4):
            raise AssertionError(f"Threshold equivalence box mismatch for {key}")
        if not np.allclose(derived_scores, direct_scores, rtol=1e-6, atol=1e-6):
            raise AssertionError(f"Threshold equivalence score mismatch for {key}")
        counts[key] = len(direct_boxes)
    return {
        "images": keys,
        "counts_at_0_50": counts,
        "maximum_box_difference": maximum_box_difference,
        "maximum_score_difference": maximum_score_difference,
        "box_atol": 1e-4,
        "score_atol": 1e-6,
        "passed": True,
    }


def _run_full_inference(detector, images_root: Path, keys: list[str]) -> dict[str, np.ndarray]:
    flat_boxes: list[np.ndarray] = []
    flat_scores: list[np.ndarray] = []
    offsets = [0]
    inference_ms: list[float] = []
    image_hashes: list[str] = []
    for number, key in enumerate(keys, start=1):
        path = images_root / key
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        started = time.perf_counter()
        boxes, scores = _detect(detector, image, 0.02)
        inference_ms.append((time.perf_counter() - started) * 1000.0)
        if not valid_boxes(boxes).all() or not np.isfinite(scores).all():
            raise ValueError(f"Invalid detector output for {key}")
        if np.any(scores < 0.02):
            raise ValueError(f"Detector returned a score below the frozen floor for {key}")
        flat_boxes.append(boxes)
        flat_scores.append(scores)
        offsets.append(offsets[-1] + len(boxes))
        image_hashes.append(sha256_file(path))
        if number % 100 == 0 or number == len(keys):
            print(f"[EX-030] processed {number}/{len(keys)} images", flush=True)
    return {
        "image_keys": np.asarray(keys),
        "image_sha256": np.asarray(image_hashes),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "boxes": np.vstack(flat_boxes).astype(np.float32) if flat_boxes else np.empty((0, 4), np.float32),
        "scores": np.concatenate(flat_scores).astype(np.float32) if flat_scores else np.empty(0, np.float32),
        "inference_ms": np.asarray(inference_ms, dtype=np.float64),
    }


def _jsonable_metrics(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"matched_gt", "curve_recall", "curve_precision"}
    }


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to mix results in existing output directory: {output_dir}")
    commit, dirty_paths = _git_status()
    source_manifest_path = DATA_DIR / "evaluation" / "widerface_sources.json"
    download_dir = DATA_DIR / "raw" / "widerface_downloads"
    dataset_root = DATA_DIR / "raw" / "widerface_validation"
    acquisition_audit_path = output_dir / "EX-029_source_and_dataset_audit.json"
    acquisition = prepare_dataset(
        source_manifest_path, download_dir, dataset_root, acquisition_audit_path
    )
    source_manifest = acquisition["source_manifest"]
    expected_model_hash = source_manifest["protocol"]["model_sha256"]
    if sha256_file(model_path).lower() != expected_model_hash.lower():
        raise ValueError("det_10g.onnx does not match the frozen SHA-256")
    images_root = Path(acquisition["extracted"]["images_root"])
    annotations_path = Path(acquisition["extracted"]["annotations_path"])
    ground_truth_dir = Path(acquisition["extracted"]["ground_truth_dir"])
    ground_truth = parse_wider_annotations(annotations_path)
    keys = sorted(ground_truth)
    if len(keys) != EXPECTED_IMAGES:
        raise AssertionError(f"Expected exactly {EXPECTED_IMAGES} images")

    detector = _load_detector(model_path)
    model_audit = _model_audit(model_path, detector)
    if model_audit["runtime_class"] != source_manifest["protocol"]["model_runtime_class"]:
        raise RuntimeError(f"Unexpected detector runtime class: {model_audit['runtime_class']}")

    sample_keys = fixed_sample(keys, 20, seed="EX-030-equivalence-20260918")
    equivalence = _equivalence_check(detector, images_root, sample_keys)
    equivalence_path = output_dir / "EX-030_detector_equivalence.json"
    equivalence_path.write_text(json.dumps(equivalence, indent=2, sort_keys=True), encoding="utf-8")

    smoke_key = fixed_sample(keys, 1, seed="EX-030-smoke-20260918")[0]
    smoke_image = cv2.imread(str(images_root / smoke_key))
    if smoke_image is None:
        raise FileNotFoundError(images_root / smoke_key)
    smoke_boxes, smoke_scores = _detect(detector, smoke_image, 0.02)
    smoke = {"image": smoke_key, "detections": len(smoke_boxes), "finite_scores": bool(np.isfinite(smoke_scores).all())}
    if not smoke["finite_scores"]:
        raise ValueError("Smoke test produced non-finite scores")
    smoke_path = output_dir / "EX-030_one_image_smoke.json"
    smoke_path.write_text(json.dumps(smoke, indent=2, sort_keys=True), encoding="utf-8")

    arrays = _run_full_inference(detector, images_root, keys)
    predictions_path = output_dir / "EX-030_widerface_predictions.npz"
    np.savez_compressed(predictions_path, **arrays)
    predictions = predictions_from_flat_arrays(
        arrays["image_keys"].tolist(), arrays["offsets"], arrays["boxes"], arrays["scores"]
    )
    if set(predictions) != set(ground_truth):
        raise AssertionError("Missing prediction files")

    levels = {
        level: load_difficulty_indices(ground_truth_dir / f"wider_{level}_val.mat")
        for level in ("easy", "medium", "hard")
    }
    difficulty_results = {
        level: evaluate_wider_official(predictions, ground_truth, keep, iou_threshold=0.5)
        for level, keep in levels.items()
    }
    valid_gt = {
        key: {index for index, attrs in enumerate(gt.attributes) if int(attrs[3]) == 0}
        for key, gt in ground_truth.items()
    }
    operating = evaluate_subset(
        predictions, ground_truth, valid_gt, iou_threshold=0.5, score_threshold=0.5
    )
    categories = recall_by_categories(ground_truth, operating["matched_gt"])
    audit = audit_sample(
        predictions,
        {key: value.boxes for key, value in ground_truth.items()},
        valid_gt,
        fixed_sample(keys, 20, seed="EX-031-audit-20260918"),
        0.5,
    )
    repeated_audit = audit_sample(
        predictions,
        {key: value.boxes for key, value in ground_truth.items()},
        valid_gt,
        audit["sample_images"],
        0.5,
    )
    if audit != repeated_audit:
        raise AssertionError("Fixed-sample audit was not deterministic")

    metrics = {
        "ap": {level: result["ap"] for level, result in difficulty_results.items()},
        "operating_threshold_0_50": _jsonable_metrics(operating),
        "mean_inference_ms_per_image": float(np.mean(arrays["inference_ms"])),
        "median_inference_ms_per_image": float(np.median(arrays["inference_ms"])),
        "images_processed": len(keys),
        "detections_at_score_floor": len(arrays["scores"]),
        "score_floor": 0.02,
        "iou_threshold": 0.5,
        "categories": categories,
    }
    if not np.isfinite(
        [*metrics["ap"].values(), metrics["mean_inference_ms_per_image"], metrics["median_inference_ms_per_image"]]
    ).all():
        raise ValueError("Non-finite aggregate metrics")
    metrics_path = output_dir / "EX-031_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    category_path = output_dir / "EX-031_category_recall.csv"
    with category_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["category", "value", "ground_truth", "matched", "recall"])
        writer.writeheader()
        writer.writerows(categories)
    audit_path = output_dir / "EX-031_independent_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    curve_path = output_dir / "EX-031_precision_recall_curves.png"
    plt.figure(figsize=(7.2, 5.2))
    for level, result in difficulty_results.items():
        plt.plot(result["curve_recall"], result["curve_precision"], label=f"{level.title()} AP={result['ap']:.6f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1.01)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=180)
    plt.close()

    manifest_path = write_run_manifest(
        output_dir / "EX-031_manifest.json",
        script_path=Path(__file__),
        method="widerface_validation_face_detection",
        configuration=source_manifest["protocol"],
        inputs={
            "source_manifest": source_manifest_path,
            "model": model_path,
            "annotations": annotations_path,
            "ground_truth_directory": ground_truth_dir,
        },
        result_files={
            "acquisition_audit": acquisition_audit_path,
            "equivalence": equivalence_path,
            "smoke": smoke_path,
            "predictions": predictions_path,
            "metrics": metrics_path,
            "category_recall": category_path,
            "independent_audit": audit_path,
            "precision_recall_curves": curve_path,
        },
        extra={
            "frozen_commit": commit,
            "preexisting_unrelated_dirty_paths": dirty_paths,
            "experiment_relevant_dirty": False,
            "model_audit": model_audit,
            "images_processed": len(keys),
            "embeddings_generated": 0,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
