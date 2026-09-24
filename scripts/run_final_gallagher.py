from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import DATA_DIR, OUTPUTS_DIR
from run_manifest import write_run_manifest


FINAL_ROOT = OUTPUTS_DIR / "final" / "gallagher"
EVALUATION_DIR = DATA_DIR / "evaluation" / "gallagher_final"
QUERY_DIR = DATA_DIR / "query" / "gallagher_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete corrected Gallagher pipeline for the final project version."
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-environment-check", action="store_true")
    parser.add_argument("--skip-error-analysis", action="store_true")
    return parser.parse_args()


def git_is_dirty() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())


def add_overwrite(command: list[str], overwrite: bool) -> list[str]:
    return [*command, "--overwrite"] if overwrite else command


def build_commands(
    *,
    python_executable: str,
    device: str,
    batch_size: int,
    workers: int,
    det_size: int,
    overwrite: bool,
    include_environment_check: bool,
    include_error_analysis: bool,
) -> list[list[str]]:
    images = DATA_DIR / "raw" / "gallagher" / "images"
    face_index = FINAL_ROOT / "face_index"
    global_index = FINAL_ROOT / "global_index"
    evaluation = FINAL_ROOT / "evaluation"
    commands: list[list[str]] = []
    if include_environment_check:
        commands.append(
            [
                python_executable,
                str(ROOT / "scripts" / "validate_environment.py"),
                "--output",
                str(OUTPUTS_DIR / "final" / "environment_report.json"),
                *(["--require-cuda"] if device == "cuda" else []),
            ]
        )
    commands.extend(
        [
            [
                python_executable,
                str(ROOT / "scripts" / "face" / "01_index_faces.py"),
                "--input-dir",
                str(images),
                "--output-dir",
                str(face_index),
                "--device",
                device,
                "--det-size",
                str(det_size),
                "--no-identity-from-parent",
            ],
            [
                python_executable,
                str(ROOT / "scripts" / "global" / "01_index_global_resnet.py"),
                "--input-dir",
                str(images),
                "--output-dir",
                str(global_index),
                "--device",
                device,
                "--batch-size",
                str(batch_size),
                "--workers",
                str(workers),
                "--no-label-from-parent",
            ],
            add_overwrite(
                [
                    python_executable,
                    str(ROOT / "scripts" / "face" / "04_prepare_gallagher_eval.py"),
                    "--index-dir",
                    str(face_index),
                    "--gallery-index-dir",
                    str(global_index),
                    "--output-dir",
                    str(EVALUATION_DIR),
                    "--query-crop-dir",
                    str(QUERY_DIR),
                    "--manifest-dir",
                    str(FINAL_ROOT / "face_protocol"),
                    "--device",
                    device,
                    "--det-size",
                    str(det_size),
                ],
                overwrite,
            ),
            add_overwrite(
                [
                    python_executable,
                    str(ROOT / "scripts" / "fusion" / "03_prepare_paired_gallagher.py"),
                    "--face-queries-csv",
                    str(EVALUATION_DIR / "gallagher_face_queries.csv"),
                    "--relevance-csv",
                    str(EVALUATION_DIR / "gallagher_relevance.csv"),
                    "--face-index-dir",
                    str(face_index),
                    "--global-index-dir",
                    str(global_index),
                    "--output-csv",
                    str(EVALUATION_DIR / "gallagher_fusion_queries.csv"),
                    "--query-manifest-csv",
                    str(EVALUATION_DIR / "gallagher_fusion_query_manifest.csv"),
                    "--manifest-dir",
                    str(FINAL_ROOT / "paired_protocol"),
                ],
                overwrite,
            ),
            add_overwrite(
                [
                    python_executable,
                    str(ROOT / "scripts" / "fusion" / "04_evaluate_paired_gallagher.py"),
                    "--queries-csv",
                    str(EVALUATION_DIR / "gallagher_fusion_queries.csv"),
                    "--relevance-csv",
                    str(EVALUATION_DIR / "gallagher_relevance.csv"),
                    "--face-index-dir",
                    str(face_index),
                    "--global-index-dir",
                    str(global_index),
                    "--output-dir",
                    str(evaluation),
                    "--device",
                    device,
                    "--det-size",
                    str(det_size),
                    "--batch-size",
                    str(min(batch_size, 20)),
                    "--dataset-name",
                    "gallagher_final",
                ],
                overwrite,
            ),
        ]
    )
    if include_error_analysis:
        commands.append(
            [
                python_executable,
                str(ROOT / "scripts" / "fusion" / "05_analyze_paired_fusion_errors.py"),
                "--results-csv",
                str(evaluation / "fusion_topk_results.csv"),
                "--per-query-csv",
                str(evaluation / "fusion_metrics_per_query.csv"),
                "--aggregate-csv",
                str(evaluation / "fusion_metrics.csv"),
                "--queries-csv",
                str(EVALUATION_DIR / "gallagher_fusion_queries.csv"),
                "--relevance-csv",
                str(EVALUATION_DIR / "gallagher_relevance.csv"),
                "--output-dir",
                str(FINAL_ROOT / "error_analysis"),
            ]
        )
    return commands


def protected_outputs() -> list[Path]:
    return [
        FINAL_ROOT / "face_index" / "face_embeddings.npy",
        FINAL_ROOT / "global_index" / "global_embeddings.npy",
        FINAL_ROOT / "evaluation" / "fusion_metrics.csv",
        FINAL_ROOT / "final_pipeline_manifest.json",
        EVALUATION_DIR / "gallagher_face_queries.csv",
        EVALUATION_DIR / "gallagher_fusion_queries.csv",
    ]


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers <= 0 or args.det_size <= 0:
        raise ValueError("batch-size, workers and det-size must be positive")
    if git_is_dirty() and not args.allow_dirty:
        raise RuntimeError(
            "The working tree is dirty. Commit the final code before the scientific run, "
            "or use --allow-dirty only for a non-final diagnostic run."
        )
    images = DATA_DIR / "raw" / "gallagher" / "images"
    annotations = DATA_DIR / "raw" / "gallagher" / "metadata" / "face_annotations.csv"
    if not images.is_dir() or not annotations.is_file():
        raise FileNotFoundError("Gallagher images or official annotations are missing")
    existing = [path for path in protected_outputs() if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Final outputs already exist; use --overwrite: {joined}")

    commands = build_commands(
        python_executable=sys.executable,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        det_size=args.det_size,
        overwrite=args.overwrite,
        include_environment_check=not args.skip_environment_check,
        include_error_analysis=not args.skip_error_analysis,
    )
    for command in commands:
        print(f"[RUN] {subprocess.list2cmdline(command)}")
        subprocess.run(command, cwd=ROOT, check=True)

    result_files = {
        "face_index_manifest": FINAL_ROOT / "face_index" / "face_index_manifest.json",
        "global_index_manifest": FINAL_ROOT / "global_index" / "global_index_manifest.json",
        "face_protocol_manifest": FINAL_ROOT / "face_protocol" / "gallagher_protocol_manifest.json",
        "paired_protocol_manifest": FINAL_ROOT / "paired_protocol" / "paired_protocol_manifest.json",
        "fusion_run_manifest": FINAL_ROOT / "evaluation" / "fusion_run_manifest.json",
        "fusion_metrics": FINAL_ROOT / "evaluation" / "fusion_metrics.csv",
        "fusion_metrics_per_query": FINAL_ROOT / "evaluation" / "fusion_metrics_per_query.csv",
    }
    if not args.skip_environment_check:
        result_files["environment"] = OUTPUTS_DIR / "final" / "environment_report.json"
    manifest_path = write_run_manifest(
        FINAL_ROOT / "final_pipeline_manifest.json",
        script_path=Path(__file__),
        method="final_corrected_gallagher_pipeline",
        configuration={
            "device": args.device,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "det_size": args.det_size,
            "environment_check": not args.skip_environment_check,
            "error_analysis": not args.skip_error_analysis,
        },
        inputs={"images": images, "annotations": annotations},
        result_files=result_files,
    )
    print(f"[OK] Final Gallagher pipeline: {FINAL_ROOT}")
    print(f"[OK] Final manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
