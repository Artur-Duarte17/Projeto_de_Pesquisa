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


FINAL_ROOT = OUTPUTS_DIR / "final"
LFW_ROOT = FINAL_ROOT / "lfw"
LFW_PROTOCOL = DATA_DIR / "evaluation" / "lfw_final"
HOLIDAYS_ROOT = FINAL_ROOT / "holidays"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete final scientific validation: LFW, INRIA Holidays "
            "and corrected Gallagher. Dataset acquisition is intentionally excluded."
        )
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-gallagher-error-analysis", action="store_true")
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
    evaluation_batch_size: int,
    workers: int,
    det_size: int,
    overwrite: bool,
    allow_dirty: bool,
    include_gallagher_error_analysis: bool,
) -> list[list[str]]:
    lfw_images = DATA_DIR / "raw" / "lfw" / "lfw_home" / "lfw_funneled"
    holidays_images = DATA_DIR / "raw" / "holidays" / "images"
    commands = [
        [
            python_executable,
            str(ROOT / "scripts" / "validate_environment.py"),
            "--output",
            str(FINAL_ROOT / "environment_report.json"),
            *(["--require-cuda"] if device == "cuda" else []),
        ],
        [
            python_executable,
            str(ROOT / "scripts" / "face" / "01_index_faces.py"),
            "--input-dir",
            str(lfw_images),
            "--output-dir",
            str(LFW_ROOT / "face_index"),
            "--device",
            device,
            "--det-size",
            str(det_size),
        ],
        add_overwrite(
            [
                python_executable,
                str(ROOT / "scripts" / "face" / "06_prepare_lfw_eval.py"),
                "--index-dir",
                str(LFW_ROOT / "face_index"),
                "--output-dir",
                str(LFW_PROTOCOL),
                "--manifest-dir",
                str(LFW_ROOT / "protocol"),
            ],
            overwrite,
        ),
        [
            python_executable,
            str(ROOT / "scripts" / "face" / "07_evaluate_lfw.py"),
            "--queries-csv",
            str(LFW_PROTOCOL / "lfw_face_queries.csv"),
            "--relevance-csv",
            str(LFW_PROTOCOL / "lfw_face_relevance.csv"),
            "--index-dir",
            str(LFW_ROOT / "face_index"),
            "--output-dir",
            str(LFW_ROOT / "evaluation"),
            "--device",
            device,
            "--batch-size",
            str(evaluation_batch_size),
        ],
        [
            python_executable,
            str(ROOT / "scripts" / "global" / "01_index_global_resnet.py"),
            "--input-dir",
            str(holidays_images),
            "--output-dir",
            str(HOLIDAYS_ROOT / "global_index"),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
            "--workers",
            str(workers),
            "--no-label-from-parent",
        ],
        [
            python_executable,
            str(ROOT / "scripts" / "global" / "04_evaluate_holidays.py"),
            "--queries-csv",
            str(DATA_DIR / "evaluation" / "holidays_global_queries.csv"),
            "--relevance-csv",
            str(DATA_DIR / "evaluation" / "holidays_relevance.csv"),
            "--index-dir",
            str(HOLIDAYS_ROOT / "global_index"),
            "--output-dir",
            str(HOLIDAYS_ROOT / "evaluation"),
            "--device",
            device,
            "--batch-size",
            str(evaluation_batch_size),
        ],
        [
            python_executable,
            str(ROOT / "scripts" / "run_final_gallagher.py"),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
            "--workers",
            str(workers),
            "--det-size",
            str(det_size),
            "--skip-environment-check",
            *(["--overwrite"] if overwrite else []),
            *(["--allow-dirty"] if allow_dirty else []),
            *(
                []
                if include_gallagher_error_analysis
                else ["--skip-error-analysis"]
            ),
        ],
    ]
    return commands


def required_inputs() -> list[Path]:
    return [
        DATA_DIR / "raw" / "lfw" / "lfw_home" / "lfw_funneled",
        DATA_DIR / "raw" / "holidays" / "images",
        DATA_DIR / "evaluation" / "holidays_global_queries.csv",
        DATA_DIR / "evaluation" / "holidays_relevance.csv",
        DATA_DIR / "raw" / "gallagher" / "images",
        DATA_DIR / "raw" / "gallagher" / "metadata" / "face_annotations.csv",
    ]


def protected_outputs() -> list[Path]:
    return [
        FINAL_ROOT / "environment_report.json",
        LFW_ROOT / "face_index" / "face_embeddings.npy",
        LFW_ROOT / "evaluation" / "face_metrics.csv",
        HOLIDAYS_ROOT / "global_index" / "global_embeddings.npy",
        HOLIDAYS_ROOT / "evaluation" / "global_metrics.csv",
        FINAL_ROOT / "gallagher" / "final_pipeline_manifest.json",
        FINAL_ROOT / "final_validation_manifest.json",
        LFW_PROTOCOL / "lfw_face_queries.csv",
    ]


def main() -> int:
    args = parse_args()
    numeric_values = (
        args.batch_size,
        args.evaluation_batch_size,
        args.workers,
        args.det_size,
    )
    if any(value <= 0 for value in numeric_values):
        raise ValueError("Batch sizes, workers and det-size must be positive")
    if git_is_dirty() and not args.allow_dirty:
        raise RuntimeError(
            "The working tree is dirty. Commit the final code before the scientific run, "
            "or use --allow-dirty only for a non-final diagnostic run."
        )

    missing = [path for path in required_inputs() if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required final-validation inputs are missing: {joined}")
    existing = [path for path in protected_outputs() if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Final outputs already exist; use --overwrite: {joined}")

    commands = build_commands(
        python_executable=sys.executable,
        device=args.device,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        workers=args.workers,
        det_size=args.det_size,
        overwrite=args.overwrite,
        allow_dirty=args.allow_dirty,
        include_gallagher_error_analysis=not args.skip_gallagher_error_analysis,
    )
    for command in commands:
        print(f"[RUN] {subprocess.list2cmdline(command)}")
        subprocess.run(command, cwd=ROOT, check=True)

    manifest_path = write_run_manifest(
        FINAL_ROOT / "final_validation_manifest.json",
        script_path=Path(__file__),
        method="complete_final_scientific_validation",
        configuration={
            "device": args.device,
            "batch_size": args.batch_size,
            "evaluation_batch_size": args.evaluation_batch_size,
            "workers": args.workers,
            "det_size": args.det_size,
            "datasets": ["LFW", "INRIA Holidays", "Gallagher"],
            "dataset_acquisition_included": False,
        },
        inputs={
            "lfw_images": required_inputs()[0],
            "holidays_images": required_inputs()[1],
            "holidays_queries": required_inputs()[2],
            "holidays_relevance": required_inputs()[3],
            "gallagher_images": required_inputs()[4],
            "gallagher_annotations": required_inputs()[5],
        },
        result_files={
            "environment": FINAL_ROOT / "environment_report.json",
            "lfw_index_manifest": LFW_ROOT / "face_index" / "face_index_manifest.json",
            "lfw_protocol_manifest": LFW_ROOT / "protocol" / "lfw_protocol_manifest.json",
            "lfw_metrics": LFW_ROOT / "evaluation" / "face_metrics.csv",
            "holidays_index_manifest": HOLIDAYS_ROOT / "global_index" / "global_index_manifest.json",
            "holidays_metrics": HOLIDAYS_ROOT / "evaluation" / "global_metrics.csv",
            "gallagher_pipeline_manifest": FINAL_ROOT / "gallagher" / "final_pipeline_manifest.json",
            "gallagher_metrics": FINAL_ROOT / "gallagher" / "evaluation" / "fusion_metrics.csv",
        },
    )
    print(f"[OK] Complete final validation: {FINAL_ROOT}")
    print(f"[OK] Final validation manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
