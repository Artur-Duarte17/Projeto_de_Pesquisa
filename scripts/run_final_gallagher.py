from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from project_paths import DATA_DIR, OUTPUTS_DIR
from retrieval.adapters.run_paths import require_empty_target, resolve_run_root
from retrieval.adapters.manifest import write_run_manifest


FINAL_ROOT = OUTPUTS_DIR / "final" / "gallagher"
EVALUATION_DIR = DATA_DIR / "evaluation" / "gallagher_final"
QUERY_DIR = DATA_DIR / "query" / "gallagher_final"


class GallagherPaths:
    def __init__(
        self,
        root: Path,
        final: Path,
        evaluation: Path,
        query: Path,
        environment: Path,
    ) -> None:
        self.root = root
        self.final = final
        self.evaluation = evaluation
        self.query = query
        self.environment = environment


def paths_for_run(run_root: Path | None = None) -> GallagherPaths:
    root = resolve_run_root(
        run_root,
        project_root=ROOT,
        outputs_root=OUTPUTS_DIR,
        baseline_root=OUTPUTS_DIR / "final",
    )
    return GallagherPaths(
        root=root,
        final=root / "gallagher",
        evaluation=EVALUATION_DIR if run_root is None else root / "protocols" / "gallagher",
        query=QUERY_DIR if run_root is None else root / "queries" / "gallagher",
        environment=root / "environment_report.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete corrected Gallagher pipeline for the final project version."
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Fresh directory under outputs/; Gallagher outputs go to <run-root>/gallagher.",
    )
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
    run_root: Path | None = None,
) -> list[list[str]]:
    paths = paths_for_run(run_root)
    images = DATA_DIR / "raw" / "gallagher" / "images"
    face_index = paths.final / "face_index"
    global_index = paths.final / "global_index"
    evaluation = paths.final / "evaluation"
    commands: list[list[str]] = []
    if include_environment_check:
        commands.append(
            [
                python_executable,
                str(ROOT / "scripts" / "validate_environment.py"),
                "--output",
                str(paths.environment),
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
                    str(paths.evaluation),
                    "--query-crop-dir",
                    str(paths.query),
                    "--manifest-dir",
                    str(paths.final / "face_protocol"),
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
                    str(paths.evaluation / "gallagher_face_queries.csv"),
                    "--relevance-csv",
                    str(paths.evaluation / "gallagher_relevance.csv"),
                    "--face-index-dir",
                    str(face_index),
                    "--global-index-dir",
                    str(global_index),
                    "--output-csv",
                    str(paths.evaluation / "gallagher_fusion_queries.csv"),
                    "--query-manifest-csv",
                    str(paths.evaluation / "gallagher_fusion_query_manifest.csv"),
                    "--manifest-dir",
                    str(paths.final / "paired_protocol"),
                ],
                overwrite,
            ),
            add_overwrite(
                [
                    python_executable,
                    str(ROOT / "scripts" / "fusion" / "04_evaluate_paired_gallagher.py"),
                    "--queries-csv",
                    str(paths.evaluation / "gallagher_fusion_queries.csv"),
                    "--relevance-csv",
                    str(paths.evaluation / "gallagher_relevance.csv"),
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
                str(paths.evaluation / "gallagher_fusion_queries.csv"),
                "--relevance-csv",
                str(paths.evaluation / "gallagher_relevance.csv"),
                "--output-dir",
                str(paths.final / "error_analysis"),
            ]
        )
    return commands


def protected_outputs(run_root: Path | None = None) -> list[Path]:
    paths = paths_for_run(run_root)
    return [
        paths.final / "face_index" / "face_embeddings.npy",
        paths.final / "global_index" / "global_embeddings.npy",
        paths.final / "evaluation" / "fusion_metrics.csv",
        paths.final / "final_pipeline_manifest.json",
        paths.evaluation / "gallagher_face_queries.csv",
        paths.evaluation / "gallagher_fusion_queries.csv",
    ]


def main() -> int:
    args = parse_args()
    if args.run_root is not None and args.overwrite:
        raise ValueError("--run-root cannot be combined with --overwrite; choose a fresh directory")
    paths = paths_for_run(args.run_root)
    if args.run_root is not None:
        for target in (paths.final, paths.evaluation, paths.query):
            require_empty_target(target)
        if not args.skip_environment_check:
            require_empty_target(paths.environment)
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
    existing = [path for path in protected_outputs(args.run_root) if path.exists()]
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
        run_root=args.run_root,
    )
    for command in commands:
        print(f"[RUN] {subprocess.list2cmdline(command)}")
        subprocess.run(command, cwd=ROOT, check=True)

    result_files = {
        "face_index_manifest": paths.final / "face_index" / "face_index_manifest.json",
        "global_index_manifest": paths.final / "global_index" / "global_index_manifest.json",
        "face_protocol_manifest": paths.final / "face_protocol" / "gallagher_protocol_manifest.json",
        "paired_protocol_manifest": paths.final / "paired_protocol" / "paired_protocol_manifest.json",
        "fusion_run_manifest": paths.final / "evaluation" / "fusion_run_manifest.json",
        "fusion_metrics": paths.final / "evaluation" / "fusion_metrics.csv",
        "fusion_metrics_per_query": paths.final / "evaluation" / "fusion_metrics_per_query.csv",
    }
    if not args.skip_environment_check:
        result_files["environment"] = paths.environment
    manifest_path = write_run_manifest(
        paths.final / "final_pipeline_manifest.json",
        script_path=Path(__file__),
        method="final_corrected_gallagher_pipeline",
        configuration={
            "device": args.device,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "det_size": args.det_size,
            "run_root": str(paths.root),
            "isolated_run": args.run_root is not None,
            "environment_check": not args.skip_environment_check,
            "error_analysis": not args.skip_error_analysis,
        },
        inputs={"images": images, "annotations": annotations},
        result_files=result_files,
    )
    print(f"[OK] Final Gallagher pipeline: {paths.final}")
    print(f"[OK] Final manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
