"""Resolve isolated validation outputs without touching the frozen baseline."""

from __future__ import annotations

from pathlib import Path


def resolve_run_root(
    requested: Path | None,
    *,
    project_root: Path,
    outputs_root: Path,
    baseline_root: Path,
) -> Path:
    if requested is None:
        return baseline_root
    candidate = (requested if requested.is_absolute() else project_root / requested).resolve()
    outputs = outputs_root.resolve()
    baseline = baseline_root.resolve()
    if candidate == outputs or outputs not in candidate.parents:
        raise ValueError("--run-root must be a child directory of outputs/")
    if candidate == baseline or baseline in candidate.parents:
        raise ValueError("--run-root must not be outputs/final or a directory inside it")
    return candidate


def require_empty_target(path: Path) -> None:
    """Refuse an old or partial run before any subprocess can replace its files."""
    if path.is_file():
        raise FileExistsError(f"Validation target is a file: {path}")
    if path.is_dir() and any(path.iterdir()):
        raise FileExistsError(f"Validation target is not empty: {path}")
