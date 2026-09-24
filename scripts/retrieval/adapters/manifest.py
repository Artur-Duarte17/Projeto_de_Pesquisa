from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from retrieval.adapters.files import rel_to_root, sha256_file


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def _artifact_entry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        return {"path": rel_to_root(resolved), "exists": False}
    if resolved.is_dir():
        return {"path": rel_to_root(resolved), "exists": True, "type": "directory"}
    return {
        "path": rel_to_root(resolved),
        "exists": True,
        "type": "file",
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_run_manifest(
    output_path: Path,
    *,
    script_path: Path,
    method: str,
    configuration: Mapping[str, Any],
    inputs: Mapping[str, Path | None],
    result_files: Mapping[str, Path],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    root = Path(__file__).resolve().parents[3]
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": rel_to_root(script_path),
        "method": method,
        "configuration": dict(configuration),
        "inputs": {
            name: None if path is None else _artifact_entry(path)
            for name, path in inputs.items()
        },
        "results": {
            name: _artifact_entry(path)
            for name, path in result_files.items()
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in ("numpy", "pandas", "torch", "torchvision", "insightface", "onnxruntime-gpu")
            },
        },
        "git": _git_state(root),
    }
    if extra:
        manifest["extra"] = dict(extra)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return output_path
