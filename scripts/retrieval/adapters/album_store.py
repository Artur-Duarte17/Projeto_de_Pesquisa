"""Local album discovery and explicit, user-triggered preparation."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COLLECTIONS_DIR = ROOT / "outputs" / "collections"
INCOMPLETE_MARKER = ".preparation_incomplete.json"
REQUIRED_FILES = (
    "collection_manifest.json",
    "face/face_embeddings.npy",
    "face/face_metadata.csv",
    "face/face_index_manifest.json",
    "global/global_embeddings.npy",
    "global/global_metadata.csv",
    "global/global_index_manifest.json",
)
GENERATED_FILES = frozenset((*REQUIRED_FILES, INCOMPLETE_MARKER, "global/global_failures.csv"))
GENERATED_DIRECTORIES = frozenset({"face", "global"})
WINDOWS_RESERVED_NAMES = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}
_PREPARATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class Album:
    name: str
    output_dir: Path
    photo_dir: Path | None
    ready: bool

    @property
    def face_index(self) -> Path:
        return self.output_dir / "face"

    @property
    def global_index(self) -> Path:
        return self.output_dir / "global"


class AlbumPreparationError(RuntimeError):
    def __init__(self, message: str, log: str = ""):
        super().__init__(message)
        self.log = log


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _validate_output(output_dir: Path, collections_dir: Path) -> Path:
    resolved = output_dir.resolve()
    root = collections_dir.resolve()
    if resolved.parent != root:
        raise ValueError("O álbum deve ficar dentro da área de coleções da aplicação")
    return resolved


def album_output_dir(name: str, collections_dir: Path = COLLECTIONS_DIR) -> Path:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")[:64]
    if not slug or slug in WINDOWS_RESERVED_NAMES:
        raise ValueError("Escolha outro nome para o álbum")
    return _validate_output(collections_dir / slug, collections_dir)


def album_is_ready(output_dir: Path) -> bool:
    return not (output_dir / INCOMPLETE_MARKER).exists() and all(
        (output_dir / name).is_file() for name in REQUIRED_FILES
    )


def discover_albums(collections_dir: Path = COLLECTIONS_DIR) -> list[Album]:
    if not collections_dir.is_dir():
        return []
    albums = []
    for directory in sorted(collections_dir.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        try:
            directory = _validate_output(directory, collections_dir)
        except ValueError:
            continue
        manifest = _read_json(directory / "collection_manifest.json")
        incomplete = _read_json(directory / INCOMPLETE_MARKER)
        if not manifest and not incomplete:
            continue
        name = incomplete.get("name") or manifest.get("configuration", {}).get("collection_name")
        name = str(name or directory.name.replace("-", " ").capitalize())
        stored_path = incomplete.get("photo_dir") or (
            manifest.get("inputs", {}).get("input_directory") or {}
        ).get("path")
        photo_dir = Path(stored_path) if stored_path else None
        if photo_dir is not None and not photo_dir.is_absolute():
            photo_dir = ROOT / photo_dir
        albums.append(Album(name, directory, photo_dir, album_is_ready(directory)))
    return albums


def preparation_command(
    photo_dir: Path,
    output_dir: Path,
    name: str,
    device: str,
    overwrite: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_collection.py"),
        "--input-dir", str(photo_dir),
        "--output-dir", str(output_dir),
        "--collection-name", name,
        "--device", device,
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def prepare_album(
    photo_dir: Path,
    output_dir: Path,
    name: str,
    *,
    device: str = "cpu",
    overwrite: bool = False,
    collections_dir: Path = COLLECTIONS_DIR,
) -> str:
    if not _PREPARATION_LOCK.acquire(blocking=False):
        raise ValueError("Outro álbum já está sendo preparado. Aguarde o término")
    try:
        return _prepare_album(photo_dir, output_dir, name, device, overwrite, collections_dir)
    finally:
        _PREPARATION_LOCK.release()


def _prepare_album(
    photo_dir: Path,
    output_dir: Path,
    name: str,
    device: str,
    overwrite: bool,
    collections_dir: Path,
) -> str:
    photo_dir = photo_dir.resolve()
    output_dir = _validate_output(output_dir, collections_dir)
    if not photo_dir.is_dir():
        raise ValueError("A pasta de fotos não existe neste computador")
    if not name.strip():
        raise ValueError("Informe um nome para o álbum")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Dispositivo inválido")
    if output_dir == photo_dir or photo_dir in output_dir.parents:
        raise ValueError("Os índices não podem ser gravados dentro da pasta de fotos")
    if output_dir.exists() and not overwrite:
        raise ValueError("Esse álbum já existe; selecione-o para atualizar")

    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / INCOMPLETE_MARKER
    marker.write_text(
        json.dumps({"name": name.strip(), "photo_dir": str(photo_dir)}, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            preparation_command(photo_dir, output_dir, name.strip(), device, overwrite),
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    except OSError as exc:
        raise AlbumPreparationError("Não foi possível iniciar a preparação do álbum", str(exc)) from exc
    log = completed.stdout + "\n" + completed.stderr
    if completed.returncode != 0 or not all((output_dir / p).is_file() for p in REQUIRED_FILES):
        # Keep the marker: an interrupted/failed update must not mix old and
        # new indexes in the normal search screen. A retry updates this album.
        raise AlbumPreparationError("A preparação não terminou; tente novamente antes de buscar", log)
    marker.unlink()
    return log


def album_summary(album: Album) -> dict:
    face = _read_json(album.face_index / "face_index_manifest.json").get("extra", {})
    global_data = _read_json(album.global_index / "global_index_manifest.json").get("extra", {})
    return {
        "images_scanned": global_data.get("images_scanned"),
        "global_failures": global_data.get("failures", 0),
        "face_read_failures": face.get("read_failures", 0),
    }


def _is_link_or_junction(path: Path) -> bool:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def remove_album(
    output_dir: Path,
    *,
    confirmed: bool = False,
    collections_dir: Path = COLLECTIONS_DIR,
) -> None:
    """Remove only a recognized generated album after explicit confirmation.

    Original-photo directories, links/junctions and unexpected files are never
    recursively removed. Sharing the preparation lock prevents an update and
    removal from running together in this local application process.
    """
    if not confirmed:
        raise ValueError("Confirme a remoção dos índices deste álbum")
    if not _PREPARATION_LOCK.acquire(blocking=False):
        raise ValueError("Um álbum está sendo preparado. Aguarde antes de remover")
    try:
        if _is_link_or_junction(output_dir):
            raise ValueError("Não é permitido remover um álbum por um atalho ou vínculo de pasta")
        output_dir = _validate_output(output_dir, collections_dir)
        manifest = _read_json(output_dir / "collection_manifest.json")
        incomplete = _read_json(output_dir / INCOMPLETE_MARKER)
        recognized_manifest = manifest.get("method") == "photo_collection_preparation"
        recognized_incomplete = bool(incomplete.get("name") and incomplete.get("photo_dir"))
        if not recognized_manifest and not recognized_incomplete:
            raise ValueError("A pasta não foi reconhecida como um álbum preparado pela aplicação")
        stored_paths = (
            incomplete.get("photo_dir"),
            (manifest.get("inputs", {}).get("input_directory") or {}).get("path"),
        )
        for stored_path in stored_paths:
            if not stored_path:
                continue
            photo_dir = Path(stored_path)
            if not photo_dir.is_absolute():
                photo_dir = ROOT / photo_dir
            photo_dir = photo_dir.resolve()
            if photo_dir == output_dir or output_dir in photo_dir.parents:
                raise ValueError("A pasta de fotos originais não pode ser removida pela aplicação")

        # Validate the exact resolved target and every entry before recursive
        # deletion. Unexpected contents may belong to the user, not the app.
        def fail_on_walk_error(exc: OSError) -> None:
            raise exc

        for parent, directories, files in os.walk(output_dir, followlinks=False, onerror=fail_on_walk_error):
            for name in (*directories, *files):
                path = Path(parent) / name
                if _is_link_or_junction(path):
                    raise ValueError("O álbum contém um vínculo de pasta ou arquivo; remoção cancelada")
                relative = path.relative_to(output_dir).as_posix()
                allowed = GENERATED_DIRECTORIES if name in directories else GENERATED_FILES
                if relative not in allowed:
                    raise ValueError("O álbum contém arquivos não gerados pela preparação; remoção cancelada")
        shutil.rmtree(output_dir)
    finally:
        _PREPARATION_LOCK.release()
