from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval.adapters.album_store import (
    INCOMPLETE_MARKER,
    REQUIRED_FILES,
    AlbumPreparationError,
    album_is_ready,
    album_output_dir,
    discover_albums,
    preparation_command,
    prepare_album,
    remove_album,
)


def make_ready_album(output_dir: Path, photo_dir: Path, name: str | None = None) -> None:
    for relative in REQUIRED_FILES:
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "synthetic test data", encoding="utf-8")
    (output_dir / "collection_manifest.json").write_text(
        json.dumps({
            "method": "photo_collection_preparation",
            "configuration": {"collection_name": name},
            "inputs": {"input_directory": {"path": str(photo_dir)}},
        }),
        encoding="utf-8",
    )


class AlbumPreparationTests(unittest.TestCase):
    def test_name_maps_to_safe_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(album_output_dir("Minha família!", root), root / "minha-familia")
            for bad in ("", "...", "NUL", "CON", "COM1"):
                with self.subTest(name=bad), self.assertRaises(ValueError):
                    album_output_dir(bad, root)

    def test_existing_cli_album_is_discovered_without_new_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "familia"
            photos = root / "source"
            make_ready_album(output, photos)
            albums = discover_albums(root)
            self.assertEqual(len(albums), 1)
            self.assertEqual(albums[0].name, "Familia")
            self.assertEqual(albums[0].photo_dir, photos)
            self.assertTrue(albums[0].ready)
            self.assertEqual(albums[0].face_index, output / "face")

    def test_preparation_command_reuses_project_python_and_cli(self):
        command = preparation_command(Path("photos"), Path("indexes"), "Família", "cpu", False)
        self.assertEqual(command[0], sys.executable)
        self.assertIn("--collection-name", command)
        self.assertNotIn("--overwrite", command)
        self.assertNotIn("--max-images", command)
        update = preparation_command(Path("photos"), Path("indexes"), "Família", "cpu", True)
        self.assertEqual(update[-1], "--overwrite")

    def test_successful_preparation_preserves_photos_and_removes_incomplete_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            root = task / "collections"
            photos = task / "photos"
            photos.mkdir()
            original = photos / "photo.jpg"
            original.write_bytes(b"original photograph")
            output = root / "familia"

            def fake_run(command, **kwargs):
                self.assertEqual(kwargs["cwd"], ROOT)
                self.assertTrue((output / INCOMPLETE_MARKER).exists())
                make_ready_album(output, photos, "Família")
                return subprocess.CompletedProcess(command, 0, "prepared", "")

            with mock.patch("retrieval.adapters.album_store.subprocess.run", side_effect=fake_run):
                log = prepare_album(photos, output, "Família", collections_dir=root)
            self.assertIn("prepared", log)
            self.assertTrue(album_is_ready(output))
            self.assertEqual(original.read_bytes(), b"original photograph")
            self.assertEqual(discover_albums(root)[0].name, "Família")

    def test_failed_update_is_not_exposed_as_a_ready_album(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            root = task / "collections"
            photos = task / "photos"
            photos.mkdir()
            output = root / "familia"
            make_ready_album(output, photos, "Família")
            with mock.patch(
                "retrieval.adapters.album_store.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1, "", "decoder failed"),
            ):
                with self.assertRaises(AlbumPreparationError) as raised:
                    prepare_album(photos, output, "Família", overwrite=True, collections_dir=root)
            self.assertIn("decoder failed", raised.exception.log)
            self.assertFalse(album_is_ready(output))
            self.assertFalse(discover_albums(root)[0].ready)

    def test_new_album_does_not_overwrite_an_existing_album(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photos = root / "photos"
            photos.mkdir()
            output = root / "familia"
            make_ready_album(output, photos)
            with mock.patch("retrieval.adapters.album_store.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "já existe"):
                    prepare_album(photos, output, "Família", collections_dir=root)
            run.assert_not_called()
            self.assertTrue(album_is_ready(output))

    def test_output_outside_collection_area_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            photos = task / "photos"
            photos.mkdir()
            with mock.patch("retrieval.adapters.album_store.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    prepare_album(photos, task / "outside", "Album", collections_dir=task / "collections")
            run.assert_not_called()

    def test_incomplete_new_album_can_be_found_for_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "familia"
            output.mkdir()
            (output / INCOMPLETE_MARKER).write_text(
                json.dumps({"name": "Família", "photo_dir": str(root / "photos")}), encoding="utf-8"
            )
            albums = discover_albums(root)
            self.assertEqual(albums[0].name, "Família")
            self.assertFalse(albums[0].ready)


class AlbumRemovalTests(unittest.TestCase):
    def test_confirmed_removal_preserves_original_photos_and_other_albums(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            root = task / "collections"
            photos = task / "photos"
            photos.mkdir()
            original = photos / "photo.heic"
            original.write_bytes(b"original photograph")
            output = root / "familia"
            other = root / "outro"
            make_ready_album(output, photos, "Família")
            make_ready_album(other, photos, "Outro")
            (output / "global/global_failures.csv").write_text("image,error", encoding="utf-8")

            remove_album(output, confirmed=True, collections_dir=root)

            self.assertFalse(output.exists())
            self.assertTrue(album_is_ready(other))
            self.assertEqual(original.read_bytes(), b"original photograph")
            self.assertEqual([album.name for album in discover_albums(root)], ["Outro"])

    def test_missing_confirmation_does_not_remove_an_album(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "familia"
            make_ready_album(output, root / "photos")
            with mock.patch("retrieval.adapters.album_store.shutil.rmtree") as removal:
                with self.assertRaisesRegex(ValueError, "Confirme"):
                    remove_album(output, collections_dir=root)
            removal.assert_not_called()
            self.assertTrue(album_is_ready(output))

    def test_root_and_paths_outside_collection_area_cannot_be_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            root = task / "collections"
            outside = task / "outside"
            root.mkdir()
            make_ready_album(outside, task / "photos")
            for target in (root, outside):
                with self.subTest(target=target), mock.patch(
                    "retrieval.adapters.album_store.shutil.rmtree"
                ) as removal:
                    with self.assertRaises(ValueError):
                        remove_album(target, confirmed=True, collections_dir=root)
                    removal.assert_not_called()
            self.assertTrue(root.is_dir())
            self.assertTrue(album_is_ready(outside))

    def test_unexpected_files_cancel_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "familia"
            make_ready_album(output, root / "photos")
            original = output / "photo.jpg"
            original.write_bytes(b"not generated by the application")
            with mock.patch("retrieval.adapters.album_store.shutil.rmtree") as removal:
                with self.assertRaisesRegex(ValueError, "não gerados"):
                    remove_album(output, confirmed=True, collections_dir=root)
            removal.assert_not_called()
            self.assertEqual(original.read_bytes(), b"not generated by the application")
            self.assertTrue(album_is_ready(output))

    def test_original_photo_directory_inside_album_cancels_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "familia"
            make_ready_album(output, output / "originals")
            with mock.patch("retrieval.adapters.album_store.shutil.rmtree") as removal:
                with self.assertRaisesRegex(ValueError, "fotos originais"):
                    remove_album(output, confirmed=True, collections_dir=root)
            removal.assert_not_called()

    def test_links_and_windows_junctions_cancel_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "familia"
            make_ready_album(output, root / "photos")
            for linked_path in (output, output / "face"):
                with self.subTest(linked_path=linked_path), mock.patch(
                    "retrieval.adapters.album_store._is_link_or_junction",
                    side_effect=lambda path: path == linked_path,
                ), mock.patch("retrieval.adapters.album_store.shutil.rmtree") as removal:
                    with self.assertRaises(ValueError):
                        remove_album(output, confirmed=True, collections_dir=root)
                    removal.assert_not_called()
            self.assertTrue(album_is_ready(output))

    def test_incomplete_album_can_also_be_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "familia"
            output.mkdir()
            (output / INCOMPLETE_MARKER).write_text(
                json.dumps({"name": "Família", "photo_dir": str(root / "photos")}), encoding="utf-8"
            )
            remove_album(output, confirmed=True, collections_dir=root)
            self.assertFalse(output.exists())

    def test_album_cannot_be_removed_during_preparation(self):
        with mock.patch("retrieval.adapters.album_store._PREPARATION_LOCK") as lock:
            lock.acquire.return_value = False
            with mock.patch("retrieval.adapters.album_store.shutil.rmtree") as removal:
                with self.assertRaisesRegex(ValueError, "sendo preparado"):
                    remove_album(Path("unused"), confirmed=True)
            removal.assert_not_called()
            lock.release.assert_not_called()

    def test_unrecognized_folder_is_not_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "unrelated"
            output.mkdir()
            (output / "collection_manifest.json").write_text('{"unrelated": true}', encoding="utf-8")
            with mock.patch("retrieval.adapters.album_store.shutil.rmtree") as removal:
                with self.assertRaisesRegex(ValueError, "não foi reconhecida"):
                    remove_album(output, confirmed=True, collections_dir=root)
            removal.assert_not_called()
            self.assertTrue(output.is_dir())

    def test_unreadable_contents_cancel_removal_before_deleting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "familia"
            make_ready_album(output, root / "photos")

            def unreadable_walk(path, *, followlinks, onerror):
                onerror(PermissionError("Cannot inspect generated album contents"))

            with mock.patch("retrieval.adapters.album_store.os.walk", side_effect=unreadable_walk), mock.patch(
                "retrieval.adapters.album_store.shutil.rmtree"
            ) as removal:
                with self.assertRaises(PermissionError):
                    remove_album(output, confirmed=True, collections_dir=root)
            removal.assert_not_called()
            self.assertTrue(album_is_ready(output))
