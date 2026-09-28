from __future__ import annotations

import ast
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from retrieval.adapters.search_gateway import search_face_index
from retrieval.application.search import rank_face_index, rank_fused_index, rank_global_index


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    return modules


def imports_any(module: str, forbidden: tuple[str, ...]) -> bool:
    return any(module == name or module.startswith(name + ".") for name in forbidden)


class DependencyBoundaryTests(unittest.TestCase):
    def test_active_code_does_not_import_removed_facades(self) -> None:
        removed = {
            "face_lib",
            "global_lib",
            "fusion_lib",
            "retrieval_common",
            "run_manifest",
        }
        paths = sorted(SCRIPTS.rglob("*.py")) + sorted((ROOT / "tests").glob("test_*.py"))
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertEqual(imported_modules(path) & removed, set())

    def test_domain_does_not_depend_on_outer_layers_or_model_frameworks(self) -> None:
        forbidden = (
            "retrieval.adapters",
            "retrieval.application",
            "adapters",
            "application",
            "cv2",
            "torch",
            "torchvision",
            "streamlit",
            "project_paths",
        )
        files = sorted((SCRIPTS / "retrieval" / "domain").glob("*.py"))
        self.assertTrue(files, "Domain package is missing")
        for path in files:
            with self.subTest(path=path.name):
                violations = [
                    module for module in imported_modules(path) if imports_any(module, forbidden)
                ]
                self.assertEqual(violations, [], f"{path} imports an outer dependency")

    def test_application_does_not_import_adapters(self) -> None:
        files = sorted((SCRIPTS / "retrieval" / "application").glob("*.py"))
        self.assertTrue(files, "Application package is missing")
        for path in files:
            with self.subTest(path=path.name):
                violations = [
                    module
                    for module in imported_modules(path)
                    if imports_any(module, ("retrieval.adapters", "adapters"))
                ]
                self.assertEqual(violations, [], f"{path} imports an adapter")


class SearchUseCaseTests(unittest.TestCase):
    def test_face_ranking_excludes_source_and_keeps_best_face_per_photo(self) -> None:
        embeddings = np.asarray(
            [[1.0, 0.0], [0.6, 0.8], [0.8, 0.6], [0.0, 1.0]],
            dtype=np.float32,
        )
        metadata = pd.DataFrame(
            [
                self._face_row("source_f00", "source"),
                self._face_row("keep_f00", "keep"),
                self._face_row("keep_f01", "keep"),
                self._face_row("other_f00", "other"),
            ]
        )

        results = rank_face_index(
            np.asarray([1.0, 0.0], dtype=np.float32),
            embeddings,
            metadata,
            topk=None,
            excluded_image_ids={"source"},
        )

        self.assertEqual(results["image_id"].tolist(), ["keep", "other"])
        self.assertEqual(results["matched_face_id"].tolist(), ["keep_f01", "other_f00"])
        self.assertEqual(results["rank"].tolist(), [1, 2])
        self.assertAlmostEqual(float(results.iloc[0]["score"]), 0.8, places=5)

    def test_global_ranking_excludes_source_before_topk(self) -> None:
        embeddings = np.asarray(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32
        )
        metadata = pd.DataFrame(
            [
                {"image_id": "source", "image_path": "source.jpg"},
                {"image_id": "keep", "image_path": "keep.jpg"},
                {"image_id": "other", "image_path": "other.jpg"},
            ]
        )

        results = rank_global_index(
            np.asarray([1.0, 0.0], dtype=np.float32),
            embeddings,
            metadata,
            topk=1,
            excluded_image_ids={"source"},
        )

        self.assertEqual(results["image_id"].tolist(), ["keep"])
        self.assertEqual(results["rank"].tolist(), [1])

    def test_unlimited_rankings_are_not_capped_at_fifty_photos(self) -> None:
        count = 73
        embeddings = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (count, 1))
        face_metadata = pd.DataFrame(
            [self._face_row(f"image_{number}_f00", f"image_{number}") for number in range(count)]
        )
        global_metadata = face_metadata[["image_id", "image_path"]].copy()
        query = np.asarray([1.0, 0.0], dtype=np.float32)

        face_results = rank_face_index(query, embeddings, face_metadata, topk=None)
        global_results = rank_global_index(query, embeddings, global_metadata, topk=None)
        fused_results = rank_fused_index(
            query, query, embeddings, face_metadata, embeddings, global_metadata,
            topk=None, face_weight=0.7, global_weight=0.3,
        )

        for results in (face_results, global_results, fused_results):
            self.assertEqual(len(results), count)
            self.assertEqual(results["rank"].tolist(), list(range(1, count + 1)))

    def test_face_gateway_excludes_uploaded_copy_by_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            source = folder / "source.jpg"
            copy = folder / "upload.jpg"
            other = folder / "other.jpg"
            source.write_bytes(b"same photograph")
            copy.write_bytes(b"same photograph")
            other.write_bytes(b"different photograph")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            other_hash = hashlib.sha256(other.read_bytes()).hexdigest()

            metadata = pd.DataFrame(
                [
                    {**self._face_row("source_f00", "source", source), "sha256": source_hash},
                    {**self._face_row("source_f01", "source", source), "sha256": source_hash},
                    {**self._face_row("other_f00", "other", other), "sha256": other_hash},
                ]
            )
            embeddings = np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32
            )

            results = search_face_index(
                np.asarray([1.0, 0.0], dtype=np.float32),
                embeddings,
                metadata,
                topk=None,
                query_path=copy,
            )

            self.assertEqual(results["image_id"].tolist(), ["other"])

    @staticmethod
    def _face_row(face_id: str, image_id: str, image_path: Path | None = None) -> dict[str, object]:
        return {
            "face_id": face_id,
            "image_id": image_id,
            "image_path": str(image_path) if image_path else f"{image_id}.jpg",
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_x2": 10,
            "bbox_y2": 10,
        }


if __name__ == "__main__":
    unittest.main()
