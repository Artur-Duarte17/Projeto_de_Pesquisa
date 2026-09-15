from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from face_lib import search_face_index
from fusion_lib import search_fusion
from global_lib import search_global_index
from retrieval_common import (
    apply_relevance_exclusions,
    average_precision,
    build_query_exclusions,
    precision_at_k,
    select_results_for_storage,
    sha256_file,
    summarize_rankings,
)
from run_manifest import write_run_manifest


class MetricTests(unittest.TestCase):
    def test_precision_at_k_uses_requested_k_as_denominator(self) -> None:
        self.assertAlmostEqual(precision_at_k(["relevant"], {"relevant"}, 5), 0.2)

    def test_average_precision_uses_full_relevance_denominator(self) -> None:
        ranked = ["irrelevant", "a", "b"]
        expected = ((1 / 2) + (2 / 3)) / 3
        self.assertAlmostEqual(average_precision(ranked, {"a", "b", "missing"}), expected)

    def test_summary_records_integral_ranking_size(self) -> None:
        summary = summarize_rankings({"q1": ["a", "b", "c"]}, {"q1": {"a"}}, ks=(2,))
        self.assertEqual(int(summary.loc[0, "ranking_size"]), 3)
        self.assertAlmostEqual(float(summary.loc[0, "precision_at_2"]), 0.5)

    def test_storage_limit_does_not_modify_integral_ranking(self) -> None:
        ranking = pd.DataFrame({"image_id": ["a", "b", "c"]})
        saved = select_results_for_storage(ranking, 2)
        self.assertEqual(ranking["image_id"].tolist(), ["a", "b", "c"])
        self.assertEqual(saved["image_id"].tolist(), ["a", "b"])


class ExclusionTests(unittest.TestCase):
    def test_query_exclusions_apply_to_explicit_relevance(self) -> None:
        queries = pd.DataFrame(
            [
                {
                    "query_id": "q1",
                    "query_path": "missing.jpg",
                    "source_image_id": "source",
                    "exclude_image_ids": "blocked_a;blocked_b",
                }
            ]
        )
        metadata = pd.DataFrame(columns=["image_id", "image_path"])
        exclusions = build_query_exclusions(queries, metadata)
        cleaned = apply_relevance_exclusions(
            {"q1": {"source", "blocked_a", "kept"}},
            exclusions,
        )
        self.assertEqual(exclusions["q1"], {"source", "blocked_a", "blocked_b"})
        self.assertEqual(cleaned["q1"], {"kept"})

    def test_global_search_excludes_uploaded_copy_by_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            source = folder / "source.jpg"
            uploaded_copy = folder / "uploaded.jpg"
            other = folder / "other.jpg"
            source.write_bytes(b"same image bytes")
            uploaded_copy.write_bytes(b"same image bytes")
            other.write_bytes(b"different image bytes")

            metadata = pd.DataFrame(
                [
                    {"image_id": "source", "image_path": str(source), "sha256": sha256_file(source)},
                    {"image_id": "other", "image_path": str(other), "sha256": sha256_file(other)},
                ]
            )
            embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            result = search_global_index(
                np.asarray([1.0, 0.0], dtype=np.float32),
                embeddings,
                metadata,
                topk=None,
                query_path=uploaded_copy,
            )
            self.assertEqual(result["image_id"].tolist(), ["other"])

    def test_face_search_excludes_all_faces_from_source_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            source = folder / "source.jpg"
            other = folder / "other.jpg"
            source.write_bytes(b"source")
            other.write_bytes(b"other")
            metadata = pd.DataFrame(
                [
                    self._face_row("source_f00", "source", source),
                    self._face_row("source_f01", "source", source),
                    self._face_row("other_f00", "other", other),
                ]
            )
            embeddings = np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
                dtype=np.float32,
            )
            result = search_face_index(
                np.asarray([1.0, 0.0], dtype=np.float32),
                embeddings,
                metadata,
                topk=None,
                query_path=source,
            )
            self.assertEqual(result["image_id"].tolist(), ["other"])

    def test_fusion_propagates_explicit_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            query = folder / "query.jpg"
            source = folder / "source.jpg"
            other = folder / "other.jpg"
            for path, content in ((query, b"query"), (source, b"source"), (other, b"other")):
                path.write_bytes(content)

            face_meta = pd.DataFrame(
                [
                    self._face_row("source_f00", "source", source),
                    self._face_row("other_f00", "other", other),
                ]
            )
            global_meta = pd.DataFrame(
                [
                    {"image_id": "source", "image_path": str(source)},
                    {"image_id": "other", "image_path": str(other)},
                ]
            )
            embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            result = search_fusion(
                query,
                np.asarray([1.0, 0.0], dtype=np.float32),
                np.asarray([1.0, 0.0], dtype=np.float32),
                embeddings,
                face_meta,
                embeddings,
                global_meta,
                topk=None,
                face_weight=0.7,
                global_weight=0.3,
                exclude_image_ids={"source"},
            )
            self.assertEqual(result["image_id"].tolist(), ["other"])

    @staticmethod
    def _face_row(face_id: str, image_id: str, image_path: Path) -> dict[str, object]:
        return {
            "face_id": face_id,
            "image_id": image_id,
            "image_path": str(image_path),
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_x2": 10,
            "bbox_y2": 10,
        }


class ManifestTests(unittest.TestCase):
    def test_manifest_records_hashes_and_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            input_path = folder / "queries.csv"
            result_path = folder / "metrics.csv"
            manifest_path = folder / "run_manifest.json"
            input_path.write_text("query_id\nq1\n", encoding="utf-8")
            result_path.write_text("mAP\n1.0\n", encoding="utf-8")

            write_run_manifest(
                manifest_path,
                script_path=Path(__file__),
                method="synthetic",
                configuration={"save_topk": 1, "ranking_depth": "all_eligible_images"},
                inputs={"queries": input_path},
                result_files={"metrics": result_path},
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["method"], "synthetic")
            self.assertEqual(manifest["inputs"]["queries"]["sha256"], sha256_file(input_path))
            self.assertEqual(manifest["results"]["metrics"]["sha256"], sha256_file(result_path))


if __name__ == "__main__":
    unittest.main()
