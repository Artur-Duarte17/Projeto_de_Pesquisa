from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "face" / "06_prepare_lfw_eval.py"
SPEC = importlib.util.spec_from_file_location("prepare_lfw_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE_LFW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE_LFW)


class LfwProtocolTests(unittest.TestCase):
    def test_protocol_is_deterministic_and_excludes_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = []
            for name in ("a1.jpg", "a2.jpg", "a3.jpg", "b1.jpg"):
                path = root / name
                path.write_bytes(b"test")
                paths.append(path)

            metadata = pd.DataFrame(
                [
                    {"image_id": "a1", "image_path": str(paths[0]), "sha256": "1" * 64, "identity": "A"},
                    {"image_id": "a1", "image_path": str(paths[0]), "sha256": "1" * 64, "identity": "A"},
                    {"image_id": "a2", "image_path": str(paths[1]), "sha256": "2" * 64, "identity": "A"},
                    {"image_id": "a3", "image_path": str(paths[2]), "sha256": "3" * 64, "identity": "A"},
                    {"image_id": "b1", "image_path": str(paths[3]), "sha256": "4" * 64, "identity": "B"},
                ]
            )

            images = PREPARE_LFW.unique_indexed_images(metadata)
            first_queries, first_relevance = PREPARE_LFW.build_protocol(
                images, seed=20260915, min_images=2
            )
            second_queries, second_relevance = PREPARE_LFW.build_protocol(
                images, seed=20260915, min_images=2
            )

            pd.testing.assert_frame_equal(first_queries, second_queries)
            pd.testing.assert_frame_equal(first_relevance, second_relevance)
            self.assertEqual(len(images), 4)
            self.assertEqual(first_queries["target_label"].tolist(), ["A"])
            self.assertEqual(len(first_relevance), 2)
            source_id = str(first_queries.iloc[0]["source_image_id"])
            self.assertNotIn(source_id, set(first_relevance["image_id"].astype(str)))

    def test_minimum_images_cannot_be_less_than_two(self) -> None:
        with self.assertRaises(ValueError):
            PREPARE_LFW.build_protocol(pd.DataFrame(), seed=1, min_images=1)


if __name__ == "__main__":
    unittest.main()
