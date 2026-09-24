from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from face_lib import pick_query_face


def load_script_module(module_name: str, relative_path: str):
    script_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load test target: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GALLAGHER_PREPARE = load_script_module(
    "gallagher_prepare_for_tests",
    "scripts/face/04_prepare_gallagher_eval.py",
)


class GallagherGroundTruthTests(unittest.TestCase):
    def test_relevant_photo_survives_missing_face_detection(self) -> None:
        annotations = pd.DataFrame(
            [
                {"image_name": "source.JPG", "identity": "2"},
                {"image_name": "detected.JPG", "identity": "2"},
                {"image_name": "missed_by_detector.JPG", "identity": "2"},
            ]
        )
        gallery_metadata = pd.DataFrame(
            [
                {"image_id": "source", "image_path": "gallery/source.JPG"},
                {"image_id": "detected", "image_path": "gallery/detected.JPG"},
                {
                    "image_id": "missed",
                    "image_path": "gallery/missed_by_detector.JPG",
                },
            ]
        )
        detected_face_image_ids = {"source", "detected"}

        mapped = GALLAGHER_PREPARE.map_annotations_to_gallery(
            annotations,
            gallery_metadata,
        )
        image_ids_by_identity = GALLAGHER_PREPARE.build_image_ids_by_identity(mapped)
        relevant = GALLAGHER_PREPARE.relevant_image_ids_for_query(
            image_ids_by_identity,
            identity="2",
            source_image_id="source",
        )

        self.assertNotIn("missed", detected_face_image_ids)
        self.assertEqual(relevant, ["detected", "missed"])


class DummyFace:
    def __init__(self, bbox: list[float]) -> None:
        self.bbox = np.asarray(bbox, dtype=np.float32)


class GallagherTargetFaceTests(unittest.TestCase):
    def test_index_mode_uses_deterministic_visual_order(self) -> None:
        lower = DummyFace([10.0, 100.0, 50.0, 140.0])
        upper_right = DummyFace([100.0, 10.0, 140.0, 50.0])
        upper_left = DummyFace([10.0, 10.0, 50.0, 50.0])

        faces = [lower, upper_right, upper_left]

        self.assertIs(pick_query_face(faces, mode="index", index=0), upper_left)
        self.assertIs(pick_query_face(faces, mode="index", index=1), upper_right)
        self.assertIs(pick_query_face(faces, mode="index", index=2), lower)

    def test_larger_neighbor_does_not_replace_annotated_target(self) -> None:
        larger_neighbor = DummyFace([0.0, 0.0, 200.0, 200.0])
        annotated_target = DummyFace([220.0, 220.0, 270.0, 270.0])
        faces = [larger_neighbor, annotated_target]

        self.assertIs(pick_query_face(faces, mode="largest"), larger_neighbor)
        selected = pick_query_face(
            faces,
            mode="annotated_eye_midpoint",
            target_x=245.0,
            target_y=245.0,
        )

        self.assertIs(selected, annotated_target)

    def test_missing_annotated_target_does_not_fall_back_to_neighbor(self) -> None:
        neighbor = DummyFace([0.0, 0.0, 200.0, 200.0])

        selected = pick_query_face(
            [neighbor],
            mode="annotated_eye_midpoint",
            target_x=245.0,
            target_y=245.0,
        )

        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
