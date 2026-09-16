from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.data.prepare_agrishow_face_query import bbox_iou, expanded_square_bbox


class AgrishowQueryGeometryTests(unittest.TestCase):
    def test_expanded_crop_is_square_and_centered(self) -> None:
        crop = expanded_square_bbox((40, 50, 60, 70), 100, 100, 2.0)
        self.assertEqual(crop, (30, 40, 70, 80))

    def test_expanded_crop_stays_inside_image(self) -> None:
        crop = expanded_square_bbox((0, 0, 20, 20), 100, 100, 2.0)
        self.assertEqual(crop, (0, 0, 40, 40))

    def test_bbox_iou_matches_identical_and_disjoint_boxes(self) -> None:
        self.assertEqual(bbox_iou((1, 2, 11, 12), (1, 2, 11, 12)), 1.0)
        self.assertEqual(bbox_iou((0, 0, 10, 10), (10, 10, 20, 20)), 0.0)


if __name__ == "__main__":
    unittest.main()
