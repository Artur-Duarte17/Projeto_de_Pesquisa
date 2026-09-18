from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from widerface_audit import audit_sample
from widerface_eval_lib import GroundTruth, evaluate_subset, evaluate_wider_official, match_image


class WiderFaceEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gt = {
            "event/image.jpg": GroundTruth(
                boxes=np.asarray([[0, 0, 9, 9]], dtype=np.float64),
                attributes=np.zeros((1, 6), dtype=np.int8),
            )
        }

    def test_perfect_result(self) -> None:
        predictions = {"event/image.jpg": (np.asarray([[0, 0, 9, 9]]), np.asarray([0.9]))}
        result = evaluate_subset(predictions, self.gt, {"event/image.jpg": {0}})
        self.assertEqual(result["ap"], 1.0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        official = evaluate_wider_official(predictions, self.gt, {"event/image.jpg": {0}})
        self.assertEqual(official["ap"], 1.0)

    def test_empty_result(self) -> None:
        result = evaluate_subset({}, self.gt, {"event/image.jpg": {0}})
        self.assertEqual(result["ap"], 0.0)
        self.assertEqual(result["recall"], 0.0)

    def test_duplicate_is_false_positive(self) -> None:
        boxes = np.asarray([[0, 0, 9, 9], [0, 0, 9, 9]])
        scores = np.asarray([0.9, 0.8])
        _, status, _, _ = match_image(boxes, scores, self.gt["event/image.jpg"].boxes, {0})
        np.testing.assert_array_equal(status, [1, 0])

    def test_invalid_prediction_box_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            match_image(np.asarray([[5, 5, 4, 4]]), np.asarray([0.9]), self.gt["event/image.jpg"].boxes, {0})

    def test_ignored_ground_truth_does_not_count_as_false_positive(self) -> None:
        predictions = {"event/image.jpg": (np.asarray([[0, 0, 9, 9]]), np.asarray([0.9]))}
        result = evaluate_subset(predictions, self.gt, {"event/image.jpg": set()})
        self.assertEqual(result["false_positives"], 0)
        self.assertEqual(result["true_positives"], 0)

    def test_torchvision_audit_repeats_deterministically(self) -> None:
        predictions = {"event/image.jpg": (np.asarray([[0, 0, 9, 9]]), np.asarray([0.9]))}
        kwargs = dict(
            predictions=predictions,
            gt_boxes={"event/image.jpg": self.gt["event/image.jpg"].boxes},
            keep_by_image={"event/image.jpg": {0}},
            sample_keys=["event/image.jpg"],
            iou_threshold=0.5,
        )
        self.assertEqual(audit_sample(**kwargs), audit_sample(**kwargs))


if __name__ == "__main__":
    unittest.main()
