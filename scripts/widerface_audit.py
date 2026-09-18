from __future__ import annotations

import hashlib

import numpy as np
import torch
from torchvision.ops import box_iou

from widerface_eval_lib import box_iou_numpy, match_image


def fixed_sample(keys: list[str], count: int, seed: str = "EX-031") -> list[str]:
    return sorted(keys, key=lambda key: hashlib.sha256(f"{seed}\0{key}".encode()).hexdigest())[:count]


def torchvision_iou_inclusive(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    first = np.asarray(boxes1, dtype=np.float64).copy().reshape((-1, 4))
    second = np.asarray(boxes2, dtype=np.float64).copy().reshape((-1, 4))
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float64)
    first[:, 2:] += 1.0
    second[:, 2:] += 1.0
    return box_iou(torch.from_numpy(first), torch.from_numpy(second)).numpy()


def audit_sample(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    gt_boxes: dict[str, np.ndarray],
    keep_by_image: dict[str, set[int]],
    sample_keys: list[str],
    iou_threshold: float,
) -> dict[str, object]:
    maximum_difference = 0.0
    associations_checked = 0
    for key in sample_keys:
        boxes, scores = predictions[key]
        gt = gt_boxes[key]
        numpy_iou = box_iou_numpy(boxes, gt)
        torch_iou = torchvision_iou_inclusive(boxes, gt)
        if numpy_iou.size:
            maximum_difference = max(maximum_difference, float(np.max(np.abs(numpy_iou - torch_iou))))
        if not np.allclose(numpy_iou, torch_iou, rtol=1e-6, atol=1e-6):
            raise AssertionError(f"NumPy/torchvision IoU mismatch for {key}")
        sorted_scores, status, matched_gt, order = match_image(
            boxes, scores, gt, keep_by_image[key], iou_threshold
        )
        tv_sorted = torch_iou[order]
        seen: set[int] = set()
        expected_status = np.zeros(len(order), dtype=np.int8)
        expected_gt = np.full(len(order), -1, dtype=np.int64)
        for index in range(len(order)):
            if len(gt) == 0:
                continue
            gt_index = int(np.argmax(tv_sorted[index]))
            if tv_sorted[index, gt_index] < iou_threshold:
                continue
            expected_gt[index] = gt_index
            if gt_index not in keep_by_image[key]:
                expected_status[index] = -1
            elif gt_index not in seen:
                expected_status[index] = 1
                seen.add(gt_index)
        if not np.array_equal(status, expected_status) or not np.array_equal(matched_gt, expected_gt):
            raise AssertionError(f"Independent association mismatch for {key}")
        if not np.array_equal(sorted_scores, scores[order]):
            raise AssertionError(f"Score ordering mismatch for {key}")
        associations_checked += len(order)
    return {
        "sample_images": sample_keys,
        "images_checked": len(sample_keys),
        "associations_checked": associations_checked,
        "maximum_iou_difference": maximum_difference,
        "tolerance": 1e-6,
        "passed": True,
    }
