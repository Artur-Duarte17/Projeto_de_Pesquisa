from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat


ATTR_NAMES = ("blur", "expression", "illumination", "invalid", "occlusion", "pose")


@dataclass(frozen=True)
class GroundTruth:
    boxes: np.ndarray
    attributes: np.ndarray


def parse_wider_annotations(path: Path) -> dict[str, GroundTruth]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, GroundTruth] = {}
    cursor = 0
    while cursor < len(lines):
        image_key = lines[cursor].strip()
        cursor += 1
        if not image_key:
            continue
        if cursor >= len(lines):
            raise ValueError(f"Missing face count after {image_key}")
        count = int(lines[cursor].strip())
        cursor += 1
        rows: list[list[int]] = []
        for _ in range(count):
            if cursor >= len(lines):
                raise ValueError(f"Truncated annotations for {image_key}")
            values = [int(value) for value in lines[cursor].split()]
            cursor += 1
            if len(values) != 10:
                raise ValueError(f"Expected 10 values for {image_key}; got {len(values)}")
            rows.append(values)
        array = np.asarray(rows, dtype=np.float64).reshape((-1, 10))
        boxes = array[:, :4].copy()
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        result[image_key] = GroundTruth(boxes=boxes, attributes=array[:, 4:].astype(np.int8))
    return result


def _mat_string(value: object) -> str:
    current = value
    while isinstance(current, np.ndarray):
        if current.size != 1:
            break
        current = current.reshape(-1)[0]
    return str(current)


def load_difficulty_indices(path: Path) -> dict[str, set[int]]:
    data = loadmat(path)
    events = data["event_list"].reshape(-1)
    file_groups = data["file_list"].reshape(-1)
    gt_groups = data["gt_list"].reshape(-1)
    result: dict[str, set[int]] = {}
    for event_raw, files_raw, keep_raw in zip(events, file_groups, gt_groups, strict=True):
        event = _mat_string(event_raw)
        files = np.asarray(files_raw).reshape(-1)
        keeps = np.asarray(keep_raw).reshape(-1)
        if len(files) != len(keeps):
            raise ValueError(f"Difficulty structure mismatch for event {event}")
        for file_raw, indices_raw in zip(files, keeps, strict=True):
            stem = _mat_string(file_raw)
            indices = np.asarray(indices_raw).reshape(-1).astype(np.int64)
            result[f"{event}/{stem}.jpg"] = {int(index) - 1 for index in indices}
    return result


def box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    a = np.asarray(boxes1, dtype=np.float64).reshape((-1, 4))
    b = np.asarray(boxes2, dtype=np.float64).reshape((-1, 4))
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(right_bottom - left_top + 1.0, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0] + 1.0, 0.0, None) * np.clip(
        a[:, 3] - a[:, 1] + 1.0, 0.0, None
    )
    area_b = np.clip(b[:, 2] - b[:, 0] + 1.0, 0.0, None) * np.clip(
        b[:, 3] - b[:, 1] + 1.0, 0.0, None
    )
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def valid_boxes(boxes: np.ndarray) -> np.ndarray:
    array = np.asarray(boxes)
    return (
        np.isfinite(array).all(axis=1)
        & (array[:, 2] >= array[:, 0])
        & (array[:, 3] >= array[:, 1])
    )


def match_image(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    keep_indices: set[int],
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(pred_boxes, dtype=np.float64).reshape((-1, 4))
    scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("Prediction box/score count mismatch")
    if not valid_boxes(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Predictions contain invalid or non-finite values")
    order = np.argsort(-scores, kind="stable")
    boxes = boxes[order]
    scores = scores[order]
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape((-1, 4))
    matched = np.zeros(len(gt), dtype=bool)
    status = np.zeros(len(boxes), dtype=np.int8)  # 1 TP, 0 FP, -1 ignored
    matched_gt = np.full(len(boxes), -1, dtype=np.int64)
    overlaps = box_iou_numpy(boxes, gt)
    for index in range(len(boxes)):
        if len(gt) == 0:
            continue
        gt_index = int(np.argmax(overlaps[index]))
        if overlaps[index, gt_index] < iou_threshold:
            continue
        matched_gt[index] = gt_index
        if gt_index not in keep_indices:
            status[index] = -1
        elif not matched[gt_index]:
            status[index] = 1
            matched[gt_index] = True
    return scores, status, matched_gt, order


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changes = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    return float(np.sum((mrec[changes] - mrec[changes - 1]) * mpre[changes]))


def evaluate_subset(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    ground_truth: dict[str, GroundTruth],
    keep_by_image: dict[str, set[int]],
    *,
    iou_threshold: float = 0.5,
    score_threshold: float | None = None,
) -> dict[str, object]:
    records: list[tuple[float, int, str, int]] = []
    positives = 0
    matched_gt: dict[str, set[int]] = {}
    for image_key in sorted(ground_truth):
        gt = ground_truth[image_key]
        keep = keep_by_image.get(image_key, set())
        positives += len(keep)
        boxes, scores = predictions.get(
            image_key, (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
        )
        scores_sorted, status, gt_indices, _ = match_image(
            boxes, scores, gt.boxes, keep, iou_threshold
        )
        hits: set[int] = set()
        for score, state, gt_index in zip(scores_sorted, status, gt_indices, strict=True):
            if score_threshold is not None and score < score_threshold:
                continue
            if state != -1:
                records.append((float(score), int(state), image_key, int(gt_index)))
            if state == 1:
                hits.add(int(gt_index))
        matched_gt[image_key] = hits
    records.sort(key=lambda item: -item[0])
    states = np.asarray([item[1] for item in records], dtype=np.int64)
    tp = np.cumsum(states == 1)
    fp = np.cumsum(states == 0)
    recall = tp / positives if positives else np.zeros(len(states), dtype=np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
    final_tp = int(tp[-1]) if len(tp) else 0
    final_fp = int(fp[-1]) if len(fp) else 0
    final_recall = final_tp / positives if positives else 0.0
    final_precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) else 0.0
    f1 = (
        2.0 * final_precision * final_recall / (final_precision + final_recall)
        if (final_precision + final_recall)
        else 0.0
    )
    return {
        "ap": voc_ap(recall, precision),
        "precision": final_precision,
        "recall": final_recall,
        "f1": f1,
        "true_positives": final_tp,
        "false_positives": final_fp,
        "ground_truth": positives,
        "matched_gt": matched_gt,
        "curve_recall": recall,
        "curve_precision": precision,
    }


def evaluate_wider_official(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    ground_truth: dict[str, GroundTruth],
    keep_by_image: dict[str, set[int]],
    *,
    iou_threshold: float = 0.5,
    threshold_count: int = 1000,
) -> dict[str, object]:
    """Reproduce the official WIDER 1000-threshold validation protocol."""
    all_scores = [np.asarray(scores, dtype=np.float64) for _, scores in predictions.values() if len(scores)]
    if all_scores:
        score_vector = np.concatenate(all_scores)
        minimum = float(np.min(score_vector))
        maximum = float(np.max(score_vector))
    else:
        minimum = maximum = 0.0
    proposals = np.zeros(threshold_count, dtype=np.int64)
    recalled = np.zeros(threshold_count, dtype=np.int64)
    positives = 0
    for image_key in sorted(ground_truth):
        keep = keep_by_image.get(image_key, set())
        positives += len(keep)
        boxes, scores = predictions.get(
            image_key, (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
        )
        sorted_scores, status, _gt_indices, _order = match_image(
            boxes, scores, ground_truth[image_key].boxes, keep, iou_threshold
        )
        if not len(sorted_scores):
            continue
        if maximum > minimum:
            normalized = (sorted_scores - minimum) / (maximum - minimum)
        else:
            normalized = np.ones_like(sorted_scores)
        cumulative_recall = np.cumsum(status == 1)
        cumulative_proposals = np.cumsum(status != -1)
        for index in range(threshold_count):
            threshold = 1.0 - (index + 1) / threshold_count
            selected = np.flatnonzero(normalized >= threshold)
            if len(selected):
                last = int(selected[-1])
                proposals[index] += int(cumulative_proposals[last])
                recalled[index] += int(cumulative_recall[last])
    precision = np.divide(
        recalled,
        proposals,
        out=np.zeros(threshold_count, dtype=np.float64),
        where=proposals > 0,
    )
    recall = recalled / positives if positives else np.zeros(threshold_count, dtype=np.float64)
    return {
        "ap": voc_ap(recall, precision),
        "ground_truth": positives,
        "score_min": minimum,
        "score_max": maximum,
        "threshold_count": threshold_count,
        "curve_recall": recall,
        "curve_precision": precision,
    }


def recall_by_categories(
    ground_truth: dict[str, GroundTruth], matched_gt: dict[str, set[int]]
) -> list[dict[str, object]]:
    counters: dict[tuple[str, str], list[int]] = {}

    def add(group: str, value: str, hit: bool) -> None:
        total, matched = counters.setdefault((group, value), [0, 0])
        counters[(group, value)] = [total + 1, matched + int(hit)]

    labels = {
        "blur": {0: "clear", 1: "normal", 2: "heavy"},
        "illumination": {0: "normal", 1: "extreme"},
        "occlusion": {0: "none", 1: "partial", 2: "heavy"},
        "pose": {0: "typical", 1: "atypical"},
    }
    for image_key, gt in ground_truth.items():
        hits = matched_gt.get(image_key, set())
        for index, (box, attrs) in enumerate(zip(gt.boxes, gt.attributes, strict=True)):
            blur, _expression, illumination, invalid, occlusion, pose = [int(x) for x in attrs]
            if invalid:
                continue
            height = float(box[3] - box[1])
            if 10 <= height < 50:
                size = "small_10_50"
            elif 50 <= height <= 300:
                size = "medium_50_300"
            elif height > 300:
                size = "large_over_300"
            else:
                size = "below_10"
            hit = index in hits
            add("size", size, hit)
            for group, value in (
                ("blur", blur),
                ("illumination", illumination),
                ("occlusion", occlusion),
                ("pose", pose),
            ):
                add(group, labels[group].get(value, f"unknown_{value}"), hit)
    return [
        {
            "category": group,
            "value": value,
            "ground_truth": total,
            "matched": matched,
            "recall": matched / total if total else 0.0,
        }
        for (group, value), (total, matched) in sorted(counters.items())
    ]


def predictions_from_flat_arrays(
    image_keys: Iterable[str], offsets: np.ndarray, boxes: np.ndarray, scores: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    keys = list(image_keys)
    offsets = np.asarray(offsets, dtype=np.int64)
    if len(offsets) != len(keys) + 1 or offsets[0] != 0 or offsets[-1] != len(boxes):
        raise ValueError("Invalid flattened prediction offsets")
    return {
        key: (boxes[offsets[index] : offsets[index + 1]], scores[offsets[index] : offsets[index + 1]])
        for index, key in enumerate(keys)
    }
