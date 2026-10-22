"""Recall@1 and mIoU with explicit coverage and failure accounting."""

from selectvp.evaluation.qvhighlights import multi_window_iou
from selectvp.utils.io import index_by_id
from selectvp.utils.parsing import valid_interval


def temporal_iou(start, end, gt_start, gt_end):
    if not valid_interval(start, end):
        return 0.0
    intersection = max(0.0, min(end, gt_end) - max(start, gt_start))
    union = max(end, gt_end) - min(start, gt_start)
    return intersection / union if union > 0 else 0.0


def summarize(ious):
    if not ious:
        raise ValueError("Cannot evaluate zero samples")
    return {"num_samples": len(ious), **{f"R1@{threshold}": round(sum(x >= threshold for x in ious) / len(ious) * 100, 2) for threshold in (0.3, 0.5, 0.7)},
            "mIoU": round(sum(ious) / len(ious) * 100, 2)}


def evaluate(predictions, ground_truth, protocol="single", allow_missing=False):
    if protocol not in ("single", "qvhighlights"):
        raise ValueError(f"Unknown evaluation protocol: {protocol}")
    gt_index = index_by_id(ground_truth)
    pred_index = index_by_id(predictions)
    if set(pred_index) - set(gt_index):
        raise ValueError("Predictions contain unknown IDs")
    missing = set(gt_index) - set(pred_index)
    if missing and not allow_missing:
        raise ValueError(f"Missing {len(missing)} predictions; use --allow-missing to count them as zero")
    ious = []
    failed = 0
    for key, truth in gt_index.items():
        pred = pred_index.get(key, {})
        start, end = pred.get("pred_start"), pred.get("pred_end")
        if not valid_interval(start, end) or pred.get("status", "ok") != "ok":
            failed += 1
        windows = truth["relevant_windows"]
        if protocol == "qvhighlights":
            iou = multi_window_iou(start, end, windows)
        else:
            if len(windows) != 1:
                raise ValueError("Single-window evaluation needs exactly one GT window")
            iou = temporal_iou(start, end, *windows[0])
        ious.append(iou)
    return {**summarize(ious), "missing_predictions": len(missing), "failed_predictions": failed}
