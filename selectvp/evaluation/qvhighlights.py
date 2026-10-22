"""Historical metrics_hd.py semantics, using every relevant window."""

from selectvp.utils.parsing import valid_interval


def multi_window_iou(start, end, windows):
    if not valid_interval(start, end) or not windows:
        return 0.0
    intersection = sum(max(0.0, min(end, b) - max(start, a)) for a, b in windows)
    union = end - start + sum(b - a for a, b in windows) - intersection
    return intersection / union if union > 0 else 0.0
