"""Timestamp parsing separated from model inference."""

import math
import re

_TIMESTAMP = r"\d+:\d+:\d+(?:\.\d*)?|\d+:\d+(?:\.\d*)?|\d+(?:\.\d*)?"


def timestamp_to_seconds(value):
    parts = str(value).strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid timestamp: {value}")
    result = 0.0
    for part in parts:
        result = result * 60 + float(part)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid timestamp: {value}")
    return result


def parse_response(response):
    if not isinstance(response, str):
        return None, None
    field = r'''(?<!\w)["']?{name}["']?\s*:\s*["']?\s*({timestamp})(?![\d:.])'''
    start = re.search(field.format(name="(?:start_time|start)", timestamp=_TIMESTAMP), response, re.I)
    end = re.search(field.format(name="(?:end_time|end)", timestamp=_TIMESTAMP), response, re.I)
    if start and end:
        return timestamp_to_seconds(start[1]), timestamp_to_seconds(end[1])
    match = re.search(rf"(?<![\d:.\-])({_TIMESTAMP})(?![\d:.]).*?(?:\bto\b|\band\b|[-–—]).*?(?<![\d:.\-])({_TIMESTAMP})(?![\d:.])", response, re.I | re.S)
    return (timestamp_to_seconds(match[1]), timestamp_to_seconds(match[2])) if match else (None, None)


def valid_interval(start, end):
    return (not isinstance(start, bool) and not isinstance(end, bool)
            and isinstance(start, (float, int)) and isinstance(end, (float, int))
            and math.isfinite(start) and math.isfinite(end) and start >= 0 and end > start)
