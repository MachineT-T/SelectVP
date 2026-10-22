"""Video operations with checked decoding and frame-exact legacy cropping."""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or min(frames, width, height) <= 0:
            raise ValueError(f"Invalid video metadata: {path}")
        return {"fps": fps, "frames": frames, "width": width, "height": height, "duration": frames / fps}
    finally:
        cap.release()


@contextmanager
def video_writer(path, fps, width, height):
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {path}")
        yield writer
    finally:
        writer.release()


def iter_frames(path, indices=None):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        if indices is None:
            for idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Video decode failed at frame {idx}: {path}")
                yield idx, frame
        else:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Video decode failed at frame {idx}: {path}")
                yield int(idx), frame
    finally:
        cap.release()


def cut_video_segment(source, destination, start, end):
    from selectvp.utils.parsing import valid_interval
    if not valid_interval(start, end):
        return False
    info = video_info(source)
    start, end = min(start, info["duration"]), min(end, info["duration"])
    if end <= start:
        return False
    first = max(0, math.floor(start * info["fps"]))
    last = min(info["frames"], math.ceil(end * info["fps"]))
    if first >= last:
        return False
    with video_writer(destination, info["fps"], info["width"], info["height"]) as writer:
        for _, frame in iter_frames(source, range(first, last)):
            writer.write(frame)
    return video_info(destination)["frames"] == last - first
