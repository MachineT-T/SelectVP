"""Canonical query-only inputs and separately loaded evaluation truth."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from selectvp.utils.io import id_key, index_by_id, read_json, write_json
from selectvp.utils.parsing import valid_interval


def safe_video_path(directory, video):
    if not isinstance(video, str) or not video or Path(video).is_absolute():
        raise ValueError(f"Video path must be a nonempty relative string: {video!r}")
    root = Path(directory).resolve()
    path = (root / video).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Video path must stay within its configured directory: {video!r}")
    return path


def normalize_annotations(source, dataset, durations=None):
    """Accept original release formats or the legacy converted query lists."""
    source = Path(source)
    if source.suffix == ".txt":
        if dataset != "charades_sta" or not durations:
            raise ValueError("Charades TXT input requires --durations with Charades CSV or a JSON duration map")
        if Path(durations).suffix == ".csv":
            with Path(durations).open(encoding="utf-8") as stream:
                duration_map = {row["id"]: float(row["length"]) for row in csv.DictReader(stream)}
        else:
            duration_map = read_json(durations)
        rows, counts = [], {}
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            label, query = line.split("##", 1)
            video, start, end = label.split()
            counts[video] = counts.get(video, 0) + 1
            rows.append({"id": f"{video}_{counts[video]}", "video": f"{video}.mp4", "query": query.strip(),
                         "duration": duration_map[video], "start_time": float(start), "end_time": float(end)})
    else:
        rows = read_json(source)
    if isinstance(rows, dict):
        if dataset != "activitynet":
            raise ValueError("Dictionary annotations are supported for ActivityNet only")
        converted = []
        for vid, row in rows.items():
            if len(row["sentences"]) != len(row["timestamps"]):
                raise ValueError(f"Mismatched sentence/timestamp lengths: {vid}")
            for i, (query, times) in enumerate(zip(row["sentences"], row["timestamps"]), 1):
                converted.append({"id": f"{vid}_{i}", "video": f"{vid}.mp4", "query": query,
                                  "duration": row["duration"], "start_time": times[0], "end_time": times[1]})
        rows = converted
    samples, truth = [], []
    for row in rows:
        qv = "qid" in row
        sample = {"id": row["qid"] if qv else row["id"],
                  "video": f"{row['vid']}.mp4" if qv else row["video"],
                  "query": row["query"], "duration": float(row["duration"])}
        samples.append(sample)
        windows = row.get("relevant_windows")
        if windows is None and row.get("start_time") is not None and row.get("end_time") is not None:
            windows = [[row["start_time"], row["end_time"]]]
        if windows is not None:
            truth.append({"id": sample["id"], "relevant_windows": windows})
    validate_samples(samples)
    validate_truth(truth)
    return samples, truth


def validate_samples(samples, video_dir=None):
    index_by_id(samples)
    if not samples:
        raise ValueError("Empty sample list")
    missing = []
    for sample in samples:
        if not isinstance(sample.get("query"), str) or not sample["query"].strip():
            raise ValueError(f"Missing query: {sample['id']}")
        duration = sample.get("duration")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid duration: {sample['id']}")
        path = safe_video_path(video_dir or ".", sample["video"])
        if video_dir and not path.is_file():
            missing.append(sample["video"])
    if missing:
        raise ValueError(f"Missing {len(missing)} videos; first: {missing[:5]}")


def validate_truth(truth):
    index_by_id(truth)
    for row in truth:
        if not row.get("relevant_windows"):
            raise ValueError(f"Missing GT windows: {row['id']}")
        for window in row["relevant_windows"]:
            if len(window) != 2 or not valid_interval(*window):
                raise ValueError(f"Invalid GT window: {row['id']}: {window}")


def filter_samples(samples, ids_file=None, full=False):
    if full:
        return samples
    if not ids_file:
        raise ValueError("A sample ID file is required; use --full explicitly for the whole split")
    ids = read_json(ids_file)
    if not isinstance(ids, list) or not ids:
        raise ValueError("Sample ID file must be a nonempty JSON list")
    requested = index_by_id([{"id": value} for value in ids])
    index = index_by_id(samples)
    missing = set(requested) - set(index)
    if missing:
        raise ValueError(f"Unknown requested IDs: {sorted(missing)[:5]}")
    return [index[key] for key in requested]


def load_samples(config, full=False, ids_file=None, check_videos=True):
    # Strip any extraneous fields to ensure GT never reaches the model.
    rows = read_json(config["data"]["annotations"])
    validate_samples(rows)
    rows = filter_samples(rows, ids_file or config["data"].get("sample_ids"), full)
    samples = [{key: row[key] for key in ("id", "video", "query", "duration")} for row in rows]
    validate_samples(samples, config["data"]["video_dir"] if check_videos else None)
    return samples


def load_truth(config, samples):
    rows = read_json(config["data"]["ground_truth"])
    validate_truth(rows)
    index = index_by_id(rows)
    try:
        return [index[id_key(sample["id"])] for sample in samples]
    except KeyError as exc:
        raise ValueError(f"Missing ground truth ID: {exc}") from exc
