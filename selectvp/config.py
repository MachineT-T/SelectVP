"""YAML configuration, explicit overrides, and portable project paths."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import yaml


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, overrides=()):
    path = Path(path).resolve()

    def read(current, stack):
        if current in stack:
            raise ValueError(f"Cyclic config inheritance: {current}")
        with current.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, dict):
            raise ValueError(f"Configuration must be a mapping: {current}")
        parent = value.pop("extends", None)
        return merge(read((current.parent / parent).resolve(), stack + [current]), value) if parent else value

    config = read(path, [])
    for item in overrides:
        if "=" not in item:
            raise ValueError("--set expects dotted.key=value")
        key, raw = item.split("=", 1)
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"Unknown config key: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"Unknown config key: {key}")
        node[parts[-1]] = yaml.safe_load(raw)
    # All relative paths are rooted beside configs/, independent of process cwd.
    root = (path.parent / config.pop("project_root", "..")).resolve()
    for section, keys in {"data": ("source", "annotations", "ground_truth", "raw_dir", "video_dir", "sample_ids"),
                          "paths": ("cache", "output")}.items():
        for key in keys:
            value = config.get(section, {}).get(key)
            if value:
                config[section][key] = str((root / value).resolve())
    if config["model"]["id"] != "Qwen/Qwen3-VL-8B-Instruct" and not Path(config["model"]["id"]).is_dir():
        raise ValueError("Only Qwen3-VL-8B-Instruct or its local directory is supported")
    fps = config["video"]["fps"]
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps != 1:
        raise ValueError("The frozen ES frame protocol requires 1 FPS")
    if config["branches"] != ["bl", "es", "od", "km"]:
        raise ValueError("Frozen branch order is bl, es, od, km")
    if config["model"]["do_sample"] is not False:
        raise ValueError("Frozen inference requires do_sample: false")
    if config["video"]["sampling"] != "opencv_integer_fps":
        raise ValueError("Unsupported video sampling protocol")
    return config


def config_identity(config):
    """Execution identity excludes only output location and sample filtering."""
    config = copy.deepcopy(config)
    config["paths"].pop("output", None)
    config["data"].pop("sample_ids", None)
    return config
