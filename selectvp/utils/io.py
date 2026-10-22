"""Atomic artifacts and reproducibility metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as stream:
            name = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def artifact_key(value):
    """Filesystem-safe key; typed IDs remain distinct in alignment."""
    return fingerprint(value)[:24]


def id_key(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"ID must be an integer or string: {value!r}")
    return json.dumps(value, ensure_ascii=False)


def index_by_id(rows):
    result = {}
    for row in rows:
        key = id_key(row["id"])
        if key in result:
            raise ValueError(f"Duplicate ID: {row['id']!r}")
        result[key] = row
    return result


def environment():
    packages = {}
    for name in ("selectvp", "torch", "torchvision", "transformers", "accelerate", "flash-attn", "numpy", "opencv-python", "salesforce-lavis", "scikit-learn", "scipy", "qwen-vl-utils", "av", "decord", "Pillow", "PyYAML", "timm", "safetensors", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"recorded_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "packages": packages}


def write_run_manifest(directory, config, samples, stage):
    directory = Path(directory)
    write_json(directory / "config.json", config)
    snapshot = environment()
    initial = directory / f"{stage}_environment.json"
    if not initial.exists():
        write_json(initial, snapshot)
    stamp = snapshot["recorded_at"].replace(":", "")
    write_json(directory / "invocations" / f"{stage}_{stamp}.json", snapshot)
    write_json(directory / "samples.json", [s["id"] for s in samples])
