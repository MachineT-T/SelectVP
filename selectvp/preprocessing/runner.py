"""Stage orchestration with content-addressed inputs and resumable artifacts."""

from __future__ import annotations

import gc
from pathlib import Path

from selectvp.data import safe_video_path
from selectvp.utils.io import artifact_key, fingerprint, id_key, read_json, sha256, write_json, write_run_manifest

STAGES = ("videos", "blip2", "events", "targets", "detection", "keyframes", "prompts")
CACHE_VERSION = 1


def cached(path, identity, compute=None):
    path = Path(path)
    signature = fingerprint({"version": CACHE_VERSION, "identity": identity})
    if path.is_file():
        record = read_json(path)
        if record.get("signature") == signature and record.get("payload_sha256") == fingerprint(record.get("data")):
            return record["data"]
    if compute is None:
        raise ValueError(f"Missing or stale cache: {path}; run its preprocessing stage")
    data = compute()
    write_json(path, {"signature": signature, "identity": identity, "payload_sha256": fingerprint(data), "data": data})
    return data


class Preprocessor:
    def __init__(self, config, samples, universe):
        self.config, self.samples = config, samples
        self.root = Path(config["paths"]["cache"])
        self.root.mkdir(parents=True, exist_ok=True)
        selected_videos = {s["video"] for s in samples}
        self.context = [s for s in universe if s["video"] in selected_videos]
        self.videos = {s["video"]: self.video(s) for s in samples}
        self.hashes = {}

    def video(self, sample):
        return safe_video_path(self.config["data"]["video_dir"], sample["video"])

    def input_identity(self, sample):
        video = sample["video"]
        if video not in self.hashes:
            self.hashes[video] = sha256(self.video(sample))
        return {"sample": sample, "video_sha256": self.hashes[video]}

    def path(self, stage, sample, per_video=False):
        key = sample["video"] if per_video else sample["id"]
        return self.root / stage / f"{artifact_key(key)}.json"

    def features(self, sample, kind):
        import numpy as np
        key = sample["video"] if kind == "visual" else sample["id"]
        path = self.root / "blip2_features" / f"{kind}_{artifact_key(key)}.npy"
        identity = {"blip2": self.config["preprocessing"]["blip2"], "input": self.input_identity(sample) if kind == "text" else
                    {"video": sample["video"], "video_sha256": self.input_identity(sample)["video_sha256"]}}
        metadata = cached(path.with_suffix(".json"), identity)
        if not path.is_file() or sha256(path) != metadata["sha256"]:
            raise ValueError(f"Missing/corrupt BLIP-2 {kind} array: {path}")
        return np.load(path, allow_pickle=False)

    def events(self, sample, compute=None):
        identity = {"input": self.input_identity(sample), "blip2": self.config["preprocessing"]["blip2"],
                    "es": self.config["preprocessing"]["es"]}
        return cached(self.path("events", sample), identity, compute)

    def targets(self, sample, compute=None):
        identity = {"input": self.input_identity(sample), "targets": self.config["preprocessing"]["targets"]}
        return cached(self.path("object_targets", sample), identity, compute)

    def keyframes(self, sample, compute=None):
        identity = {"input": self.input_identity(sample), "keyframes": self.config["preprocessing"]["keyframes"]}
        return cached(self.path("keyframes", sample), identity, compute)

    def detections(self, sample, compute=None):
        targets = self.video_targets(sample["video"])
        identity = {"video": sample["video"], "video_sha256": self.input_identity(sample)["video_sha256"],
                    "targets": targets, "detection": self.config["preprocessing"]["detection"]}
        return cached(self.path("detections", sample, per_video=True), identity,
                      (lambda: compute(targets)) if compute else None)

    def video_targets(self, video):
        from selectvp.preprocessing.object_targets import aggregate_targets
        rows = [dict(self.targets(s), video=video) for s in self.context if s["video"] == video]
        return aggregate_targets(rows)[video]

    def prompted(self, sample, branch, create=False):
        from selectvp.visual_prompts import event, keyframe, object as object_prompt
        cfg = self.config["preprocessing"]
        if branch == "es":
            payload = self.events(sample)
            params = cfg["es"]
        elif branch == "km":
            payload = self.keyframes(sample)
            params = cfg["keyframes"]
        elif branch == "od":
            payload = self.detections(sample)
            params = cfg["detection"]
        else:
            raise ValueError(f"Unknown prompt branch: {branch}")
        identity = {"input": self.input_identity(sample), "branch": branch, "payload": fingerprint(payload), "params": params}
        destination = self.root / "prompted_videos" / branch / f"{artifact_key(sample['id'])}.mp4"

        def render():
            destination.parent.mkdir(parents=True, exist_ok=True)
            flags = None
            if branch == "es":
                flags = event.annotate_video(self.video(sample), destination, payload, cfg["es"])
            elif branch == "km":
                keyframe.annotate_video(self.video(sample), destination, payload)
            else:
                object_prompt.annotate_video(self.video(sample), destination, payload, cfg["detection"]["threshold"])
            return {"sha256": sha256(destination), "is_white_frame": flags}

        metadata = cached(destination.with_suffix(".json"), identity, render if create else None)
        if not destination.is_file() or sha256(destination) != metadata["sha256"]:
            if not create:
                raise ValueError(f"Missing/corrupt prompted video: {destination}")
            metadata = render()
            # Invalidate the sidecar before rewriting its payload.
            destination.with_suffix(".json").unlink(missing_ok=True)
            metadata = cached(destination.with_suffix(".json"), identity, lambda: metadata)
        return destination, metadata["is_white_frame"]

    def run(self, stages):
        unknown = set(stages) - set(STAGES)
        if unknown:
            raise ValueError(f"Unknown preprocessing stages: {unknown}")
        for stage in STAGES:
            if stage not in stages:
                continue
            print(f"Preprocessing {stage}: {len(self.samples)} selected queries", flush=True)
            getattr(self, f"run_{stage}")()
            gc.collect()
            import sys
            if "torch" in sys.modules:
                sys.modules["torch"].cuda.empty_cache()
        write_run_manifest(Path(self.config["paths"]["output"]), self.config, self.samples, "preprocessing")

    def run_videos(self):
        from selectvp.preprocessing.videos import preprocess_video
        for video, output in self.videos.items():
            preprocess_video(safe_video_path(self.config["data"]["raw_dir"], video), output, self.config["video"])

    def run_blip2(self):
        import numpy as np
        from selectvp.preprocessing.blip2_features import Blip2Features
        model = None
        for sample in self.samples:
            for kind in ("visual", "text"):
                try:
                    self.features(sample, kind)
                    continue
                except (ValueError, FileNotFoundError):
                    pass
                if model is None:
                    model = Blip2Features(self.config["preprocessing"]["blip2"], self.config["preprocessing"]["device"])
                key = sample["video"] if kind == "visual" else sample["id"]
                path = self.root / "blip2_features" / f"{kind}_{artifact_key(key)}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                array = model.video(self.video(sample)) if kind == "visual" else model.query(sample["query"])
                np.save(path, array, allow_pickle=False)
                identity = {"blip2": self.config["preprocessing"]["blip2"], "input": self.input_identity(sample) if kind == "text" else
                            {"video": sample["video"], "video_sha256": self.input_identity(sample)["video_sha256"]}}
                path.with_suffix(".json").unlink(missing_ok=True)
                cached(path.with_suffix(".json"), identity, lambda: {"sha256": sha256(path), "shape": list(array.shape)})
                print(f"BLIP-2 {kind}: {sample['id']}", flush=True)

    def run_events(self):
        from selectvp.preprocessing.event_segmentation import event_boundaries
        for sample in self.samples:
            self.events(sample, lambda: event_boundaries(self.features(sample, "visual"), self.features(sample, "text"),
                                                         sample["duration"], self.config["preprocessing"]["es"]))

    def run_targets(self):
        from selectvp.preprocessing.object_targets import TargetExtractor
        model = None
        for sample in self.context:
            def compute():
                nonlocal model
                if model is None:
                    model = TargetExtractor(self.config["preprocessing"]["targets"], self.config["preprocessing"]["device"])
                return model.extract(self.video(sample), sample["query"])
            self.targets(sample, compute)
            print(f"Object targets: {sample['id']}", flush=True)

    def run_detection(self):
        from selectvp.preprocessing.object_detection import ObjectDetector
        model, done = None, set()
        for sample in self.samples:
            if sample["video"] in done:
                continue
            def compute(targets):
                nonlocal model
                if model is None:
                    model = ObjectDetector(self.config["preprocessing"]["detection"], self.config["preprocessing"]["device"])
                return model.detect(self.video(sample), targets)
            self.detections(sample, compute)
            done.add(sample["video"])

    def run_keyframes(self):
        from selectvp.preprocessing.keyframes import KeyframeSelector
        model = None
        for sample in self.samples:
            def compute():
                nonlocal model
                if model is None:
                    model = KeyframeSelector(self.config["preprocessing"]["keyframes"], self.config["preprocessing"]["device"])
                return model.select(self.video(sample), sample["query"])
            self.keyframes(sample, compute)

    def run_prompts(self):
        for sample in self.samples:
            for branch in ("es", "od", "km"):
                self.prompted(sample, branch, create=True)
