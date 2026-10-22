"""Legacy OpenCV 1 FPS, 336x336 sampling (integer source-FPS step)."""

from pathlib import Path

from selectvp.utils.io import fingerprint, read_json, sha256, write_json
from selectvp.utils.video import iter_frames, video_info, video_writer


def preprocess_video(source, destination, config):
    import cv2
    destination = Path(destination)
    stamp = destination.with_suffix(".meta.json")
    identity = {"source_sha256": sha256(source), "video": config, "implementation": "opencv-integer-fps-v1"}
    signature = fingerprint(identity)
    if destination.exists() and stamp.exists():
        metadata = read_json(stamp)
        if metadata.get("signature") == signature and metadata.get("output_sha256") == sha256(destination):
            video_info(destination)
            return metadata
    info = video_info(source)
    step = int(info["fps"] / config["fps"])
    if step < 1:
        raise ValueError(f"Source FPS is below target FPS: {source}")
    width, height = config["width"], config["height"]
    temporary = destination.with_name(destination.stem + ".tmp.mp4")
    try:
        with video_writer(temporary, config["fps"], width, height) as writer:
            for _, frame in iter_frames(source, range(0, info["frames"], step)):
                writer.write(cv2.resize(frame, (width, height)))
        output = video_info(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {**identity, "signature": signature, "input": info, "output": output,
                "output_sha256": sha256(destination), "annotation_duration_policy": "keep-original"}
    write_json(stamp, metadata)
    return metadata
