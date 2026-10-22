"""Original red, one-pixel bounding boxes without labels."""

from selectvp.utils.video import iter_frames, video_info, video_writer


def annotate_video(source, destination, detections, threshold=0.55):
    import cv2
    info = video_info(source)
    if len(detections) != info["frames"]:
        raise ValueError("Detection cache must have one entry per frame")
    with video_writer(destination, info["fps"], info["width"], info["height"]) as writer:
        for idx, frame in iter_frames(source):
            for detection in detections[idx]:
                if detection["score"] > threshold:
                    x1, y1, x2, y2 = map(int, detection["box"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            writer.write(frame)
