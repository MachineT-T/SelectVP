"""Original bottom-right filled red triangle, size 30, margin 20."""

from selectvp.utils.video import iter_frames, video_info, video_writer


def draw_red_triangle(frame):
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    size, margin = 30, 20
    vertices = np.array([(width-margin-size, height-margin), (width-margin, height-margin),
                         (width-margin-size//2, height-margin-size)])
    cv2.drawContours(frame, [vertices], 0, (0, 0, 255), -1)
    return frame


def annotate_video(source, destination, keyframes):
    info = video_info(source)
    if any(int(x) != x or x < 0 for x in keyframes):
        raise ValueError("Keyframes must contain nonnegative integral frame indices")
    keys = set(keyframes)
    with video_writer(destination, info["fps"], info["width"], info["height"]) as writer:
        for idx, frame in iter_frames(source):
            writer.write(draw_red_triangle(frame) if idx in keys else frame)
