"""Insert one white title frame before the video and after each boundary."""

from selectvp.utils.video import iter_frames, video_info, video_writer


def title_frame(width, height, number, style):
    import cv2
    import numpy as np
    text, margin, thickness = f"Event {number}", 10, 4
    scale = style["font_scale"]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    positions = {"top_left": (margin, th + margin), "top_right": (width - tw - margin, th + margin),
                 "bottom_left": (margin, height - margin), "bottom_right": (width - tw - margin, height - margin),
                 "center": ((width - tw) // 2, (height + th) // 2)}
    colors = {"red": (0, 0, 255), "black": (0, 0, 0), "green": (0, 255, 0), "blue": (255, 0, 0)}
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(frame, text, positions[style["position"]], cv2.FONT_HERSHEY_SIMPLEX, scale,
                colors[style["color"]], thickness, cv2.LINE_AA)
    return frame


def annotate_video(source, destination, change_points, style):
    info = video_info(source)
    if abs(info["fps"] - 1) > 1e-5:
        raise ValueError("ES expects the frozen 1 FPS video")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in change_points):
        raise ValueError("Change points must be nonnegative integer frame indices")
    points = set(change_points)
    flags, event = [True], 1
    with video_writer(destination, info["fps"], info["width"], info["height"]) as writer:
        writer.write(title_frame(info["width"], info["height"], event, style))
        for idx, frame in iter_frames(source):
            writer.write(frame)
            flags.append(False)
            if idx in points:
                event += 1
                writer.write(title_frame(info["width"], info["height"], event, style))
                flags.append(True)
    if video_info(destination)["frames"] != len(flags):
        raise RuntimeError("ES title mask does not match encoded frame count")
    return flags
