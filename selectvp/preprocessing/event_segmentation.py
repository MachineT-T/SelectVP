"""Query-conditioned BLIP-2 selection, temporal smoothing and MeanShift.

Retain the experiment's boundary indexing: cluster transition at i gives i-1
in the masked sequence, then int(index / original_frame_count * duration).
It deliberately does not remap through the retained-frame indices.
"""


def event_boundaries(video_features, query_features, duration, config):
    import numpy as np
    from sklearn.cluster import MeanShift
    video = np.asarray(video_features, dtype=np.float32)
    query = np.asarray(query_features, dtype=np.float32)
    if video.ndim != 3 or query.ndim != 2 or video.shape[-1] != query.shape[-1] or not len(video):
        raise ValueError("Expected BLIP-2 features [frames, queries, dim] and text [queries, dim]")
    if not np.isfinite(video).all() or not np.isfinite(query).all():
        raise ValueError("Non-finite BLIP-2 features")
    normalized_video = video / np.maximum(np.linalg.norm(video, axis=-1, keepdims=True), 1e-12)
    normalized_query = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-12)
    similarities = np.einsum("md,npd->mnp", normalized_query, normalized_video)
    scores = similarities.max(axis=-1).mean(axis=0)
    # One text query per sample in the source implementation.
    if len(query) != 1:
        raise ValueError("Exactly one query embedding per sample is required")
    selected = video[np.arange(len(video)), similarities.argmax(axis=-1)[0]]
    mask = scores > config["similarity_threshold"]
    if len(scores) >= 3:
        padded = np.pad(mask, (1, 1))
        final = padded.copy()
        for i in range(2, len(padded) - 1):
            if padded[i-1:i+2].sum() < 2:
                final[i] = False
        mask = final[1:-1]
    if not mask.any():
        mask[:] = True
    time = (np.arange(len(video), dtype=np.float32) / len(video))[:, None]
    features = np.concatenate([selected, time], axis=1)[mask]
    width = config["temporal_window_size"]
    if not isinstance(width, int) or width < 1 or width % 2 != 1:
        raise ValueError("Temporal smoothing window must be positive and odd")
    padded_features = np.pad(features, ((width//2, width//2), (0, 0)), mode="edge")
    # Same depthwise moving average as the original torch.conv1d implementation.
    smoothed = sum(padded_features[i:i+len(features)] for i in range(width)) / width
    labels = MeanShift(n_jobs=1).fit(smoothed).labels_ if len(features) > 1 else np.zeros(1, dtype=int)
    transitions = np.flatnonzero(labels[1:] != labels[:-1])
    return [int(int(index) / len(video) * duration) for index in transitions]
