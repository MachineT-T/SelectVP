# Adapted from Transformers 4.57.1.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See THIRD_PARTY_NOTICES.md.
# Modified for ES title masks and the original experiment's timestamp semantics.
"""ES timestamps and lazy construction of the Qwen3 visual processor.

The multimodal token expansion is adapted from Transformers 4.57.1 and the
original experiment's Qwen3VLChangePointProcessor. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations


def event_timestamps(indices, video_fps, merge_size=2, is_white_frame=None):
    indices = list(indices)
    if not indices or video_fps <= 0 or merge_size < 1:
        raise ValueError("Empty video or invalid timestamp parameters")
    if is_white_frame is None:
        pad = (-len(indices)) % merge_size
        indices += [indices[-1]] * pad
        times = [idx / video_fps for idx in indices]
        return [(times[i] + times[i+merge_size-1]) / 2 for i in range(0, len(times), merge_size)]
    flags = list(is_white_frame)
    if len(indices) != len(flags):
        raise ValueError("ES mask length does not match sampled frames; do not subsample prompted videos")
    flags += [flags[-1]] * ((-len(flags)) % merge_size)
    # Preserve the original tail rule: repeated non-title frames advance time.
    times = [0.0]
    for previous_title in flags[:-1]:
        times.append(times[-1] if previous_title else times[-1] + 1.0 / video_fps)
    return [(times[i] + times[i+merge_size-1]) / 2 for i in range(0, len(times), merge_size)]


def processor_class():
    import numpy as np
    from transformers import Qwen3VLProcessor
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorKwargs

    class Qwen3VLChangePointProcessor(Qwen3VLProcessor):
        def _calculate_timestamps(self, indices, video_fps, merge_size=2, is_white_frame=None):
            return event_timestamps(indices, video_fps, merge_size, is_white_frame)

        def __call__(self, images=None, text=None, videos=None, **kwargs):
            flags = kwargs.pop("is_white_frame", None)
            if flags is None:
                return super().__call__(images=images, text=text, videos=videos, **kwargs)
            if images is not None:
                raise ValueError("The ES processor accepts a single video without images")
            flags = list(flags)
            output_kwargs = self._merge_kwargs(Qwen3VLProcessorKwargs,
                                               tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs)
            video_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            grids = video_inputs["video_grid_thw"]
            metadata = video_inputs.pop("video_metadata")
            if len(grids) != 1:
                raise ValueError("ES generation is sequential, one video per call")
            texts = [text] if isinstance(text, str) else list(text)
            if len(texts) != 1:
                raise ValueError("ES accepts one prompt per video")
            meta = metadata[0]
            if meta.fps is None:
                raise ValueError("ES requires explicit video FPS")
            times = event_timestamps(meta.frames_indices, meta.fps, self.video_processor.merge_size, flags)
            merge = self.video_processor.merge_size
            padded = flags + [flags[-1]] * ((-len(flags)) % merge)
            frame_tokens = int(grids[0][1:].prod()) // merge**2
            if int(grids[0][0]) != len(times):
                raise ValueError("ES temporal grid and timestamp count disagree")
            placeholder = ""
            for idx, timestamp in enumerate(times):
                if not padded[idx * merge]:
                    placeholder += f"<{timestamp:.1f} seconds>"
                placeholder += self.vision_start_token + "<|placeholder|>" * frame_tokens + self.vision_end_token
            wrapped = self.vision_start_token + self.video_token + self.vision_end_token
            if wrapped in texts[0]:
                texts[0] = texts[0].replace(wrapped, placeholder, 1)
            elif self.video_token in texts[0]:
                texts[0] = texts[0].replace(self.video_token, placeholder, 1)
            else:
                raise ValueError("Missing video token in ES prompt")
            texts[0] = texts[0].replace("<|placeholder|>", self.video_token)
            return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
            mm_types = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
            text_inputs = self.tokenizer(texts, **output_kwargs["text_kwargs"])
            self._check_special_mm_tokens(texts, text_inputs, modalities=["image", "video"])
            if mm_types:
                array = np.array(text_inputs["input_ids"])
                text_inputs["mm_token_type_ids"] = (array == self.image_token_id).astype(int).tolist()
            return BatchFeature(data={**text_inputs, **video_inputs}, tensor_type=return_tensors)

    return Qwen3VLChangePointProcessor
