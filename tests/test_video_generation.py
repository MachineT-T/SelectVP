"""Exercise real video processing without downloading checkpoints or using a GPU."""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None
                         for name in ("torch", "transformers", "cv2", "av")),
                     "Install the main inference requirements for video regression checks")
class VideoGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessorFast, Qwen3VLVideoProcessor
        from selectvp.models.processor import processor_class

        special = ["<|endoftext|>", "<|vision_start|>", "<|vision_end|>",
                   "<|image_pad|>", "<|video_pad|>"]
        vocab = {token: index for index, token in enumerate(special + sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=special[0]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, additional_special_tokens=special)
        template = ("{% for message in messages %}{% for content in message['content'] %}"
                    "{% if content['type'] == 'video' %}"
                    "<|vision_start|><|video_pad|><|vision_end|>"
                    "{% else %}{{ content['text'] }}{% endif %}{% endfor %}{% endfor %}")
        cls.processor = processor_class()(
            image_processor=Qwen2VLImageProcessorFast(), tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(do_resize=False), chat_template=template)
        cls.torch = torch

    def generate_video(self, frames, with_titles):
        import numpy as np
        from selectvp.models.qwen3_vl import Qwen3VL
        from selectvp.utils.video import video_writer

        torch, processor = self.torch, self.processor
        captured = {}

        def generate(input_ids, video_grid_thw, **kwargs):
            captured["grid"] = video_grid_thw.tolist()
            captured["video_tokens"] = int((input_ids == processor.video_token_id).sum())
            answer = processor.tokenizer("yes", add_special_tokens=False, return_tensors="pt").input_ids
            return torch.cat((input_ids, answer), dim=1)

        model = Qwen3VL.__new__(Qwen3VL)
        model.processor = processor
        model.model = SimpleNamespace(device="cpu", generate=generate)
        model.generation = None
        flags = [index == 0 for index in range(frames)] if with_titles else None
        original_flags = list(flags) if flags is not None else None
        with tempfile.TemporaryDirectory(prefix="selectvp-video-test-") as directory:
            path = Path(directory) / "demo.mp4"
            with video_writer(path, 1, 32, 32) as writer:
                for index in range(frames):
                    writer.write(np.full((32, 32, 3), index % 255, dtype=np.uint8))
            self.assertEqual(model.generate(path, "Find the event", flags), "yes")
        self.assertEqual(flags, original_flags)
        self.assertEqual(captured["grid"], [[(frames + 1) // 2, 2, 2]])
        self.assertEqual(captured["video_tokens"], (frames + 1) // 2)

    def test_es_masks_stay_aligned_for_short_and_long_videos(self):
        for frames in (7, 8, 770, 771):
            with self.subTest(frames=frames):
                self.generate_video(frames, with_titles=True)

    def test_long_unprompted_video_keeps_all_frames(self):
        self.generate_video(770, with_titles=False)


if __name__ == "__main__":
    unittest.main()
