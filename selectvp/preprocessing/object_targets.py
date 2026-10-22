"""Qwen2-VL names and video-level aggregation across all queries."""

import json

TARGET_PROMPT = ('Based on the event description "[EVENT]", your task is to capture a **list of objects/targets** '
                 'that are within the description. These targets will be used for object detection and grounding via '
                 'a object detection model. Choose no more than 3 targets, ideally 2 or fewer. Please response in '
                 'the form of a Python string list(e.g., ["object1", "object2"]).')


def parse_targets(response):
    # Retain original strict JSON parsing and period filtering.
    value = json.loads(response)
    if not isinstance(value, list):
        raise ValueError("Target response must be a JSON list")
    objects = [x.strip() for x in value if isinstance(x, str) and x.strip() and "." not in x]
    if not any(x in objects for x in ("person", "man", "woman", "people")):
        objects.append("person")
    return objects


def aggregate_targets(rows):
    result = {}
    for row in rows:
        result.setdefault(row["video"], set()).update(row["targets"])
    # Source used list(set(...)); sorting removes process-dependent object order.
    return {video: sorted(objects) for video, objects in result.items()}


class TargetExtractor:
    def __init__(self, config, device="cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.config, self.device = config, device
        args = {"revision": config["revision"], "local_files_only": config["local_files_only"]}
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config["id"], dtype=torch.bfloat16, device_map=device, attn_implementation=config["attention"], **args).eval()
        self.processor = AutoProcessor.from_pretrained(config["id"], **args)

    def extract(self, video, query):
        import torch
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [{"type": "text", "text": TARGET_PROMPT.replace("[EVENT]", query)},
                                                  {"type": "video", "video": str(video)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.config["max_new_tokens"], do_sample=False)
        response = self.processor.batch_decode([out[len(inp):] for inp, out in zip(inputs.input_ids, output)],
                                               skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {"response": response, "targets": parse_targets(response)}
