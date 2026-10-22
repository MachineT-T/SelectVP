"""One shared Qwen3-VL-8B model for four candidates and their verification."""

from selectvp.models.processor import processor_class
from selectvp.selection.verification import calculate_conf


class Qwen3VL:
    def __init__(self, config):
        import torch
        from transformers import GenerationConfig, Qwen3VLForConditionalGeneration
        self.config = config
        args = {"revision": config["revision"], "local_files_only": config["local_files_only"]}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["id"], dtype=getattr(torch, config["dtype"]), device_map=config["device_map"],
            attn_implementation=config["attention"], **args).eval()
        self.processor = processor_class().from_pretrained(config["id"], **args)
        # Sampling controls from the experiment have no effect with do_sample=False.
        self.generation = GenerationConfig(do_sample=False, max_new_tokens=config["max_new_tokens"],
                                           eos_token_id=self.model.generation_config.eos_token_id,
                                           pad_token_id=self.model.generation_config.pad_token_id)
        self.yes_id = self.processor.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.processor.tokenizer.convert_tokens_to_ids("no")
        if self.yes_id == self.no_id or self.processor.tokenizer.unk_token_id in (self.yes_id, self.no_id):
            raise ValueError("Tokenizer does not support the frozen lowercase yes/no tokens")

    def metadata(self):
        import torch
        return {"model_id": self.config["id"], "requested_revision": self.config["revision"],
                "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                "cuda": torch.version.cuda, "devices": {str(k): str(v) for k, v in self.model.hf_device_map.items()},
                "yes_id": self.yes_id, "no_id": self.no_id,
                "generation": self.generation.to_dict(), "explicit_generate_overrides": {"do_sample": False}}

    def generate(self, video, text, is_white_frame=None, output_logits=False):
        import torch
        messages = [{"role": "user", "content": [{"type": "video", "video": str(video), "fps": 1},
                                                  {"type": "text", "text": text}]}]
        # Keep every preprocessed frame so ES title masks stay aligned.
        kwargs = {"do_sample_frames": False}
        if is_white_frame is not None:
            kwargs["is_white_frame"] = list(is_white_frame)
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt", **kwargs).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=self.generation, do_sample=False,
                                           return_dict_in_generate=output_logits, output_logits=output_logits)
        if output_logits:
            return outputs
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def verify(self, video, text):
        outputs = self.generate(video, text, output_logits=True)
        score = calculate_conf(outputs, self.yes_id, self.no_id)
        generated = outputs.sequences[0, -len(outputs.logits):]
        return score, self.processor.decode(generated, skip_special_tokens=True)
