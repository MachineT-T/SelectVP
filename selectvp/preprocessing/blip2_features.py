"""LAVIS BLIP-2 visual Q-Former and query features in a separate environment."""

from pathlib import Path


class Blip2Features:
    def __init__(self, config, device="cuda"):
        import torch
        from lavis.models import load_model_and_preprocess
        from torchvision import transforms
        self.config, self.device = config, device
        self.model, processors, _ = load_model_and_preprocess(config["name"], config["type"], device=device, is_eval=True)
        self.model.eval()
        self.transform = transforms.Compose([t for t in processors["eval"].transform.transforms
                                              if not isinstance(t, transforms.ToTensor)])

    def video(self, path):
        import cv2
        import numpy as np
        import torch
        from selectvp.utils.video import iter_frames, video_info
        info = video_info(path)
        count = max(1, round(info["duration"] * self.config["fps"]))
        indices = np.linspace(0, info["frames"] - 1, num=count).round().astype(np.int32)
        result, batch = [], []

        def encode(frames):
            tensors = torch.stack(frames).to(self.device)
            with torch.inference_mode():
                images = self.transform(tensors)
                with self.model.maybe_autocast():
                    embeds = self.model.ln_vision(self.model.visual_encoder(images))
                embeds = embeds.float()
                attention = torch.ones(embeds.size()[:-1], dtype=torch.long, device=self.device)
                query = self.model.query_tokens.expand(embeds.shape[0], -1, -1)
                outputs = self.model.Qformer.bert(query_embeds=query, encoder_hidden_states=embeds,
                                                  encoder_attention_mask=attention, return_dict=True)
                return self.model.vision_proj(outputs.last_hidden_state).cpu().half().numpy()

        for _, frame in iter_frames(path, indices):
            batch.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1).float() / 255)
            if len(batch) == self.config["batch_size"]:
                result.append(encode(batch))
                batch = []
        if batch:
            result.append(encode(batch))
        return np.concatenate(result)

    def query(self, query):
        import torch
        with torch.inference_mode():
            text = self.model.tokenizer(query, padding="max_length", truncation=True, max_length=35, return_tensors="pt").to(self.device)
            outputs = self.model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
            features = self.model.text_proj(outputs.last_hidden_state[:, 0, :])
        return features.cpu().float().numpy()
