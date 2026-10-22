"""CLIP top-5 with the original 500-frame prefix restriction."""


def sampled_indices(total, max_frames=500):
    if total < 1:
        raise ValueError("Cannot select keyframes from an empty video")
    indices = [int(i * total / max_frames) for i in range(max_frames)] if total > max_frames else list(range(total))
    return [idx for idx in indices if idx < min(total, max_frames)]


class KeyframeSelector:
    def __init__(self, config, device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.config, self.device = config, device
        args = {"revision": config["revision"], "local_files_only": config["local_files_only"]}
        self.model = CLIPModel.from_pretrained(config["id"], use_safetensors=True, **args).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(config["id"], **args)

    def select(self, video, query):
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from selectvp.utils.video import iter_frames, video_info
        indices = sampled_indices(video_info(video)["frames"], self.config["max_frames"])
        features, frames = [], []

        def encode(images):
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                return self.model.get_image_features(**inputs)

        for _, frame in iter_frames(video, indices):
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if len(frames) >= self.config["batch_size"]:
                features.append(encode(frames))
                frames = []
        if frames:
            features.append(encode(frames))
        image_features = torch.cat(features)
        with torch.inference_mode():
            text_features = self.model.get_text_features(**self.processor(text=[query], return_tensors="pt", padding=True).to(self.device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(1).cpu().numpy()
        # For short videos keep only real frames; the old fallback could loop forever.
        top = np.argsort(similarity)[::-1][:self.config["top_k"]]
        return sorted(indices[int(i)] for i in top)
