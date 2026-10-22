"""The original Transformers port of MM-Grounding-DINO Large All."""


class ObjectDetector:
    def __init__(self, config, device="cuda"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.config, self.device = config, device
        args = {"revision": config["revision"], "local_files_only": config["local_files_only"]}
        self.processor = AutoProcessor.from_pretrained(config["id"], **args)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(config["id"], **args).to(device).eval()

    def detect(self, video, targets):
        import cv2
        import torch
        from contextlib import nullcontext
        from PIL import Image
        from selectvp.utils.video import iter_frames
        labels = [[obj if obj.startswith(("a ", "an ")) else f"a {obj}" for obj in targets]]
        result = []
        for _, frame in iter_frames(video):
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=image, text=labels, return_tensors="pt").to(self.device)
            context = torch.autocast("cuda") if str(self.device).startswith("cuda") else nullcontext()
            with torch.inference_mode(), context:
                outputs = self.model(**inputs)
            detected = self.processor.post_process_grounded_object_detection(outputs, threshold=self.config["threshold"],
                                                                              target_sizes=[frame.shape[:2]])[0]
            names = detected.get("text_labels", detected.get("labels"))
            result.append([{"box": box.tolist(), "score": float(score), "label": str(label)}
                           for box, score, label in zip(detected["boxes"], detected["scores"], names)])
        return result
