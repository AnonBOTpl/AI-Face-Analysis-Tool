import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import logging

from src.config import EMOTION_MODEL, DEVICE

logger = logging.getLogger(__name__)

EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise",
]


class EmotionRecognizer:
    def __init__(self, model_name: str = EMOTION_MODEL):
        self.device = torch.device("cuda:0" if DEVICE == "cuda" else "cpu")
        logger.info("Loading emotion model: %s (device=%s)", model_name, self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, face_img: np.ndarray) -> dict:
        if face_img.size == 0:
            return {"emotion": "Unknown", "emotions": {}}
        try:
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
            id2label = self.model.config.id2label
            if isinstance(probs, list):
                emotions = {id2label.get(str(i), EMOTION_LABELS[i]): round(p, 3)
                            for i, p in enumerate(probs)}
            else:
                emotions = {id2label.get("0", EMOTION_LABELS[0]): round(probs, 3)}
            top_label = max(emotions, key=emotions.get)
            return {"emotion": top_label, "emotions": emotions}
        except Exception as e:
            logger.warning("Emotion prediction failed: %s", e)
            return {"emotion": "Unknown", "emotions": {}}
