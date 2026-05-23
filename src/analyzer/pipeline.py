import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.analyzer.detector import create_detector
from src.analyzer.alignment import FaceAligner
from src.analyzer.age_gender import create_age_gender_model
from src.analyzer.emotion import EmotionRecognizer
from src.config import DEVICE, CUDA_DEVICE, DEFAULT_AGE_GENDER_MODEL

logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    bbox: tuple[int, int, int, int]
    confidence: float
    age: Optional[int] = None
    gender: Optional[str] = None
    gender_confidence: Optional[float] = None
    emotion: Optional[str] = None
    emotions: dict = field(default_factory=dict)


class AnalysisPipeline:
    def __init__(
        self,
        detector_backend: str = "mediapipe",
        age_gender_model: str = DEFAULT_AGE_GENDER_MODEL,
        enable_age: bool = True,
        enable_gender: bool = True,
        enable_emotion: bool = True,
    ):
        use_gpu = DEVICE == "cuda"
        logger.info("Initializing pipeline (detector=%s, ag_model=%s, gpu=%s)",
                     detector_backend, age_gender_model, use_gpu)
        self.detector = create_detector(detector_backend, use_gpu=use_gpu)
        self.detector_backend = detector_backend
        self.age_gender_model = age_gender_model
        self.aligner = FaceAligner() if detector_backend == "mediapipe" else None
        if (enable_age or enable_gender) and detector_backend == "mediapipe":
            try:
                self.age_gender = create_age_gender_model(age_gender_model, ctx_id=CUDA_DEVICE)
            except Exception as e:
                logger.error("Failed to load age/gender model '%s': %s", age_gender_model, e)
                self.age_gender = None
        else:
            self.age_gender = None
        self.emotion = EmotionRecognizer() if enable_emotion else None
        self.enable_age = enable_age
        self.enable_gender = enable_gender
        self.enable_emotion = enable_emotion

    def analyze(self, image: np.ndarray) -> list[FaceResult]:
        if image is None or image.size == 0:
            raise ValueError("Empty image")
        faces = self.detector.detect(image)
        results = []
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            face_img = image[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            result = FaceResult(
                bbox=face["bbox"],
                confidence=face["confidence"],
            )
            if self.aligner:
                try:
                    face_img = self.aligner.align(face_img)
                except Exception as e:
                    logger.warning("Alignment failed: %s", e)
            if self.detector_backend == "insightface" and "_face_obj" in face:
                fo = face["_face_obj"]
                if self.enable_age:
                    result.age = max(0, min(100, int(round(fo.age))))
                if self.enable_gender:
                    gv = fo.gender
                    result.gender = "Man" if gv > 0.5 else "Woman"
                    result.gender_confidence = round(float(gv) if gv > 0.5 else float(1.0 - gv), 3)
            elif self.age_gender:
                try:
                    ag = self.age_gender.predict(face_img)
                    if self.enable_age:
                        result.age = ag["age"]
                    if self.enable_gender:
                        result.gender = ag["gender"]
                        result.gender_confidence = ag["gender_confidence"]
                except Exception as e:
                    logger.warning("Age/gender prediction failed: %s", e)
            if self.emotion:
                try:
                    em = self.emotion.predict(face_img)
                    result.emotion = em["emotion"]
                    result.emotions = em["emotions"]
                except Exception as e:
                    logger.warning("Emotion prediction failed: %s", e)
            results.append(result)
        return results

    def cleanup(self):
        import torch
        import gc
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
