import os
import cv2
import numpy as np
import urllib.request
import logging

from src.config import MODELS_DIR, INSIGHTFACE_DIR

logger = logging.getLogger(__name__)

_BLAZE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/latest/"
    "blaze_face_short_range.tflite"
)
_BLAZE_MODEL_PATH = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")


def _ensure_blaze_model():
    if not os.path.exists(_BLAZE_MODEL_PATH):
        logger.info("Downloading MediaPipe face detection model...")
        urllib.request.urlretrieve(_BLAZE_MODEL_URL, _BLAZE_MODEL_PATH)
        logger.info("Model downloaded to %s", _BLAZE_MODEL_PATH)


class MediaPipeDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        _ensure_blaze_model()
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=_BLAZE_MODEL_PATH),
            min_detection_confidence=min_detection_confidence,
        )
        self.detector = FaceDetector.create_from_options(options)

    def detect(self, image: np.ndarray) -> list[dict]:
        from mediapipe import Image as MpImage
        from mediapipe import ImageFormat
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img = MpImage(ImageFormat.SRGB, rgb)
        results = self.detector.detect(mp_img)
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for det in results.detections:
                bbox = det.bounding_box
                x1 = max(0, int(bbox.origin_x))
                y1 = max(0, int(bbox.origin_y))
                x2 = min(w, int(bbox.origin_x + bbox.width))
                y2 = min(h, int(bbox.origin_y + bbox.height))
                faces.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(det.categories[0].score),
                    "detector": "mediapipe",
                })
        return faces


class InsightFaceDetector:
    def __init__(self, ctx_id: int = -1):
        import insightface
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ctx_id >= 0
            else ["CPUExecutionProvider"]
        )
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=INSIGHTFACE_DIR,
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id)

    def detect(self, image: np.ndarray) -> list[dict]:
        faces = self.app.get(image)
        results = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            results.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(face.det_score),
                "detector": "insightface",
                "_face_obj": face,
            })
        return results


def create_detector(backend: str, use_gpu: bool = True):
    ctx_id = 0 if use_gpu else -1
    if backend == "mediapipe":
        return MediaPipeDetector()
    elif backend == "insightface":
        return InsightFaceDetector(ctx_id=ctx_id)
    raise ValueError(f"Unknown detector backend: {backend}")
