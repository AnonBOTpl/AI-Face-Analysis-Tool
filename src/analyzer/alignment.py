import os
import cv2
import numpy as np
import urllib.request
import logging

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

LEFT_IRIS = 468
RIGHT_IRIS = 473

TARGET_SIZE = 224
DESIRED_EYE_DIST = 90.0

_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)
_LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")


def _ensure_landmarker_model():
    if not os.path.exists(_LANDMARKER_MODEL_PATH):
        logger.info("Downloading MediaPipe face landmarker model...")
        urllib.request.urlretrieve(_LANDMARKER_MODEL_URL, _LANDMARKER_MODEL_PATH)
        logger.info("Model downloaded to %s", _LANDMARKER_MODEL_PATH)


class FaceAligner:
    def __init__(self):
        _ensure_landmarker_model()
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_LANDMARKER_MODEL_PATH),
            num_faces=1,
            min_face_detection_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def _pad_crop(self, img: np.ndarray, factor: float = 0.2) -> np.ndarray:
        h, w = img.shape[:2]
        pad_x, pad_y = int(w * factor), int(h * factor)
        padded = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x,
                                     cv2.BORDER_REPLICATE)
        return padded

    def align(self, face_img: np.ndarray) -> np.ndarray:
        try:
            padded = self._pad_crop(face_img, 0.2)
            h, w = padded.shape[:2]
            from mediapipe import Image as MpImage, ImageFormat
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            mp_img = MpImage(ImageFormat.SRGB, rgb)
            result = self.landmarker.detect(mp_img)

            if not result.face_landmarks:
                return cv2.resize(face_img, (TARGET_SIZE, TARGET_SIZE))

            lm = result.face_landmarks[0]
            left_eye = np.array([lm[LEFT_IRIS].x * w, lm[LEFT_IRIS].y * h])
            right_eye = np.array([lm[RIGHT_IRIS].x * w, lm[RIGHT_IRIS].y * h])

            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))

            actual_dist = np.linalg.norm(right_eye - left_eye)
            scale = DESIRED_EYE_DIST / actual_dist if actual_dist > 0 else 1.0

            center = ((left_eye + right_eye) / 2).tolist()

            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += TARGET_SIZE * 0.5 - center[0]
            M[1, 2] += TARGET_SIZE * 0.4 - center[1]

            aligned = cv2.warpAffine(padded, M, (TARGET_SIZE, TARGET_SIZE),
                                      flags=cv2.INTER_LINEAR)
            return aligned

        except Exception as e:
            logger.warning("Face alignment failed, fallback to resize: %s", e)
            return cv2.resize(face_img, (TARGET_SIZE, TARGET_SIZE))
