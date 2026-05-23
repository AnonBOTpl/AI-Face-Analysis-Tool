import os
import torch

APP_NAME = "AI Face Analysis Tool"
APP_VERSION = "2.0.0"
APP_ICON = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_DEVICE = 0 if DEVICE == "cuda" else -1
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
INSIGHTFACE_DIR = os.path.join(MODELS_DIR, "insightface")
HUGGINGFACE_DIR = os.path.join(MODELS_DIR, "huggingface")
os.makedirs(INSIGHTFACE_DIR, exist_ok=True)
os.makedirs(HUGGINGFACE_DIR, exist_ok=True)

# Redirect HuggingFace cache to local models/ folder
os.environ["HF_HOME"] = HUGGINGFACE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HUGGINGFACE_DIR, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HUGGINGFACE_DIR, "transformers")

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

DETECTOR_BACKENDS = {
    "MediaPipe": "mediapipe",
    "InsightFace (RetinaFace)": "insightface",
}

DEFAULT_AGE_GENDER_MODEL = "mivolo"

EMOTION_MODEL = "DrGM/DrGM-ConvNeXt-V2L-Facial-Emotion-Recognition"
EMOTION_LABELS_PL = {
    "angry": "złość", "disgust": "wstręt", "fear": "strach",
    "happy": "szczęście", "sad": "smutek", "surprise": "zaskoczenie",
    "neutral": "neutralny"
}
