import os
import cv2
import numpy as np
import logging

from src.config import INSIGHTFACE_DIR, HUGGINGFACE_DIR

logger = logging.getLogger(__name__)

GENDERAGE_MODEL_PATH = os.path.join(INSIGHTFACE_DIR, "models", "buffalo_l", "genderage.onnx")


AGE_GENDER_MODELS = {
    "insightface": {
        "name": "InsightFace genderage",
        "description": "ONNX genderage.onnx, 96×96, ~100MB VRAM, MAE ~11.7",
    },
    "faceage": {
        "name": "FaceAge ClientScan",
        "description": "DINOv3-ViT-L ONNX, 224×224, ~600MB VRAM, MAE 3.55, Gender 97.8%. Wymaga `huggingface-cli login` (model gated)",
    },
    "mivolo": {
        "name": "MiVOLO v2",
        "description": "Transformer 384×384 (HF), ~300MB VRAM, MAE ~4.22, Gender ~97%",
    },
}


class InsightFaceGenderAge:
    def __init__(self, ctx_id: int = -1):
        import insightface
        if not os.path.exists(GENDERAGE_MODEL_PATH):
            logger.info("Downloading InsightFace genderage model...")
            app = insightface.app.FaceAnalysis(
                name="buffalo_l", root=INSIGHTFACE_DIR,
            )
            app.prepare(ctx_id=-1)

        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ctx_id >= 0
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(GENDERAGE_MODEL_PATH, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (96, 96)

    def predict(self, face_img: np.ndarray) -> dict:
        if face_img.size == 0:
            return {"age": 0, "gender": "Unknown", "gender_confidence": 0.0}
        try:
            aimg = cv2.resize(face_img, self.input_size)
            blob = cv2.dnn.blobFromImage(
                aimg, 1.0 / 128.0, self.input_size,
                (127.5, 127.5, 127.5), swapRB=True,
            )
            pred = self.session.run(None, {self.input_name: blob})[0][0]
            gender_idx = int(np.argmax(pred[:2]))
            age_val = int(np.round(pred[2] * 100))
            gender_label = "Man" if gender_idx == 1 else "Woman"
            gender_conf = float(pred[1]) if gender_idx == 1 else float(pred[0])
            return {
                "age": max(0, min(100, age_val)),
                "gender": gender_label,
                "gender_confidence": round(max(gender_conf, 0.5), 3),
            }
        except Exception as e:
            logger.warning("InsightFace genderage failed: %s", e)
            return {"age": 30, "gender": "Woman", "gender_confidence": 0.6}


class FaceAgeClientScan:
    def __init__(self):
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort

        logger.info("Loading FaceAge ClientScan model...")
        try:
            model_path = hf_hub_download(
                repo_id="TrungTran/faceage_ClientScan",
                filename="faceage_dino_fp32.onnx",
                cache_dir=os.path.join(HUGGINGFACE_DIR, "hub"),
            )
        except Exception as e:
            logger.error(
                "FaceAge ClientScan requires HuggingFace authentication. "
                "Run: huggingface-cli login"
            )
            raise RuntimeError(
                "FaceAge ClientScan to model gated. "
                "Uruchom 'huggingface-cli login' z tokenem z huggingface.co/settings/tokens, "
                "lub wybierz model MiVOLO v2 w ustawieniach."
            ) from e

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (224, 224)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict(self, face_img: np.ndarray) -> dict:
        if face_img.size == 0:
            return {"age": 0, "gender": "Unknown", "gender_confidence": 0.0}
        try:
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, self.input_size).astype(np.float32)
            arr = (resized / 255.0 - self.mean) / self.std
            blob = arr.transpose(2, 0, 1)[np.newaxis]

            age_logits, gender_logits = self.session.run(
                None, {self.input_name: blob}
            )
            age = float((1.0 / (1.0 + np.exp(-age_logits[0]))).sum())
            gender_idx = int(np.argmax(gender_logits[0]))
            gender_label = "Man" if gender_idx == 1 else "Woman"
            gender_conf = float(np.max(gender_logits[0]))
            return {
                "age": max(0, min(100, int(round(age)))),
                "gender": gender_label,
                "gender_confidence": round(gender_conf, 3),
            }
        except Exception as e:
            logger.warning("FaceAge ClientScan failed: %s", e)
            return {"age": 30, "gender": "Woman", "gender_confidence": 0.6}


class MiVOLOv2:
    def __init__(self, device_id: int = 0):
        import torch
        from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor

        logger.info("Loading MiVOLO v2 model...")
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        repo = "iitolstykh/mivolo_v2"

        self.config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForImageClassification.from_pretrained(
            repo, trust_remote_code=True, dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.image_processor = AutoImageProcessor.from_pretrained(
            repo, trust_remote_code=True,
        )

    def predict(self, face_img: np.ndarray) -> dict:
        if face_img.size == 0:
            return {"age": 0, "gender": "Unknown", "gender_confidence": 0.0}
        try:
            import torch
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            inputs = self.image_processor(images=[rgb], return_tensors="pt")
            faces_input = inputs["pixel_values"].to(
                dtype=self.model.dtype, device=self.device
            )
            with torch.no_grad():
                output = self.model(faces_input=faces_input, body_input=[None])
            age = output.age_output[0].item()
            gender_idx = output.gender_class_idx[0].item()
            id2label = self.config.gender_id2label
            gender_label = id2label[gender_idx]
            gender_conf = output.gender_probs[0].item()

            return {
                "age": max(0, min(100, int(round(age)))),
                "gender": gender_label,
                "gender_confidence": round(gender_conf, 3),
            }
        except Exception as e:
            logger.warning("MiVOLO v2 failed: %s", e)
            return {"age": 30, "gender": "Woman", "gender_confidence": 0.6}


def create_age_gender_model(model_name: str, ctx_id: int = -1):
    if model_name == "insightface":
        return InsightFaceGenderAge(ctx_id=ctx_id)
    elif model_name == "faceage":
        return FaceAgeClientScan()
    elif model_name == "mivolo":
        return MiVOLOv2(device_id=max(0, ctx_id))
    raise ValueError(f"Unknown age/gender model: {model_name}")
