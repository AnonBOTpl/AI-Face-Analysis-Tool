import cv2
import numpy as np

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


def draw_results(image: np.ndarray, results: list, lang: str = "en") -> np.ndarray:
    img = image.copy()
    for i, res in enumerate(results):
        x1, y1, x2, y2 = res.bbox
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        labels = []
        if res.age is not None:
            labels.append(f"Age: {res.age}")
        if res.gender:
            labels.append(f"{res.gender}")
        if res.emotion:
            if lang == "pl":
                from src.config import EMOTION_LABELS_PL
                label = EMOTION_LABELS_PL.get(res.emotion, res.emotion)
            else:
                label = res.emotion
            labels.append(f"{label}")
        if res.confidence:
            labels.append(f"{res.confidence:.0%}")
        text = " | ".join(labels)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_boxes_preview(image: np.ndarray, results: list) -> np.ndarray:
    img = image.copy()
    for i, res in enumerate(results):
        x1, y1, x2, y2 = res.bbox
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def numpy_to_pixmap(arr: np.ndarray, max_size: int = 800):
    from PyQt6.QtGui import QPixmap, QImage
    h, w = arr.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        arr = cv2.resize(arr, (new_w, new_h))
    if arr.shape[2] == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    bytes_per_line = rgb.shape[1] * 3
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)
