# AI Face Analysis Tool v2

Nowoczesne narzędzie do analizy twarzy wykorzystujące **MediaPipe** + **InsightFace** + **PyQt6**.
Następca oryginalnego projektu opartego na Tkinter i DeepFace.

## Zastosowane technologie

| Komponent | Technologia | Uwagi |
|---|---|---|
| GUI | PyQt6 (dark theme) | Nowoczesny, stylowany interfejs |
| Detekcja twarzy | MediaPipe BlazeFace / InsightFace RetinaFace | Wybór w UI |
| Wiek + płeć | InsightFace (genderage.onnx) + GPU CUDA | ~96% accuracy |
| Emocje | HuggingFace Transformers (PyTorch + CUDA) | 7 klas emocji |
| Rasa | HuggingFace Transformers (PyTorch + CUDA) | UTKFace |

## Wymagania

- Python 3.10 – 3.12
- NVIDIA GPU z CUDA 12+ (opcjonalnie, działa też na CPU)
- Windows 10/11

## Instalacja

```bash
# 1. Klonujemy repo
git clone <repo-url>
cd AI-Face-Analysis-Tool-main

# 2. Tworzymy wirtualne środowisko
python -m venv venv
.\venv\Scripts\activate

# 3. Instalujemy zależności (w tym PyTorch z CUDA)
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 4. Uruchamiamy
python -m src.main
```

## Użycie

1. **Pojedyncze zdjęcie** — przeciągnij i upuść lub wybierz plik, kliknij "Analizuj"
2. **Analiza wsadowa** — wybierz folder, wyniki w tabeli, eksport do CSV
3. **Kamera na żywo** — wykrywanie i analiza w czasie rzeczywistym
4. **Zapisz wyniki** — obraz z adnotacjami, CSV, JSON

## Ustawienia

- **Detektor**: MediaPipe (szybki, CPU) lub InsightFace (dokładny, GPU)
- **Analizowane cechy**: wiek, płeć, emocje, rasa (z odznaczeniem)
- **Eksport**: obrazy z bounding boxami, CSV z wynikami, JSON

## Różnice względem v1

| Cecha | v1 (Tkinter + DeepFace) | v2 (PyQt6 + MediaPipe + InsightFace) |
|---|---|---|
| GUI | Tkinter | PyQt6 (dark theme, zakładki) |
| Detekcja twarzy | DeepFace (OpenCV/SSD/MTCNN) | MediaPipe BlazeFace / InsightFace RetinaFace |
| Wiek + płeć | DeepFace (TensorFlow) | InsightFace (ONNX + CUDA) |
| Emocje | DeepFace (TensorFlow) | HuggingFace Transformers (PyTorch + CUDA) |
| GPU | αδρανές | CUDA 12+ (GTX 1060+) |
| Webcam | ❌ | ✅ |
| Batch processing | ❌ | ✅ |
| Export CSV/JSON | ❌ | ✅ |
| Drag & drop | ❌ | ✅ |
