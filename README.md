# AI Face Analysis Tool v2

A modern face analysis desktop app using **MediaPipe**, **InsightFace**, **PyQt6**, and **HuggingFace Transformers** with GPU acceleration.

**🇵🇱 Polish version:** [README_PL.md](README_PL.md)

---

## Features

- **Face Detection** — MediaPipe BlazeFace (CPU) or InsightFace RetinaFace (GPU)
- **Age & Gender** — MiVOLO v2 (~4.22 MAE) or FaceAge ClientScan (~3.55 MAE, requires HF login)
- **Emotion Recognition** — ConvNeXt V2 Large (7 classes, ~97.8% accuracy)
- **Three modes:**
  - Single Image — drag & drop, analyze, save annotations
  - Batch Processing — analyze folders, export CSV
  - Live Webcam — real-time detection and analysis
- **Export** — annotated images, CSV, JSON
- **GPU acceleration** — CUDA 12+ (tested on GTX 1060 6GB)
- **Face alignment** — MediaPipe FaceMesh landmarker, aligned face crops for all models

## Requirements

- Python 3.10 – 3.12
- NVIDIA GPU with CUDA 12+ (optional, CPU fallback works)
- Windows 10/11

## Quick Start

```cmd
:: 1. Clone
git clone https://github.com/AnonBOTpl/AI-Face-Analysis-Tool.git
cd AI-Face-Analysis-Tool

:: 2. Run installer (creates venv, installs PyTorch + CUDA + dependencies)
install.bat

:: 3. Launch
uruchom.bat
```

Or manually:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install git+https://github.com/WildChlamydia/MiVOLO.git --no-build-isolation
python -m src.main
```

## Model Selection

| Model | Type | VRAM | MAE | Gender Acc | Access |
|---|---|---|---|---|---|
| MiVOLO v2 (default) | Transformer 384×384 | ~300MB | ~4.22 | ~97% | Open |
| FaceAge ClientScan | DINOv3-ViT-L ONNX 224×224 | ~600MB | 3.55 | 97.8% | Gated (HF login) |
| InsightFace genderage | ONNX 96×96 | ~100MB | ~11.7 | ~95% | Open |

> **Note:** FaceAge ClientScan requires HuggingFace authentication: `huggingface-cli login`

## Tech Stack

| Component | Technology |
|---|---|
| GUI | PyQt6 (Catppuccin dark theme) |
| Face Detection | MediaPipe BlazeFace / InsightFace RetinaFace |
| Face Alignment | MediaPipe FaceMesh (468 landmarks) |
| Age + Gender | MiVOLO v2 / FaceAge ClientScan / InsightFace |
| Emotion | ConvNeXt V2 Large (HuggingFace) |
| GPU | CUDA 12.8, ONNX Runtime, PyTorch 2.7 |

## Differences from v1

| Feature | v1 (Tkinter + DeepFace) | v2 (PyQt6 + MediaPipe + InsightFace) |
|---|---|---|
| GUI | Tkinter | PyQt6 (dark theme, tabs) |
| Face Detection | DeepFace (OpenCV/SSD/MTCNN) | MediaPipe BlazeFace / InsightFace RetinaFace |
| Age + Gender | DeepFace (TensorFlow) | MiVOLO v2 / FaceAge ClientScan (MAE 3.5–4.2) |
| Emotion | DeepFace (TensorFlow) | ConvNeXt V2 Large (97.8% acc, PyTorch + CUDA) |
| Face Alignment | ❌ | ✅ MediaPipe FaceMesh (468 landmarks) |
| GPU | ❌ TensorFlow on CPU | ✅ CUDA 12+ (GTX 1060+) |
| Webcam | ❌ | ✅ |
| Batch Processing | ❌ | ✅ |
| Export CSV/JSON | ❌ | ✅ |
| Drag & Drop | ❌ | ✅ |
| VRAM Management | ❌ | ✅ torch.cuda.empty_cache() between analyses |

## Project Structure

```
src/
├── analyzer/
│   ├── alignment.py      # MediaPipe face alignment
│   ├── detector.py       # Face detection backends
│   ├── age_gender.py     # Age + gender models
│   ├── emotion.py        # Emotion classification
│   └── pipeline.py       # Orchestrator
├── ui/
│   ├── main_window.py    # PyQt6 UI (3 tabs)
│   └── styles.py         # Dark theme QSS
├── utils/
│   ├── image.py          # Drawing, preview, pixmap
│   └── i18n.py           # PL/EN translations
├── config.py             # Paths, device detection
└── main.py               # Entry point
```

## License

[MIT](LICENSE)
