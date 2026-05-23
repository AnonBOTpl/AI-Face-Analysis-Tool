# AI Face Analysis Tool v2

Nowoczesne narzędzie do analizy twarzy wykorzystujące **MediaPipe**, **InsightFace**, **PyQt6** i **HuggingFace Transformers** z akceleracją GPU.

**🇬🇧 English version:** [README.md](README.md)

---

## Funkcje

- **Detekcja twarzy** — MediaPipe BlazeFace (CPU) lub InsightFace RetinaFace (GPU)
- **Wiek i płeć** — MiVOLO v2 (~4.22 MAE) lub FaceAge ClientScan (~3.55 MAE, wymaga logowania HF)
- **Rozpoznawanie emocji** — ConvNeXt V2 Large (7 klas, ~97.8% dokładności)
- **Trzy tryby:**
  - Pojedyncze zdjęcie — przeciągnij i upuść, analiza, zapis adnotacji
  - Analiza wsadowa — przetwarzanie folderów, eksport do CSV
  - Kamera na żywo — wykrywanie i analiza w czasie rzeczywistym
- **Eksport** — obrazy z adnotacjami, CSV, JSON
- **Akceleracja GPU** — CUDA 12+ (testowane na GTX 1060 6GB)
- **Wyrównanie twarzy** — MediaPipe FaceMesh, wykrywane twarze są alignowane przed analizą przez wszystkie modele

## Wymagania

- Python 3.10 – 3.12
- NVIDIA GPU z CUDA 12+ (opcjonalnie, działa też na CPU)
- Windows 10/11

## Szybki start

```cmd
:: 1. Klonujemy repo
git clone https://github.com/AnonBOTpl/AI-Face-Analysis-Tool.git
cd AI-Face-Analysis-Tool

:: 2. Uruchamiamy instalator (tworzy venv, instaluje PyTorch + CUDA + zależności)
install.bat

:: 3. Uruchamiamy
uruchom.bat
```

Lub ręcznie:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install git+https://github.com/WildChlamydia/MiVOLO.git --no-build-isolation
python -m src.main
```

## Wybór modelu wieku/płci

| Model | Typ | VRAM | MAE | Dokł. płci | Dostęp |
|---|---|---|---|---|---|
| MiVOLO v2 (domyślny) | Transformer 384×384 | ~300MB | ~4.22 | ~97% | Otwarty |
| FaceAge ClientScan | DINOv3-ViT-L ONNX 224×224 | ~600MB | 3.55 | 97.8% | Gated (wymaga HF login) |
| InsightFace genderage | ONNX 96×96 | ~100MB | ~11.7 | ~95% | Otwarty |

> **Uwaga:** FaceAge ClientScan wymaga uwierzytelnienia w HuggingFace: `huggingface-cli login`

## Technologie

| Komponent | Technologia |
|---|---|
| GUI | PyQt6 (motyw Catppuccin dark) |
| Detekcja twarzy | MediaPipe BlazeFace / InsightFace RetinaFace |
| Wyrównanie twarzy | MediaPipe FaceMesh (468 landmarków) |
| Wiek + płeć | MiVOLO v2 / FaceAge ClientScan / InsightFace |
| Emocje | ConvNeXt V2 Large (HuggingFace) |
| GPU | CUDA 12.8, ONNX Runtime, PyTorch 2.7 |

## Różnice względem v1

| Cecha | v1 (Tkinter + DeepFace) | v2 (PyQt6 + MediaPipe + InsightFace) |
|---|---|---|
| GUI | Tkinter | PyQt6 (dark theme, zakładki) |
| Detekcja twarzy | DeepFace (OpenCV/SSD/MTCNN) | MediaPipe BlazeFace / InsightFace RetinaFace |
| Wiek + płeć | DeepFace (TensorFlow) | MiVOLO v2 / FaceAge ClientScan (MAE 3.5–4.2) |
| Emocje | DeepFace (TensorFlow) | ConvNeXt V2 Large (97.8% acc, PyTorch + CUDA) |
| Wyrównanie twarzy | ❌ | ✅ MediaPipe FaceMesh (468 punktów) |
| GPU | ❌ TensorFlow na CPU | ✅ CUDA 12+ (GTX 1060+) |
| Webcam | ❌ | ✅ |
| Batch processing | ❌ | ✅ |
| Export CSV/JSON | ❌ | ✅ |
| Drag & drop | ❌ | ✅ |
| Zarządzanie VRAM | ❌ | ✅ torch.cuda.empty_cache() między analizami |

## Struktura projektu

```
src/
├── analyzer/
│   ├── alignment.py      # Wyrównanie twarzy (MediaPipe)
│   ├── detector.py       # Backendy detekcji twarzy
│   ├── age_gender.py     # Modele wieku + płci
│   ├── emotion.py        # Klasyfikacja emocji
│   └── pipeline.py       # Orchestrator
├── ui/
│   ├── main_window.py    # GUI PyQt6 (3 zakładki)
│   └── styles.py         # Dark theme QSS
├── utils/
│   ├── image.py          # Rysowanie, podgląd, pixmap
│   └── i18n.py           # Tłumaczenia PL/EN
├── config.py             # Ścieżki, wykrywanie urządzenia
└── main.py               # Punkt wejścia
```

## Licencja

[MIT](LICENSE)
