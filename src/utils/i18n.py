import locale

TRANSLATIONS = {
    "pl": {
        "app_title": "AI Face Analysis Tool",
        "tab_single": "Pojedyncze zdjęcie",
        "tab_batch": "Analiza wsadowa",
        "tab_webcam": "Kamera na żywo",
        "settings": "Ustawienia analizy",
        "detector": "Detektor twarzy:",
        "age_gender_model": "Model wieku/płci:",
        "actions": "Analizuj:",
        "age": "Wiek",
        "gender": "Płeć",
        "emotion": "Emocja",
        "select_image": "Wybierz zdjęcie",
        "select_folder": "Wybierz folder",
        "start_webcam": "Start kamery",
        "stop_webcam": "Stop",
        "analyze": "Analizuj",
        "analyzing": "Analizowanie...",
        "save_image": "Zapisz obraz",
        "save_csv": "Eksportuj CSV",
        "save_json": "Eksportuj JSON",
        "results": "Wyniki",
        "face": "Twarz",
        "no_faces": "Nie wykryto twarzy",
        "error": "Błąd",
        "ready": "Gotowy",
        "processing": "Przetwarzanie...",
        "completed": "Zakończono",
        "image_info": "Informacje o zdjęciu",
        "drag_drop": "Przeciągnij zdjęcie tutaj",
        "webcam_hint": "Kliknij 'Start kamery' aby rozpocząć",
        "batch_progress": "Postęp: {current}/{total}",
        "start_camera": "Uruchom kamerę",
        "language": "Język",
    },
    "en": {
        "app_title": "AI Face Analysis Tool",
        "tab_single": "Single Image",
        "tab_batch": "Batch Processing",
        "tab_webcam": "Live Webcam",
        "settings": "Analysis Settings",
        "detector": "Face Detector:",
        "age_gender_model": "Age/Gender Model:",
        "actions": "Analyze:",
        "age": "Age",
        "gender": "Gender",
        "emotion": "Emotion",
        "select_image": "Select Image",
        "select_folder": "Select Folder",
        "start_webcam": "Start Webcam",
        "stop_webcam": "Stop",
        "analyze": "Analyze",
        "analyzing": "Analyzing...",
        "save_image": "Save Image",
        "save_csv": "Export CSV",
        "save_json": "Export JSON",
        "results": "Results",
        "face": "Face",
        "no_faces": "No faces detected",
        "error": "Error",
        "ready": "Ready",
        "processing": "Processing...",
        "completed": "Completed",
        "image_info": "Image Info",
        "drag_drop": "Drag & drop image here",
        "webcam_hint": "Click 'Start Webcam' to begin",
        "batch_progress": "Progress: {current}/{total}",
        "start_camera": "Start Camera",
        "language": "Language",
    },
}


def detect_language() -> str:
    try:
        lang_code, _ = locale.getdefaultlocale()
        if lang_code and lang_code.startswith("pl"):
            return "pl"
    except Exception:
        pass
    return "en"


_current_lang = detect_language()


def set_language(lang: str):
    global _current_lang
    if lang in TRANSLATIONS:
        _current_lang = lang


def get_language() -> str:
    return _current_lang


def _(key: str) -> str:
    return TRANSLATIONS.get(_current_lang, TRANSLATIONS["en"]).get(key, key)
