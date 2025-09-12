from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import shutil
import os
import uuid
import sys
import traceback
import locale

# ------------------ Translacje ------------------
translations = {
    "pl": {
        "choose_detector_title": "Wybór detektora DeepFace",
        "choose_detector": "Wybierz detektor twarzy",
        "detector_label": "Detektor twarzy:",
        "ok": "OK",
        "cancel": "Anuluj",
        "detectors_info": """Detektory:
• opencv - szybki, podstawowy (dobry start)
• ssd - średni, lepszy od opencv
• mtcnn - wolniejszy, dokładniejszy
• retinaface - najlepszy, najwolniejszy (polecany dla dzieci)

Program używa domyślnych modeli DeepFace dla wieku, płci, rasy i emocji.""",
        "file_dialog_title": "Wybierz zdjęcie do analizy",
        "no_detector": "❌ Nie wybrano detektora. Zamykanie programu.",
        "selected_detector": "✅ Wybrany detektor: {det}",
        "no_file": "❌ Nie wybrano pliku. Zamykanie programu.",
        "file_selected": "✅ Wybrano plik: {file}",
        "copy_fail": "❌ Nie udało się skopiować pliku do katalogu tymczasowego:",
        "analyzing": "⏳ Analizowanie...",
        "no_faces": "❌ Nie wykryto twarzy na obrazie.",
        "faces_detected": "✅ Wykryto {n} twarzy",
        "face": "🔍 Twarz #{i}",
        "age": "   Wiek: {age} lat (przedział: {range})",
        "gender": "   Płeć - Kobieta: {w:.1f}%, Mężczyzna: {m:.1f}%",
        "race": "   Rasa (top 3):",
        "emotion": "   Emocje (top 3):",
        "child_warning": "   ⚠️ UWAGA: Wykryty wiek {age} lat - wyniki mogą być mniej dokładne dla dzieci!",
        "summary": "\n📊 Podsumowanie wyników:",
        "image_fail": "❌ Nie udało się wczytać tymczasowego obrazu do OpenCV.",
        "plot_title": "Analiza twarzy z DeepFace (detektor: {det})",
        "finished": "🏁 Program zakończony"
    },
    "en": {
        "choose_detector_title": "DeepFace Detector Selection",
        "choose_detector": "Choose face detector",
        "detector_label": "Face detector:",
        "ok": "OK",
        "cancel": "Cancel",
        "detectors_info": """Detectors:
• opencv - fast, basic (good start)
• ssd - medium, better than opencv
• mtcnn - slower, more accurate
• retinaface - best, slowest (recommended for children)

The program uses default DeepFace models for age, gender, race and emotions.""",
        "file_dialog_title": "Select a photo for analysis",
        "no_detector": "❌ No detector selected. Exiting program.",
        "selected_detector": "✅ Selected detector: {det}",
        "no_file": "❌ No file selected. Exiting program.",
        "file_selected": "✅ File selected: {file}",
        "copy_fail": "❌ Failed to copy file to temp directory:",
        "analyzing": "⏳ Analyzing...",
        "no_faces": "❌ No faces detected in image.",
        "faces_detected": "✅ {n} face(s) detected",
        "face": "🔍 Face #{i}",
        "age": "   Age: {age} years (range: {range})",
        "gender": "   Gender - Woman: {w:.1f}%, Man: {m:.1f}%",
        "race": "   Race (top 3):",
        "emotion": "   Emotions (top 3):",
        "child_warning": "   ⚠️ WARNING: Detected age {age} years - results may be less accurate for children!",
        "summary": "\n📊 Results summary:",
        "image_fail": "❌ Failed to load temporary image into OpenCV.",
        "plot_title": "DeepFace Face Analysis (detector: {det})",
        "finished": "🏁 Program finished"
    }
}

# ------------------ Wybór języka z systemu ------------------
lang, _ = locale.getdefaultlocale()
if lang and lang.startswith("pl"):
    LANG = "pl"
else:
    LANG = "en"

T = translations[LANG]

# ------------------ Tkinter ------------------
print("🔍 Testowanie tkinter...")
try:
    import tkinter as tk
    from tkinter import filedialog, ttk
    print("✅ Tkinter zaimportowany pomyślnie")
except Exception as e:
    print(f"❌ Błąd z tkinter: {e}")
    sys.exit(1)

# ---- Funkcja wyboru detektora ----
def choose_detector():
    try:
        root = tk.Tk()
        root.title(T["choose_detector_title"])
        root.geometry("500x300")
        root.lift()
        root.attributes('-topmost', True)

        detector_var = tk.StringVar(value='opencv')

        tk.Label(root, text=T["choose_detector"], font=("Arial", 16, "bold")).pack(pady=20)

        frame_detector = tk.Frame(root)
        frame_detector.pack(fill="x", padx=40, pady=20)

        tk.Label(frame_detector, text=T["detector_label"], font=("Arial", 12), width=15, anchor="w").pack(side="left")

        detector_combo = ttk.Combobox(frame_detector, textvariable=detector_var,
                                     values=['opencv', 'ssd', 'mtcnn', 'retinaface'],
                                     state="readonly", font=("Arial", 11))
        detector_combo.pack(side="right", fill="x", expand=True, padx=(20,0))

        tk.Label(root, text=T["detectors_info"], justify="left", font=("Arial", 10), wraplength=450).pack(pady=20)

        result = {'cancelled': True}

        def on_ok():
            result['cancelled'] = False
            result['detector'] = detector_var.get()
            root.quit()

        def on_cancel():
            result['cancelled'] = True
            root.quit()

        button_frame = tk.Frame(root)
        button_frame.pack(side="bottom", pady=20)

        ok_btn = tk.Button(button_frame, text=T["ok"], command=on_ok,
                          bg="#4CAF50", fg="white", padx=30, font=("Arial", 12, "bold"))
        ok_btn.pack(side="left", padx=15)

        cancel_btn = tk.Button(button_frame, text=T["cancel"], command=on_cancel,
                              bg="#f44336", fg="white", padx=25, font=("Arial", 12, "bold"))
        cancel_btn.pack(side="left", padx=15)

        root.deiconify()
        root.focus_force()
        root.mainloop()
        root.destroy()

        if result['cancelled']:
            return None
        else:
            return result['detector']

    except Exception as e:
        traceback.print_exc()
        return None

# ------------------ Główna część programu ------------------
print("🚀 Uruchamianie programu...")

selected_detector = choose_detector()
if not selected_detector:
    print(T["no_detector"])
    sys.exit(0)

print(T["selected_detector"].format(det=selected_detector))

root = tk.Tk()
root.withdraw()
img_path = filedialog.askopenfilename(
    title=T["file_dialog_title"],
    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
)
root.destroy()

if not img_path:
    print(T["no_file"])
    sys.exit(0)

print(T["file_selected"].format(file=img_path))

ext = os.path.splitext(img_path)[1]
tmp_path = os.path.join(tempfile.gettempdir(), f"tmp_img_{uuid.uuid4().hex}{ext}")

try:
    shutil.copy2(img_path, tmp_path)
except Exception as e:
    print(T["copy_fail"], e)
    sys.exit(1)

try:
    print(T["analyzing"])
    results = DeepFace.analyze(
        img_path=tmp_path,
        actions=['age', 'gender', 'race', 'emotion'],
        detector_backend=selected_detector,
        enforce_detection=False
    )

    if isinstance(results, dict):
        results = [results]

    if not results:
        print(T["no_faces"])
        sys.exit(0)

    print(T["faces_detected"].format(n=len(results)))

    rows = []
    for idx, r in enumerate(results):
        print(T["face"].format(i=idx+1))

        age = r.get("age", None)
        age_text = f"{int(age)-3}-{int(age)+3}" if age is not None else "?"

        print(T["age"].format(age=int(age) if age else "N/A", range=age_text))

        gender_data = r.get("gender", {})
        if isinstance(gender_data, dict):
            woman_conf = gender_data.get('Woman', 0)
            man_conf = gender_data.get('Man', 0)
            print(T["gender"].format(w=woman_conf, m=man_conf))

        race_data = r.get("race", {})
        if isinstance(race_data, dict):
            print(T["race"])
            sorted_races = sorted(race_data.items(), key=lambda x: x[1], reverse=True)[:3]
            for race, conf in sorted_races:
                print(f"     {race}: {conf:.1f}%")

        emotion_data = r.get("emotion", {})
        if isinstance(emotion_data, dict):
            print(T["emotion"])
            sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)[:3]
            for emotion, conf in sorted_emotions:
                print(f"     {emotion}: {conf:.1f}%")

        if age and age < 16:
            print(T["child_warning"].format(age=int(age)))

        rows.append({
            "Age": age_text,
            "Gender": r.get("dominant_gender", ""),
            "Race": r.get("dominant_race", ""),
            "Emotion": r.get("dominant_emotion", "")
        })

    df = pd.DataFrame(rows)
    print(T["summary"])
    print(df.to_string(index=False))

    img = cv2.imread(tmp_path)
    if img is None:
        print(T["image_fail"])
        sys.exit(1)

    h_img, w_img = img.shape[:2]
    colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,128,0),(128,0,128)]

    for idx, r in enumerate(results):
        region = r.get("region", {})
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))

        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        color = colors[idx % len(colors)]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        age = r.get("age", None)
        age_text = f"{int(age)-3}-{int(age)+3}" if age is not None else "?"

        label = f"#{idx+1}: {r.get('dominant_gender','')}, {age_text}, {r.get('dominant_race','')}, {r.get('dominant_emotion','')}"
        cv2.putText(img, label, (x, y-10 if y-10>10 else y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(T["plot_title"].format(det=selected_detector))
    plt.show()

except Exception as e:
    traceback.print_exc()
finally:
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass

print(T["finished"])
