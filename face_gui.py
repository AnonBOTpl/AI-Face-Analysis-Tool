# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from deepface import DeepFace
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tempfile
import os
import uuid
import locale
import traceback
import threading
import numpy as np

# --- Translacje ---
translations = {
    "pl": {
        "choose_detector_title": "Wybór detektora DeepFace",
        "choose_detector": "Wybierz detektor twarzy",
        "detector_label": "Detektor twarzy:",
        "ok": "OK",
        "cancel": "Anuluj",
        "detectors_info": """Detektory:\n• opencv - szybki, podstawowy (dobry start)\n• ssd - średni, lepszy od opencv\n• mtcnn - wolniejszy, dokładniejszy\n• retinaface - najlepszy, najwolniejszy (polecany dla dzieci)\n\nProgram używa domyślnych modeli DeepFace dla wieku, płci, rasy i emocji.""",
        "file_dialog_title": "Wybierz zdjęcie do analizy",
        "analyze": "Analizuj",
        "no_file": "Nie wybrano pliku.",
        "no_faces": "Nie wykryto twarzy na obrazie.",
        "faces_detected": "Wykryto {n} twarzy",
        "face": "Twarz #{i}",
        "age": "Wiek: {age} lat (przedział: {range})",
        "gender": "Płeć - Kobieta: {w:.1f}%, Mężczyzna: {m:.1f}%",
        "race": "Rasa (top 3):",
        "emotion": "Emocje (top 3):",
        "child_warning": "UWAGA: Wykryty wiek {age} lat - wyniki mogą być mniej dokładne dla dzieci!",
        "summary": "Podsumowanie wyników:",
        "image_fail": "Nie udało się wczytać obrazu.",
        "plot_title": "Analiza twarzy z DeepFace (detektor: {det})",
        "select_file": "Wybierz plik",
        "selected_file": "Wybrano plik: {file}",
        "select_detector": "Wybierz detektor",
        "finished": "Program zakończony"
    },
    "en": {
        "choose_detector_title": "DeepFace Detector Selection",
        "choose_detector": "Choose face detector",
        "detector_label": "Face detector:",
        "ok": "OK",
        "cancel": "Cancel",
        "detectors_info": """Detectors:\n• opencv - fast, basic (good start)\n• ssd - medium, better than opencv\n• mtcnn - slower, more accurate\n• retinaface - best, slowest (recommended for children)\n\nThe program uses default DeepFace models for age, gender, race and emotions.""",
        "file_dialog_title": "Select a photo for analysis",
        "analyze": "Analyze",
        "no_file": "No file selected.",
        "no_faces": "No faces detected in image.",
        "faces_detected": "{n} face(s) detected",
        "face": "Face #{i}",
        "age": "Age: {age} years (range: {range})",
        "gender": "Gender - Woman: {w:.1f}%, Man: {m:.1f}%",
        "race": "Race (top 3):",
        "emotion": "Emotions (top 3):",
        "child_warning": "WARNING: Detected age {age} years - results may be less accurate for children!",
        "summary": "Results summary:",
        "image_fail": "Failed to load image.",
        "plot_title": "DeepFace Face Analysis (detector: {det})",
        "select_file": "Select file",
        "selected_file": "File selected: {file}",
        "select_detector": "Select detector",
        "finished": "Program finished"
    }
}

lang, _ = locale.getdefaultlocale()
if lang and lang.startswith("pl"):
    LANG = "pl"
else:
    LANG = "en"
T = translations[LANG]

class FaceApp:
    def __init__(self, master):
        self.master = master
        master.title("DeepFace Face Detector")
        master.geometry("1350x1050")
        master.resizable(True, True)

        self.detector_var = tk.StringVar(value='opencv')
        self.img_path = None
        self.img_panel = None
        self.result_img = None

        # Detector selection
        detector_frame = tk.LabelFrame(master, text=T["choose_detector_title"], padx=10, pady=10)
        detector_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(detector_frame, text=T["detector_label"], font=("Arial", 12)).pack(side="left")
        detector_combo = ttk.Combobox(detector_frame, textvariable=self.detector_var,
                                      values=['opencv', 'ssd', 'mtcnn', 'retinaface'],
                                      state="readonly", font=("Arial", 11))
        detector_combo.pack(side="left", padx=10)
        tk.Label(detector_frame, text=T["detectors_info"], justify="left", font=("Arial", 10), wraplength=600).pack(side="left", padx=20)

        # File selection
        file_frame = tk.Frame(master)
        file_frame.pack(fill="x", padx=10, pady=5)
        self.file_label = tk.Label(file_frame, text=T["no_file"], font=("Arial", 11))
        self.file_label.pack(side="left")
        file_btn = tk.Button(file_frame, text=T["select_file"], command=self.select_file, font=("Arial", 11))
        file_btn.pack(side="left", padx=10)

        # Analyze button
        analyze_btn = tk.Button(master, text=T["analyze"], command=self.analyze, font=("Arial", 13, "bold"), bg="#4CAF50", fg="white", padx=30)
        analyze_btn.pack(pady=10)

        # Save image button
        save_btn = tk.Button(master, text="Zapisz obraz z ramkami", command=self.save_image, font=("Arial", 11), bg="#2196F3", fg="white")
        save_btn.pack(pady=2)

        # Progress section
        self.progress_label = tk.Label(master, text="", font=("Arial", 12))
        self.progress_label.pack(pady=2)
        self.progressbar = ttk.Progressbar(master, mode="indeterminate")
        self.progressbar.pack(fill="x", padx=20, pady=2)
        self.progressbar.pack_forget()

        # Results frame
        self.results_frame = tk.LabelFrame(master, text=T["summary"], padx=10, pady=10)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Table for results
        self.tree = ttk.Treeview(self.results_frame, columns=("Nr", "Age", "Gender", "Race", "Emotion"), show="headings")
        for col, label in zip(("Nr", "Age", "Gender", "Race", "Emotion"), ["Nr" if LANG=="pl" else "Face #", "Age", "Gender", "Race", "Emotion"]):
            self.tree.heading(col, text=label)
            self.tree.column(col, width=80 if col=="Nr" else 120)
        self.tree.pack(side="left", fill="y", padx=10)

        # Image panel
        self.img_panel = tk.Label(self.results_frame)
        self.img_panel.pack(side="left", fill="both", expand=True, padx=10)

        # Akcje DeepFace - checkboxy
        actions_frame = tk.LabelFrame(master, text="Wybierz akcje do analizy", padx=10, pady=5)
        actions_frame.pack(fill="x", padx=10, pady=5)
        self.action_vars = {}
        for action, label in zip(['age', 'gender', 'race', 'emotion'], [T['age'].split(':')[0], T['gender'].split('-')[0], T['race'].split('(')[0], T['emotion'].split('(')[0]]):
            var = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(actions_frame, text=label, variable=var, font=("Arial", 11))
            cb.pack(side="left", padx=10)
            self.action_vars[action] = var

    def select_file(self):
        path = filedialog.askopenfilename(title=T["file_dialog_title"], filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img_path = path
            self.file_label.config(text=T["selected_file"].format(file=os.path.basename(path)))
        else:
            self.img_path = None
            self.file_label.config(text=T["no_file"])

    def analyze(self):
        if not self.img_path:
            messagebox.showerror("Error", T["no_file"])
            return
        actions = [a for a, v in self.action_vars.items() if v.get()]
        if not actions:
            messagebox.showerror("Error", "Wybierz przynajmniej jedną akcję do analizy.")
            return
        self.progress_label.config(text=T["analyzing"] if "analyzing" in T else "Analizowanie...")
        self.progressbar.pack(fill="x", padx=20, pady=2)
        self.progressbar.start()
        self.master.update_idletasks()
        detector_backend = self.detector_var.get()
        threading.Thread(target=self._analyze_worker, args=(detector_backend, actions), daemon=True).start()

    def save_image(self):
        if not hasattr(self, 'last_img_with_boxes') or self.last_img_with_boxes is None:
            messagebox.showerror("Błąd", "Brak obrazu do zapisania. Najpierw wykonaj analizę.")
            return
        filetypes = [("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")]
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=filetypes, title="Zapisz obraz z ramkami")
        if save_path:
            try:
                self.last_img_with_boxes.save(save_path)
                messagebox.showinfo("Sukces", f"Obraz zapisany: {save_path}")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się zapisać obrazu: {e}")

    def _analyze_worker(self, detector_backend, actions):
        ext = os.path.splitext(self.img_path)[1]

        # Use secure temporary file creation
        fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="tmp_img_")
        os.close(fd)

        try:
            # More efficient direct copy
            import shutil
            shutil.copy2(self.img_path, tmp_path)
        except Exception as e:
            # Only show generic error to avoid information exposure
            self.master.after(0, lambda: messagebox.showerror("Error", f"{T['image_fail']}"))
            self._hide_progress()
            return

        try:
            results = DeepFace.analyze(
                img_path=tmp_path,
                actions=actions,
                detector_backend=detector_backend,
                enforce_detection=False
            )
            if isinstance(results, dict):
                results = [results]
            if not results:
                self.master.after(0, lambda: messagebox.showinfo("Info", T["no_faces"]))
                self._hide_progress()
                return
            def update_results():
                self.tree.delete(*self.tree.get_children())
                try:
                    pil_img = Image.open(self.img_path).convert("RGB")
                except Exception:
                    messagebox.showerror("Error", T["image_fail"])
                    self._hide_progress()
                    return
                img = np.array(pil_img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h_img, w_img = img.shape[:2]
                colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,128,0),(128,0,128)]
                face_idx = 1
                for r in results:
                    region = r.get("region", {})
                    w = int(region.get("w", 0))
                    h = int(region.get("h", 0))
                    if w < 40 or h < 40:
                        continue  # pomiń bardzo małe twarze
                    age = r.get("age", None)
                    age_text = f"{int(age)-3}-{int(age)+3}" if age is not None else "?"
                    race = r.get("dominant_race", "")
                    emotion = r.get("dominant_emotion", "")
                    self.tree.insert("", "end", values=(face_idx, age_text, r.get("dominant_gender", ""), race, emotion))
                    x = int(region.get("x", 0))
                    y = int(region.get("y", 0))
                    color = colors[(face_idx-1) % len(colors)]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    label = f"#{face_idx}: {r.get('dominant_gender','')}, {age_text}, {race}, {emotion}"
                    cv2.putText(img, label, (x, y-10 if y-10>10 else y+h+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                    face_idx += 1
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                self.last_img_with_boxes = pil_img.copy()  # do zapisu
                orig_w, orig_h = pil_img.size
                max_w = self.img_panel.winfo_width() or 900
                max_h = self.img_panel.winfo_height() or 700
                ratio = min(max_w / orig_w, max_h / orig_h)
                new_w = int(orig_w * ratio)
                new_h = int(orig_h * ratio)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                self.result_img = ImageTk.PhotoImage(pil_img)
                self.img_panel.config(image=self.result_img)
                self.img_panel.image = self.result_img
                self._hide_progress()
            self.master.after(0, update_results)
        except Exception:
            # Sanitize exception handling to not expose full stack traces
            self.master.after(0, lambda: messagebox.showerror("Error", "Wystąpił błąd podczas analizy obrazu."))
            self._hide_progress()
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _hide_progress(self):
        self.progressbar.stop()
        self.progressbar.pack_forget()
        self.progress_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
