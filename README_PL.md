# Analiza twarzy z DeepFace

Prosta aplikacja desktopowa do analizy twarzy na zdjęciach z użyciem biblioteki [DeepFace](https://github.com/serengil/deepface).  
Pozwala wykrywać **wiek, płeć, rasę i emocje** przez interfejs graficzny w **Tkinterze**.

## 🖥️ Wymagania
- Python **3.8 – 3.11** (polecana wersja: 3.10)  
- Dostęp do internetu (do pobrania bibliotek)  

## 🚀 Szybki start (najłatwiejsza metoda)
Jeżeli chcesz po prostu uruchomić program:
```bash
python setup_and_run.py
```
Skrypt:
1. Sprawdzi, czy wszystkie biblioteki są dostępne.  
2. Automatycznie doinstaluje brakujące.  
3. Uruchomi program analizy twarzy.  

## 📦 Instalacja ręczna
1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/yourusername/face-analyzer.git
   cd face-analyzer
   ```

2. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt
   ```

   lub ręcznie:
   ```bash
   pip install deepface opencv-python matplotlib pandas
   ```

3. Upewnij się, że masz **Tkinter**:
   - **Windows/Mac** → zazwyczaj jest wbudowany  
   - **Linux (Ubuntu/Debian):**
     ```bash
     sudo apt-get install python3-tk
     ```

## ▶️ Użycie
```bash
python analyze_face.py
```

1. Wybierz **detektor twarzy**.  
2. Wskaż plik ze zdjęciem (`.jpg`, `.jpeg`, `.png`, `.bmp`).  
3. Wyniki pojawią się w konsoli oraz na obrazie w oknie.  
