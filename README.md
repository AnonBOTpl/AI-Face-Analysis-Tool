<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" />
  <img src="https://img.shields.io/badge/DeepFace-Enabled-green" />
  <img src="https://img.shields.io/badge/OpenCV-Analysis-orange" />
  <img src="https://img.shields.io/badge/GUI-Tkinter-lightgrey" />
</p>

# AI Face Analysis Tool
*A Python GUI app for face analysis (age, gender, race, emotions) using DeepFace and OpenCV.*

# Face Analysis with DeepFace

This project is a simple desktop application for analyzing faces in images using [DeepFace](https://github.com/serengil/deepface).  
It detects **age, gender, race, and emotions** with a graphical interface built in **Tkinter**.

## 🖥️ Requirements
- Python **3.8 – 3.11** (recommended: Python 3.10)  
- Internet connection (for installing dependencies)  

## 🚀 Quick Start (Easy Way)
If you just want to run the program without worrying about dependencies:
```bash
python setup_and_run.py
```
This script will:
1. Check if required libraries are installed.
2. Install missing packages automatically.
3. Run the face analyzer.

## 📦 Manual Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AnonBOTpl/AI-Face-Analysis-Tool
   cd face-analyzer
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

   or manually:
   ```bash
   pip install deepface opencv-python matplotlib pandas
   ```

3. Make sure **Tkinter** is installed:
   - **Windows/Mac** → included by default  
   - **Linux (Ubuntu/Debian):**
     ```bash
     sudo apt-get install python3-tk
     ```

## ▶️ Usage
```bash
python analyze_face.py
```

1. Choose the **face detector**.  
2. Select an image (`.jpg`, `.jpeg`, `.png`, `.bmp`).  
3. Results will be shown in the terminal and in a window with the processed image.

   About: This project is intended for educational/demo purposes. The predictions may not always be accurate, especially in challenging lighting conditions or with small/low-resolution images.
   
