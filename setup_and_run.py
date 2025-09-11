import subprocess
import sys

# Lista wymaganych paczek
REQUIRED = [
    "deepface",
    "opencv-python",
    "matplotlib",
    "pandas"
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("🔍 Checking required packages...")
    for pkg in REQUIRED:
        try:
            __import__(pkg.split("-")[0])  # np. 'opencv-python' -> 'opencv'
            print(f"✅ {pkg} already installed")
        except ImportError:
            print(f"⬇️ Installing {pkg}...")
            install(pkg)

    print("🚀 Running face analyzer...")
    subprocess.check_call([sys.executable, "analyze_face.py"])

if __name__ == "__main__":
    main()
