#!/usr/bin/env python3
"""Quick launcher for AI Face Analysis Tool v2."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "src.main"])
