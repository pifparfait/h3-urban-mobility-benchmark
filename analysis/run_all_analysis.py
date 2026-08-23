#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
scripts = [
    ROOT / "analysis" / "01_architecture_scalability.py",
    ROOT / "analysis" / "02_urban_fidelity_figures.py",
    ROOT / "analysis" / "03_spatial_heatmaps.py",
]
for script in scripts:
    print(f"\n=== {script.name} ===")
    subprocess.run([sys.executable, str(script)], check=True)
