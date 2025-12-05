#!/usr/bin/env python3
import sys

packages = {
    'numpy': 'NumPy',
    'pandas': 'Pandas', 
    'scipy': 'SciPy',
    'tensorflow': 'TensorFlow',
    'sklearn': 'Scikit-learn',
    'matplotlib': 'Matplotlib',
    'gwpy': 'GWpy',
    'tabulate': 'Tabulate'
}

print("="*50)
print("DEPENDENCY CHECK")
print("="*50)

for module, name in packages.items():
    try:
        __import__(module)
        print(f"✓ {name:15s} OK")
    except ImportError as e:
        print(f"✗ {name:15s} FEHLT")

# PyCBC separat (optional)
print("-"*50)
try:
    from pycbc.waveform import get_td_waveform
    print("✓ PyCBC          OK (Profi-Wellenformen verfügbar)")
except ImportError:
    print("⚠ PyCBC          Fallback (Newton-Physik wird verwendet)")

# GWD Core Test
print("-"*50)
try:
    from gwd_core.waveforms import generate_astrophysical_chirp
    print("✓ GWD Core       OK")
except ImportError as e:
    print(f"✗ GWD Core       FEHLER: {e}")

print("="*50)
