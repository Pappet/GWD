#!/usr/bin/env python3
import fileinput
import sys

files = ['dataset/train_cnn.py', 'evaluate_model_with_roc.py', 
         'compare_models_roc.py', 'optimize_threshold.py', 'evaluate_classic.py']

for fname in files:
    try:
        with open(fname, 'r') as f:
            lines = f.readlines()
        
        # Backend hinzufügen nach matplotlib import
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            if 'import matplotlib.pyplot as plt' in line:
                if i > 0 and 'matplotlib.use' not in lines[i-1]:
                    new_lines.insert(-1, "import matplotlib\nmatplotlib.use('Agg')\n")
        
        # plt.show() durch savefig ersetzen
        content = ''.join(new_lines)
        content = content.replace('plt.show()', 
                                 "plt.savefig('plot.png', dpi=150)\nprint('📊 Plot: plot.png')\nplt.close()")
        
        with open(fname, 'w') as f:
            f.write(content)
        print(f"✓ {fname}")
    except FileNotFoundError:
        print(f"⊘ {fname} nicht gefunden")
