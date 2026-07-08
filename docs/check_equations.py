import glob
import re

files = glob.glob('/home/pabloore/conjuntoV/universidad/TFG/docs/**/*.tex', recursive=True)

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # split by lines
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if r'\end{equation}' in line or r'\]' in line or r'\end{align}' in line:
            # check the next non-empty line
            for j in range(i+1, len(lines)):
                next_line = lines[j].strip()
                if next_line:
                    # if starts with lowercase letter or 'donde'
                    if next_line and next_line[0].islower():
                        print(f"{filepath}:{i+1}")
                        print(f"  {lines[i-1].strip()}")
                        print(f"  {line.strip()}")
                        print(f"  {next_line}")
                        print()
                    break
