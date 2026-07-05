import re
import os

files_to_modify = [
    'src/probarSis.py',
    'src/experimento_reales.py'
]

for filepath in files_to_modify:
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        if '.set_title(' in line or 'plt.title(' in line or '.suptitle(' in line:
            new_lines.append('# ' + line)
        else:
            new_lines.append(line)
            
    with open(filepath, 'w') as f:
        f.writelines(new_lines)

print("Títulos comentados con éxito.")
