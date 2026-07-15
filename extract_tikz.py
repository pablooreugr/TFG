import os
import re
import subprocess
import shutil

TFG_DIR = "/home/pabloore/conjuntoV/universidad/TFG"
CONTENT_DIR = os.path.join(TFG_DIR, "docs", "content")
EXPORT_DIR = os.path.join(TFG_DIR, "docs", "tikz_svg_export")

os.makedirs(EXPORT_DIR, exist_ok=True)

latex_template = r"""\documentclass[tikz,11pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathpazo}
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{eurosym}
\usepackage{bbding}
\usepackage{pifont}
\usepackage{latexsym}
\usepackage[spanish,es-nodecimaldot,es-tabla,es-lcroman,es-nosectiondot]{babel}
\usetikzlibrary{arrows.meta,babel,positioning,shapes.geometric,calc}
\usepackage{color}
\begin{document}
%s
\end{document}
"""

def extract_tikz_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract blocks
    # Since tikzpictures can be nested or have line breaks, we'll use a simple regex if not nested,
    # or a better matching method. Regular tikz pictures usually aren't nested.
    pattern = re.compile(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', re.DOTALL)
    matches = pattern.findall(content)
    return matches

def main():
    tex_files = [f for f in os.listdir(CONTENT_DIR) if f.endswith('.tex')]
    
    count = 1
    for tf in tex_files:
        filepath = os.path.join(CONTENT_DIR, tf)
        tikz_blocks = extract_tikz_from_file(filepath)
        
        for block in tikz_blocks:
            base_name = f"{tf[:-4]}_fig{count}"
            tex_path = os.path.join(EXPORT_DIR, f"{base_name}.tex")
            
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(latex_template % block)
                
            print(f"Compiling {base_name}.tex...")
            # Run pdflatex
            res = subprocess.run(['pdflatex', '-interaction=nonstopmode', f"{base_name}.tex"], cwd=EXPORT_DIR, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"Error compiling {base_name}.tex")
                print(res.stdout)
            else:
                # Convert to svg
                pdf_path = os.path.join(EXPORT_DIR, f"{base_name}.pdf")
                svg_path = os.path.join(EXPORT_DIR, f"{base_name}.svg")
                subprocess.run(['pdftocairo', '-svg', pdf_path, svg_path], cwd=EXPORT_DIR)
                print(f"Created {svg_path}")
                
            count += 1

if __name__ == "__main__":
    main()
