import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
import os

# Asegurar que el directorio de salida existe
os.makedirs('docs/img', exist_ok=True)

# Cargar la imagen
img_path = 'data/IMG_20221223_132853.jpg'
img = Image.open(img_path)

# Recortar al centro para hacerla cuadrada y escalar para convolución rápida
width, height = img.size
min_dim = min(width, height)
left = (width - min_dim)/2
top = (height - min_dim)/2
right = (width + min_dim)/2
bottom = (height + min_dim)/2
img = img.crop((left, top, right, bottom))
img = img.resize((600, 600), Image.Resampling.LANCZOS)

img_arr = np.array(img) / 255.0

# Crear una PSF (Gaussiana desenfocada)
size = 41
sigma = 8.0
x = np.arange(-size//2 + 1., size//2 + 1.)
y = np.arange(-size//2 + 1., size//2 + 1.)
x, y = np.meshgrid(x, y)
psf = np.exp(-(x**2 + y**2) / (2. * sigma**2))
psf = psf / np.sum(psf)

# Convolucionar la imagen original con la PSF
degraded = np.zeros_like(img_arr)
if len(img_arr.shape) == 3:
    for i in range(3):
        degraded[:,:,i] = convolve2d(img_arr[:,:,i], psf, mode='same', boundary='symm')
else:
    degraded = convolve2d(img_arr, psf, mode='same', boundary='symm')

degraded = np.clip(degraded, 0, 1)

# Crear la figura didáctica
fig = plt.figure(figsize=(15, 5))

# 1. Imagen Original
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(img_arr)
ax1.axis('off')

# 2. Símbolo de Convolución
fig.text(0.34, 0.5, r'$\ast$', fontsize=50, ha='center', va='center')

# 3. PSF
ax2 = fig.add_subplot(1, 3, 2)
im2 = ax2.imshow(psf, cmap='inferno')
ax2.axis('off')

# 4. Símbolo de Igualdad
fig.text(0.66, 0.5, r'$=$', fontsize=50, ha='center', va='center')

# 5. Imagen Degradada
ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(degraded)
ax3.axis('off')

plt.tight_layout()
# Ajustar espacio para que los símbolos no se solapen
plt.subplots_adjust(wspace=0.4)

out_path = 'docs/img/ejemplo_degradacion.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"¡Imagen generada con éxito en {out_path}!")
