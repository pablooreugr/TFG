import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
import deconvolucion as decon

# Setup
sns.set_theme(style="white")
os.makedirs('../output/experimentos', exist_ok=True)

def calcular_metricas(original, procesado):
    # SSIM
    rango_datos = original.max() - original.min()
    if original.ndim == 3:
        ssims = []
        for i in range(original.shape[0]):
            ssims.append(ssim(original[i], procesado[i], data_range=rango_datos))
        ssim_val = np.mean(ssims)
    else:
        ssim_val = ssim(original, procesado, data_range=rango_datos)
        
    # SNR (Signal-to-Noise Ratio formula: 10 * log10( sum(signal^2) / sum(noise^2) )
    ruido = original - procesado
    snr_val = 10 * np.log10(np.sum(original**2) / (np.sum(ruido**2) + 1e-12))
    
    return snr_val, ssim_val

def add_zoomed_inset(ax, image, cmap, x1, x2, y1, y2):
    # Crear inset en la esquina superior derecha
    axins = ax.inset_axes([0.55, 0.55, 0.4, 0.4]) 
    axins.imshow(image, cmap=cmap)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y2, y1)
    axins.set_xticks([])
    axins.set_yticks([])
    ax.indicate_inset_zoom(axins, edgecolor="white", linewidth=2, alpha=0.8)
    for spine in axins.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(1.5)

print("Iniciando experimento 1...")

# 1. Cargar datos
print("Cargando datos...")
datos_cargados = np.load('../data/datos_sunspot.npz')
data = datos_cargados['stokes']
intensidad = data[:, 0, :, :]

# 2. Generar PSF
print("Generando PSF...")
psf = decon.generar_psf_airy(tamano_matriz=31, radio_piz=3)

# 3. Degradar la imagen (Borroso + 2% Ruido)
print("Aplicando degradación...")
intenBorrosa = decon.convolucion3D(intensidad, psf)
I_continuo = np.max(intensidad)
sigma_ruido = 0.02 * I_continuo # 2% de ruido
ruido = np.random.normal(0, sigma_ruido, intensidad.shape)
intenBorrosaRuido = intenBorrosa + ruido

# 4. Deconvoluciones
print("Aplicando Fourier...")
inten_fourier = decon.deconvolucion3D(intenBorrosaRuido, psf, metodo='fourier')
print("Aplicando Wiener...")
inten_wiener = decon.deconvolucion3D(intenBorrosaRuido, psf, metodo='wiener')
print("Aplicando Richardson-Lucy...")
inten_rl = decon.deconvolucion3D(intenBorrosaRuido, psf, metodo='rl', pasos=10)

# 5. Calcular Métricas
metodos = ['Borroso + Ruido', 'Fourier', 'Wiener', 'Richardson-Lucy']
imagenes = [intenBorrosaRuido, inten_fourier, inten_wiener, inten_rl]

snrs = []
ssims = []

print("\n--- Resultados ---")
for nombre, img in zip(metodos, imagenes):
    snr_val, ssim_val = calcular_metricas(intensidad, img)
    snrs.append(snr_val)
    ssims.append(ssim_val)
    print(f"{nombre} -> SNR: {snr_val:.2f} dB, SSIM: {ssim_val:.4f}")

# 6. Visualizaciones
idx_z = intensidad.shape[0] // 2

# Colormaps
cmap_intensidad = sns.color_palette("rocket", as_cmap=True)
cmap_psf = sns.color_palette("mako", as_cmap=True)

print("Generando gráficas...")

# 6.1 Original y Borrosa
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6))
im1 = axes1[0].imshow(intensidad[idx_z], cmap=cmap_intensidad)
axes1[0].set_title("Intensidad Original (Sintética)", fontsize=14, fontweight='bold')
axes1[0].axis('off')
fig1.colorbar(im1, ax=axes1[0], fraction=0.046, pad=0.04)

im2 = axes1[1].imshow(intenBorrosaRuido[idx_z], cmap=cmap_intensidad)
axes1[1].set_title("Degradada (PSF + 2% Ruido)", fontsize=14, fontweight='bold')
axes1[1].axis('off')
fig1.colorbar(im2, ax=axes1[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('../output/experimentos/1_original_borrosa.png', dpi=300, bbox_inches='tight')

# 6.2 PSF
fig2, ax2 = plt.subplots(figsize=(6, 6))
im_psf = ax2.imshow(psf, cmap=cmap_psf)
ax2.set_title("PSF de Airy", fontsize=14, fontweight='bold')
ax2.axis('off')
fig2.colorbar(im_psf, ax=ax2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('../output/experimentos/2_psf.png', dpi=300, bbox_inches='tight')

# 6.3 Deconvoluciones
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
metodos_plot = ['Fourier', 'Wiener', 'Richardson-Lucy']
imgs_plot = [inten_fourier[idx_z], inten_wiener[idx_z], inten_rl[idx_z]]

for ax, nom, img in zip(axes3, metodos_plot, imgs_plot):
    im = ax.imshow(img, cmap=cmap_intensidad)
    ax.set_title(f"Deconvolución: {nom}", fontsize=14, fontweight='bold')
    ax.axis('off')
    fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('../output/experimentos/3_deconvoluciones.png', dpi=300, bbox_inches='tight')

# 6.4 Gráfico de Barras - SNR
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.barplot(x=metodos, y=snrs, ax=ax4, hue=metodos, palette="viridis", legend=False)
ax4.set_title("Comparación de SNR", fontsize=16, fontweight='bold')
ax4.set_ylabel("SNR (dB)", fontsize=12)
for i, v in enumerate(snrs):
    ax4.text(i, v + 0.5, f"{v:.2f}", color='black', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../output/experimentos/4_barras_snr.png', dpi=300, bbox_inches='tight')

# 6.5 Gráfico de Barras - SSIM
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.barplot(x=metodos, y=ssims, ax=ax5, hue=metodos, palette="magma", legend=False)
ax5.set_title("Comparación de SSIM", fontsize=16, fontweight='bold')
ax5.set_ylabel("SSIM", fontsize=12)
for i, v in enumerate(ssims):
    ax5.text(i, v + 0.01, f"{v:.4f}", color='black', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../output/experimentos/5_barras_ssim.png', dpi=300, bbox_inches='tight')

# 6.6 Zoom Insets
# Definimos la región del zoom
x1, x2, y1, y2 = 130, 200, 130, 200

# Plot 6: Original y Borrosa (Zoomed)
fig6, axes6 = plt.subplots(1, 2, figsize=(12, 6))
im6_1 = axes6[0].imshow(intensidad[idx_z], cmap=cmap_intensidad)
axes6[0].set_title("Intensidad Original (Inset)", fontsize=14, fontweight='bold')
axes6[0].axis('off')
add_zoomed_inset(axes6[0], intensidad[idx_z], cmap_intensidad, x1, x2, y1, y2)

im6_2 = axes6[1].imshow(intenBorrosaRuido[idx_z], cmap=cmap_intensidad)
axes6[1].set_title("Degradada (Inset)", fontsize=14, fontweight='bold')
axes6[1].axis('off')
add_zoomed_inset(axes6[1], intenBorrosaRuido[idx_z], cmap_intensidad, x1, x2, y1, y2)
plt.tight_layout()
plt.savefig('../output/experimentos/6_original_borrosa_zoom.png', dpi=300, bbox_inches='tight')

# Plot 7: Deconvoluciones (Zoomed)
fig7, axes7 = plt.subplots(1, 3, figsize=(18, 6))
for ax, nom, img in zip(axes7, metodos_plot, imgs_plot):
    im = ax.imshow(img, cmap=cmap_intensidad)
    ax.set_title(f"Deconvolución {nom} (Inset)", fontsize=14, fontweight='bold')
    ax.axis('off')
    add_zoomed_inset(ax, img, cmap_intensidad, x1, x2, y1, y2)

plt.tight_layout()
plt.savefig('../output/experimentos/7_deconvoluciones_zoom.png', dpi=300, bbox_inches='tight')

print("¡Experimento completado y guardado en '../output/experimentos/'!")
