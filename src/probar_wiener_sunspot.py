import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from skimage.metrics import structural_similarity as ssim

# Ajustar paths
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import deconvolucion as decon

sns.set_theme()

def calcular_metricas(original, procesado):
    rmse_val = np.sqrt(np.mean((original - procesado)**2))
    rango_datos = original.max() - original.min()
    
    if original.ndim == 3:
        ssims = []
        for i in range(original.shape[0]):
            ssims.append(ssim(original[i], procesado[i], data_range=rango_datos))
        ssim_val = np.mean(ssims)
    else:
        ssim_val = ssim(original, procesado, data_range=rango_datos)

    return rmse_val, ssim_val

def simular_ruido_telescopio(intensidad, snr=1000):
    I_continuo = np.max(intensidad)
    sigma_ruido = I_continuo / snr
    ruido_I = np.random.normal(loc=0.0, scale=sigma_ruido, size=intensidad.shape)
    return intensidad + ruido_I

def main():
    # 1. Cargar datos usando ruta absoluta segura
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datos_path = os.path.join(base_dir, '..', 'data', 'datos_sunspot.npz')
    print(f"Cargando datos desde {datos_path}...")
    datos_cargados = np.load(datos_path)
    
    data = datos_cargados['stokes']
    intensidad = data[:, 0, :, :]
    
    # 2. Generar PSF de Airy
    psf = decon.generar_psf_airy(tamano_matriz=31, radio_piz=3)
    
    # 3. Degradar la imagen (Borroso + Ruido)
    print("Aplicando degradación (PSF + Ruido con SNR=1000)...")
    intenBorrosa = decon.convolucion3D(intensidad, psf)
    intenBorrosaRuido = simular_ruido_telescopio(intenBorrosa, snr=1000)
    
    # 4. Deconvoluciones
    print("Aplicando Deconvolución Richardson-Lucy...")
    inten_rl = decon.deconvolucion3D(intenBorrosaRuido, psf, metodo='rl', pasos=20)
    
    print("Aplicando Deconvolución Fourier...")
    inten_fourier = decon.deconvolucion3D(intenBorrosaRuido, psf, metodo='fourier')
    
    print("Aplicando Deconvolución Wiener (corregido)...")
    inten_wiener = decon.deconvolucion3D(intenBorrosaRuido, psf, metodo='wiener')
    
    # 5. Calcular métricas
    metodos = ['Borroso + Ruido', 'Richardson-Lucy', 'Fourier', 'Wiener (Corregido)']
    imagenes = [intenBorrosaRuido, inten_rl, inten_fourier, inten_wiener]
    
    print("\n--- RESULTADOS DE LAS METRICAS (SUNSPOT INTENSIDAD) ---")
    for nombre, img in zip(metodos, imagenes):
        rmse_val, ssim_val = calcular_metricas(intensidad, img)
        print(f"{nombre:20s} -> RMSE: {rmse_val:.6f}, SSIM: {ssim_val:.6f}")
    print("-------------------------------------------------------\n")
    
    # 6. Generar gráficos comparativos (slice central)
    idx_z = intensidad.shape[0] // 2
    os.makedirs(os.path.join(base_dir, '..', 'output'), exist_ok=True)
    
    # Graficar la comparación de los 3 métodos en el slice central
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    cmap_rocket = sns.color_palette("rocket", as_cmap=True)
    
    # Original
    axes[0, 0].imshow(intensidad[idx_z], cmap=cmap_rocket)
    axes[0, 0].set_title("Original (Ground Truth)")
    axes[0, 0].axis('off')
    
    # Degradada
    rmse_b, ssim_b = calcular_metricas(intensidad, intenBorrosaRuido)
    axes[0, 1].imshow(intenBorrosaRuido[idx_z], cmap=cmap_rocket)
    axes[0, 1].set_title(f"Borroso + Ruido\nRMSE: {rmse_b:.5f} | SSIM: {ssim_b:.5f}")
    axes[0, 1].axis('off')
    
    # Vacío o PSF
    axes[0, 2].imshow(psf, cmap='mako')
    axes[0, 2].set_title("PSF (Airy)")
    axes[0, 2].axis('off')
    
    # RL
    rmse_rl, ssim_rl = calcular_metricas(intensidad, inten_rl)
    axes[1, 0].imshow(inten_rl[idx_z], cmap=cmap_rocket)
    axes[1, 0].set_title(f"Richardson-Lucy\nRMSE: {rmse_rl:.5f} | SSIM: {ssim_rl:.5f}")
    axes[1, 0].axis('off')
    
    # Fourier
    rmse_f, ssim_f = calcular_metricas(intensidad, inten_fourier)
    axes[1, 1].imshow(inten_fourier[idx_z], cmap=cmap_rocket)
    axes[1, 1].set_title(f"Deconv Fourier\nRMSE: {rmse_f:.5f} | SSIM: {ssim_f:.5f}")
    axes[1, 1].axis('off')
    
    # Wiener
    rmse_w, ssim_w = calcular_metricas(intensidad, inten_wiener)
    axes[1, 2].imshow(inten_wiener[idx_z], cmap=cmap_rocket)
    axes[1, 2].set_title(f"Wiener Corregido\nRMSE: {rmse_w:.5f} | SSIM: {ssim_w:.5f}")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    out_img_path = os.path.join(base_dir, '..', 'output', 'comparacion_wiener_sunspot.png')
    plt.savefig(out_img_path)
    print(f"Gráfica de resultados guardada en: {out_img_path}")

if __name__ == "__main__":
    main()
