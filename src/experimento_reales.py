import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
import os
import deconvolucion as decon
import magnetismo as mag

def main():
    print("Iniciando experimento con datos reales (Telescopio) - Cubo Completo")
    os.makedirs('output/exper', exist_ok=True)
    
    # 1. Cargar PSF y Normalizar
    print("Cargando y normalizando PSF (Recortada a 61x61)...")
    psf_bruta = np.load('data/PSF_517_1600_x_1600_px.npy')
    
    # Recortar la PSF alrededor de su centro (suelen ser de tamaño impar para no desplazar la imagen 0.5px)
    cy, cx = psf_bruta.shape[0] // 2, psf_bruta.shape[1] // 2
    radio = 30 # Tamaño total 61x61 (30 izquierda, centro, 30 derecha)
    psf_recortada = psf_bruta[cy-radio:cy+radio+1, cx-radio:cx+radio+1]
    
    psf = psf_recortada / np.sum(psf_recortada)  # Normalización fotométrica sobre el recorte
    
    # 2. Cargar datos FITS
    print("Cargando cubo FITS...")
    fits_path = 'data/01_QSUN_TM_00_Mg1_10_10072024T131604_LV_1.0_v0.4.fits'
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        # data shape: (10, 4, 1600, 1600) = (Lambda, Stokes, Y, X)
        intensidad = data[:, 0, :, :] # Stokes I (10, 1600, 1600)
        compV = data[:, 3, :, :]      # Stokes V (10, 1600, 1600)
    print(f"Datos cargados. Forma I: {intensidad.shape}, Forma V: {compV.shape}")
    
    # Vector de lambdas: la línea espectral observada es Mg I b1 (5170 Angstroms).
    # Generamos un vector centrado en 5170 con un paso realista (ej. 50 mA = 0.05 A)
    lambdas = 5170.0 + (np.arange(10) - 4.5) * 0.05 
    
    # =========================================================================
    # EXPERIMENTO 4.1: Intensidad (Stokes I)
    # =========================================================================
    print("\n--- Experimento 4.1: Deconvolución de Stokes I ---")
    I_centro = intensidad[5, :, :] # Cogemos un punto central del espectro
    I_centro_3d = np.expand_dims(I_centro, axis=0)
    
    print("Aplicando Richardson-Lucy sobre Intensidad (1600x1600)...")
    I_centro_rl_3d = decon.deconvolucion3D(I_centro_3d, psf, metodo='rl', pasos=15)
    I_centro_rl = I_centro_rl_3d[0]
    
    # Métrica: Contraste RMS (Desviación estándar normalizada por la media)
    contraste_orig = np.std(I_centro) / np.mean(I_centro)
    contraste_rl = np.std(I_centro_rl) / np.mean(I_centro_rl)
    print(f"Contraste RMS Original: {contraste_orig:.4f}")
    print(f"Contraste RMS RL: {contraste_rl:.4f}")
    
    # =========================================================================
    # EXPERIMENTO 4.2: Campo Magnético (Stokes V) con NOOR en ROI (Zoom)
    # =========================================================================
    print("\n--- Experimento 4.2: Algoritmo de Noor sobre ROI (400x400) ---")
    
    # Calculamos B completo primero para sacar la zona de más varianza
    B_completo_crudo, _ = mag.calcularCampoMagnetico(intensidad, compV, lambdas, g=1.5)
    
    # Buscamos la zona de mayor varianza en B_completo_crudo
    step = 400
    max_var = 0
    best_y, best_x = 800, 800
    for y in range(0, 1600-step, step//2):
        for x in range(0, 1600-step, step//2):
            var = np.var(B_completo_crudo[y:y+step, x:x+step])
            if var > max_var:
                max_var = var
                best_y, best_x = y + step//2, x + step//2
                
    size = 200 # Ventana de 400x400
    y_min, y_max = max(0, best_y-size), min(1600, best_y+size)
    x_min, x_max = max(0, best_x-size), min(1600, best_x+size)
    
    # Extraemos el ROI de I y V para todos los lambdas
    intensidad_roi = intensidad[:, y_min:y_max, x_min:x_max]
    compV_roi = compV[:, y_min:y_max, x_min:x_max]
    
    print(f"Calculando Campo B directo sobre el ROI [{y_min}:{y_max}, {x_min}:{x_max}]...")
    B_directo_roi, _ = mag.calcularCampoMagnetico(intensidad_roi, compV_roi, lambdas, g=1.5)

    print("Calculando Campo B con Intensidad Deconvolucionada sobre el ROI...")
    I_roi_decon = decon.deconvolucion3D(intensidad_roi, psf, pasos=20)
    B_semi_roi, _ = mag.calcularCampoMagnetico(I_roi_decon, compV_roi, lambdas, g=1.5)
    
    print("Ejecutando Noor (Deconvolución Acoplada) SOBRE EL ROI (400x400)...")
    print("Al ser más pequeño, podemos permitirle 100 iteraciones para que converja de verdad.")
    k_max_actual = mag.calcular_k_max(intensidad_roi, lambdas, g=1.5)
    peso_universal = 0.5  # Peso general unificado
    lambda_dinamico = peso_universal * (k_max_actual**2)

    B_noor_roi, _ = mag.algoritmoDeNoor(
        intensidad_roi, compV_roi, lambdas, psf, g=1.5, 
        lambdaReg=lambda_dinamico, relLim=1e-6, pasosFor=100, cg_auto_close=True
    )
    
    # Métrica 2: Varianza del Laplaciano (Nitidez/Sharpness) para el campo B
    from scipy.ndimage import laplace
    lap_directo = np.var(laplace(B_directo_roi))
    lap_semi = np.var(laplace(B_semi_roi))
    lap_noor = np.var(laplace(B_noor_roi))
    print(f"\nNitidez (Varianza Laplaciano) B Directo (ROI): {lap_directo:.2e}")
    print(f"Nitidez (Varianza Laplaciano) B Semi-Deconvolucionado (ROI): {lap_semi:.2e}")
    print(f"Nitidez (Varianza Laplaciano) B Noor (ROI): {lap_noor:.2e}")
    
    # =========================================================================
    # VISUALIZACIONES
    # =========================================================================
    print("\nGenerando visualizaciones en output/exper/...")
    
    # 1. Intensidad
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    vmin, vmax = np.percentile(I_centro, [1, 99.5])
    axes[0].imshow(I_centro, cmap=sns.color_palette("magma", as_cmap=True), vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Stokes I Original (FITS)\nContraste: {contraste_orig:.4f}')
    axes[0].axis('off')
    
    axes[1].imshow(I_centro_rl, cmap=sns.color_palette("magma", as_cmap=True), vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Stokes I Deconvolucionado (RL)\nContraste: {contraste_rl:.4f} (Mejora: {contraste_rl/contraste_orig:.1f}x)')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('output/exper/exp4_1_intensidad.png', dpi=300)
    plt.close()
    
    # 2. Campo Magnético (Zoom ROI) - Comparamos aquí el efecto real
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    b_max = np.percentile(np.abs(B_completo_crudo), 99.5) # Usamos el maximo global para tener la misma escala de color
    cmap_b = sns.color_palette("icefire", as_cmap=True)
    
    axes[0].imshow(B_directo_roi, cmap=cmap_b, vmin=-b_max, vmax=b_max)
    axes[0].set_title(f'B Directo (I crudo, V crudo)\nNitidez: {lap_directo:.2e}')
    axes[0].axis('off')
    
    axes[1].imshow(B_semi_roi, cmap=cmap_b, vmin=-b_max, vmax=b_max)
    axes[1].set_title(f'B Semideconvolucionado (I decon, V crudo)\nNitidez: {lap_semi:.2e}')
    axes[1].axis('off')

    axes[2].imshow(B_noor_roi, cmap=cmap_b, vmin=-b_max, vmax=b_max)
    axes[2].set_title(f'B Noor Completo (100 iter)\nNitidez: {lap_noor:.2e}')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig('output/exper/exp4_3_campoB_zoom.png', dpi=300)
    plt.close()
    
    # 3. PSF Real (escala logarítmica)
    from matplotlib.colors import LogNorm
    fig_psf, ax_psf = plt.subplots(figsize=(6, 5))
    im_psf = ax_psf.imshow(psf, cmap=sns.color_palette("mako", as_cmap=True), norm=LogNorm(vmin=1e-5, vmax=psf.max()))
    ax_psf.axis('off')
    fig_psf.colorbar(im_psf, ax=ax_psf, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('output/exper/exp4_4_psf.png', dpi=300)
    plt.close()
    
    # 4. Mapa de Diferencia Absoluta (Residuos: |B Noor - B Directo|)
    diferencia_B_abs = np.abs(B_noor_roi - B_directo_roi)
    dif_max = np.percentile(diferencia_B_abs, 99.5)
    
    fig_dif, ax_dif = plt.subplots(figsize=(8, 8))
    im_dif = ax_dif.imshow(diferencia_B_abs, cmap=sns.color_palette("viridis", as_cmap=True), vmin=0, vmax=dif_max)
    ax_dif.set_title(f'Mapa de Residuos (|Noor - Directo|)\nIntensidad máx corrección: {dif_max:.1f} G')
    ax_dif.axis('off')
    fig_dif.colorbar(im_dif, ax=ax_dif, fraction=0.046, pad=0.04, label='Diferencia Absoluta (Gauss)')
    plt.tight_layout()
    plt.savefig('output/exper/exp4_5_diferencia.png', dpi=300)
    plt.close()
    
    print("Experimentos finalizados. Revisa 'output/exper/'.")

if __name__ == '__main__':
    main()
