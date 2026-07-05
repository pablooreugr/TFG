import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import visualizacion as vis
import deconvolucion as decon
from skimage.metrics import structural_similarity as ssim
import magnetismo as mag

# Aplicar el tema de seaborn por defecto
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


def simular_ruido_telescopio(intensidad, compV, snr=1000):
    """
    Añade ruido gaussiano aditivo sigma_ruidosimulando un detector polarimétrico.
    El ruido escala con la intensidad máxima (continuo).
    """
    # 1. Aproximamos la intensidad del continuo (el máximo del mapa de intensidad)
    I_continuo = np.max(intensidad)
    
    # 2. Calculamos la desviación estándar del ruido basada en el SNR
    sigma_ruido = I_continuo / snr
    
    # 3. Generamos los mapas de ruido independiente para I y V
    ruido_I = np.random.normal(loc=0.0, scale=sigma_ruido, size=intensidad.shape)
    ruido_V = np.random.normal(loc=0.0, scale=sigma_ruido, size=compV.shape)
    
    # 4. Sumamos el ruido a los datos
    intensidad_ruidosa = intensidad + ruido_I
    compV_ruidosa = compV + ruido_V
    
    return intensidad_ruidosa, compV_ruidosa


def simular_ruido_telescopio_porcentaje(intensidad, compV, porcentaje_ruido=1):
    """
    Añade ruido gaussiano como porcentaje de la amplitud máxima de cada componente.
    
    Args:
        intensidad: Componente I del vector de Stokes
        compV: Componente V del vector de Stokes
        porcentaje_ruido: Ruido como % de la amplitud máxima respectiva (ej: 5 = 5%)
    
    Returns:
        Tupla (intensidad_ruidosa, compV_ruidosa)
    """
    # 1. Calculamos la amplitud máxima para Intensidad y compV
    I_max = np.max(intensidad)
    V_max = np.max(np.abs(compV))
    
    # 2. Calculamos la desviación estándar del ruido como porcentaje
    sigma_ruido_I = I_max * (porcentaje_ruido / 100.0)
    sigma_ruido_V = V_max * (porcentaje_ruido / 100.0)
    
    # 3. Generamos los mapas de ruido independiente para I y V
    ruido_I = np.random.normal(loc=0.0, scale=sigma_ruido_I, size=intensidad.shape)
    ruido_V = np.random.normal(loc=0.0, scale=sigma_ruido_V, size=compV.shape)
    
    # 4. Sumamos el ruido a los datos
    intensidad_ruidosa = intensidad + ruido_I
    compV_ruidosa = compV + ruido_V
    
    return intensidad_ruidosa, compV_ruidosa


def experimento_algoritmo_noor():
    print("Iniciando experimento del Algoritmo de Noor modificado...")
    import os
    os.makedirs('output/exper', exist_ok=True)
    
    # 1. Cargar datos
    datos_cargados = np.load('data/datos_sunspot.npz')
    data = datos_cargados['stokes']
    lambdas = datos_cargados['lam']

    intensidad = data[:, 0, :, :]
    compV = data[:, 1, :, :]
    lambdas_absolutas = 6173.0 + (lambdas / 1000.0)

    # 2. Generar PSF
    psf = decon.generar_psf_airy(tamano_matriz=31, radio_piz=3)

    # 3. Ground Truth (Campo B Original)
    campoMagnetico, _ = mag.calcularCampoMagnetico(intensidad, compV, lambdas_absolutas)

    # --- EXPERIMENTO BASE (Recuperación y Residuos) ---
    print("\n--- Evaluando algoritmo con ruido del 0.5% ---")
    intenBorrosa = decon.convolucion3D(intensidad, psf)
    compVborrosa = decon.convolucion3D(compV, psf)

    intenBorrosaRuido, compVborrosaRuido = simular_ruido_telescopio_porcentaje(intenBorrosa, compVborrosa, porcentaje_ruido=0.5)
    
    campoBorroso, _ = mag.calcularCampoMagnetico(intenBorrosaRuido, compVborrosaRuido, lambdas_absolutas)
    
    k_max_actual = mag.calcular_k_max(intenBorrosaRuido, lambdas_absolutas, g=3)
    peso_universal = 0.5  # Peso general unificado
    lambda_dinamico = peso_universal * (k_max_actual**2)

    campoMagDeco, historial_conv = mag.algoritmoDeNoor(
        intenBorrosaRuido, compVborrosaRuido, lambdas_absolutas, psf, 
        pasos=20, trabajadores=-1, pasosFor=30, relLim=1e-30, lambdaReg=lambda_dinamico, cg_auto_close=True
    )

    
    rmse_b, ssim_b = calcular_metricas(campoMagnetico, campoBorroso)
    rmse_d, ssim_d = calcular_metricas(campoMagnetico, campoMagDeco)
    
    print(f'Borroso -> RMSE={rmse_b:.4f}, SSIM={ssim_b:.4f}')
    print(f'Deconvolucionado -> RMSE={rmse_d:.4f}, SSIM={ssim_d:.4f}')    

    # --- GRÁFICA DE CONVERGENCIA ---
    fig_c, ax_c = plt.subplots(figsize=(8, 5))
    ax_c.plot(range(1, len(historial_conv)+1), historial_conv, marker='o', color='tab:red')
    ax_c.set_yscale('log')
    ax_c.set_xlabel('Iteración', fontweight='bold')
    ax_c.set_ylabel('Diferencia iterativa (norma)', fontweight='bold')
#     ax_c.set_title('Convergencia del Algoritmo de Noor', fontweight='bold')
    ax_c.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('output/exper/experimento_noor_1_convergencia.png')

    # Configurar colormap para B (icefire con centro blanco)
    cmap_b = sns.color_palette("icefire", as_cmap=True)
    vmax_b = np.percentile(np.abs(campoMagnetico), 99.5)

    # --- COMPARATIVA VISUAL (B) ---
    fig_b, axes_b = plt.subplots(1, 3, figsize=(18, 5))
    
    im1 = axes_b[0].imshow(campoMagnetico, cmap=cmap_b, vmin=-vmax_b, vmax=vmax_b)
#     axes_b[0].set_title("B Original (Ground Truth)", fontweight='bold')
    axes_b[0].axis('off')
    fig_b.colorbar(im1, ax=axes_b[0], fraction=0.046, pad=0.04)

    im2 = axes_b[1].imshow(campoBorroso, cmap=cmap_b, vmin=-vmax_b, vmax=vmax_b)
#     axes_b[1].set_title(f"B Borroso (Ruido 0.5%)\nSSIM: {ssim_b:.4f}", fontweight='bold')
    axes_b[1].axis('off')
    fig_b.colorbar(im2, ax=axes_b[1], fraction=0.046, pad=0.04)

    im3 = axes_b[2].imshow(campoMagDeco, cmap=cmap_b, vmin=-vmax_b, vmax=vmax_b)
#     axes_b[2].set_title(f"B Deconvolucionado (Noor)\nSSIM: {ssim_d:.4f}", fontweight='bold')
    axes_b[2].axis('off')
    fig_b.colorbar(im3, ax=axes_b[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('output/exper/experimento_noor_2_comparativa_b.png')

    # --- MAPA DE RESIDUOS ---
    residuos = np.abs(campoMagnetico - campoMagDeco)
    residuos_borroso = np.abs(campoMagnetico - campoBorroso)
    
    cmap_residuos = sns.color_palette("viridis", as_cmap=True)
    vmax_r = np.percentile(residuos_borroso, 99.5)
    
    fig_r, axes_r = plt.subplots(1, 2, figsize=(12, 5))
    im_r1 = axes_r[0].imshow(residuos_borroso, cmap=cmap_residuos, vmax=vmax_r)
#     axes_r[0].set_title("Residuos (Borroso vs Original)", fontweight='bold')
    axes_r[0].axis('off')
    fig_r.colorbar(im_r1, ax=axes_r[0], fraction=0.046, pad=0.04)

    im_r2 = axes_r[1].imshow(residuos, cmap=cmap_residuos, vmax=vmax_r)
#     axes_r[1].set_title("Residuos (Deconvolucionado vs Original)", fontweight='bold')
    axes_r[1].axis('off')
    fig_r.colorbar(im_r2, ax=axes_r[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('output/exper/experimento_noor_3_residuos.png')
    
    # --- COMPARATIVA VISUAL (B) - ZOOM ---
    y_ini, y_fin = 140, 235
    x_ini, x_fin = 163, 255
    
    fig_b_z, axes_b_z = plt.subplots(1, 3, figsize=(18, 5))
    
    im1_z = axes_b_z[0].imshow(campoMagnetico[y_ini:y_fin, x_ini:x_fin], cmap=cmap_b, vmin=-vmax_b, vmax=vmax_b)
#     axes_b_z[0].set_title("B Original (Zoom)", fontweight='bold')
    axes_b_z[0].axis('off')
    fig_b_z.colorbar(im1_z, ax=axes_b_z[0], fraction=0.046, pad=0.04)

    im2_z = axes_b_z[1].imshow(campoBorroso[y_ini:y_fin, x_ini:x_fin], cmap=cmap_b, vmin=-vmax_b, vmax=vmax_b)
#     axes_b_z[1].set_title("B Borroso (Zoom)", fontweight='bold')
    axes_b_z[1].axis('off')
    fig_b_z.colorbar(im2_z, ax=axes_b_z[1], fraction=0.046, pad=0.04)

    im3_z = axes_b_z[2].imshow(campoMagDeco[y_ini:y_fin, x_ini:x_fin], cmap=cmap_b, vmin=-vmax_b, vmax=vmax_b)
#     axes_b_z[2].set_title("B Deconvolucionado (Zoom)", fontweight='bold')
    axes_b_z[2].axis('off')
    fig_b_z.colorbar(im3_z, ax=axes_b_z[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('output/exper/experimento_noor_2b_comparativa_b_zoom.png')

    # --- MAPA DE RESIDUOS - ZOOM ---
    fig_r_z, axes_r_z = plt.subplots(1, 2, figsize=(12, 5))
    
    im_r1_z = axes_r_z[0].imshow(residuos_borroso[y_ini:y_fin, x_ini:x_fin], cmap=cmap_residuos, vmax=vmax_r)
#     axes_r_z[0].set_title("Residuos (Borroso vs Original) - Zoom", fontweight='bold')
    axes_r_z[0].axis('off')
    fig_r_z.colorbar(im_r1_z, ax=axes_r_z[0], fraction=0.046, pad=0.04)

    im_r2_z = axes_r_z[1].imshow(residuos[y_ini:y_fin, x_ini:x_fin], cmap=cmap_residuos, vmax=vmax_r)
#     axes_r_z[1].set_title("Residuos (Deconvolucionado vs Original) - Zoom", fontweight='bold')
    axes_r_z[1].axis('off')
    fig_r_z.colorbar(im_r2_z, ax=axes_r_z[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('output/exper/experimento_noor_3b_residuos_zoom.png')

    # --- LÍMITES DEL MÉTODO (Ruido) ---
    print("\n--- Evaluando límites del método ante el ruido ---")
    niveles_ruido = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]
    ssims_b_ruido = []
    ssims_d_ruido = []

    for ruido in niveles_ruido:
        i_r, v_r = simular_ruido_telescopio_porcentaje(intenBorrosa, compVborrosa, porcentaje_ruido=ruido)
        cb, _ = mag.calcularCampoMagnetico(i_r, v_r, lambdas_absolutas)
        
        # Reducimos los pasos de CG para ir más rápido en el bucle
        k_max_actual = mag.calcular_k_max(i_r, lambdas_absolutas, g=3)
        peso_universal = 0.5  # Peso general unificado
        lambda_dinamico = peso_universal * (k_max_actual**2)

        cd, _ = mag.algoritmoDeNoor(
            i_r, v_r, lambdas_absolutas, psf, 
            pasos=20, trabajadores=-1, pasosFor=15, relLim=1e-3, lambdaReg=lambda_dinamico, cg_auto_close=True
        )
        
        _, sb = calcular_metricas(campoMagnetico, cb)
        _, sd = calcular_metricas(campoMagnetico, cd)
        
        ssims_b_ruido.append(sb)
        ssims_d_ruido.append(sd)
        print(f"Ruido {ruido:>4.1f}% | SSIM Borroso: {sb:.4f} | SSIM Deconvolucionado: {sd:.4f}")

    fig_l, ax_l = plt.subplots(figsize=(10, 6))
    ax_l.plot(niveles_ruido, ssims_b_ruido, marker='x', linestyle=':', color='gray', linewidth=2, label='Borroso')
    ax_l.plot(niveles_ruido, ssims_d_ruido, marker='o', linestyle='-', color='tab:blue', linewidth=2, label='Deconvolucionado (Noor)')
    
    # Encontrar límite
    for i in range(len(niveles_ruido) - 1):
        if ssims_d_ruido[i] > ssims_b_ruido[i] and ssims_d_ruido[i+1] <= ssims_b_ruido[i+1]:
            ax_l.axvline(x=niveles_ruido[i+1], color='orange', linestyle='-.', alpha=0.7)
            ax_l.text(niveles_ruido[i+1], ax_l.get_ylim()[1]*0.9, ' Límite de mejora', color='orange')
            break
            
    ax_l.set_xlabel('Nivel de Ruido (%)', fontweight='bold')
    ax_l.set_ylabel('SSIM', fontweight='bold')
#     ax_l.set_title('Tolerancia al ruido en la estimación del Campo Magnético (B)', fontweight='bold')
    ax_l.grid(True, linestyle='--', alpha=0.6)
    ax_l.legend()
    plt.tight_layout()
    plt.savefig('output/exper/experimento_noor_4_limite_ruido.png')
    
    print("\nExperimento de Noor completado. Gráficas guardadas en 'output/exper/'.")
    plt.close('all')


def experimento_comparacion_intensidad():
    print("Iniciando experimento de comparación de métodos de deconvolución para la Intensidad...")
    
    # 1. Cargar datos
    datos_cargados = np.load('data/datos_sunspot.npz')
    data = datos_cargados['stokes']
    intensidad = data[:, 0, :, :]
    
    # 2. Generar PSF
    psf = decon.generar_psf_airy(tamano_matriz=31, radio_piz=3)
    
    # 3. Degradar la imagen (Borroso + Ruido)
    print("Aplicando degradación (PSF + Ruido)...")
    intenBorrosa = decon.convolucion3D(intensidad, psf)
    
    # Usamos dummy compV porque sólo nos interesa intensidad
    dummy_compV = np.zeros_like(intensidad)
    intenBorrosaRuido, _ = simular_ruido_telescopio_porcentaje(intenBorrosa, dummy_compV, porcentaje_ruido=0.1)
    
    # 4. Deconvolución con los tres métodos
    print("Aplicando Deconvolución Richardson-Lucy...")
    inten_rl = decon.deconvolucion3D(intenBorrosaRuido, psf, metodo='rl', pasos=20)
    
    print("Aplicando Deconvolución Fourier...")
    inten_fourier = decon.deconvolucion3D(intenBorrosaRuido, psf, metodo='fourier')
    
    print("Aplicando Deconvolución Wiener...")
    inten_wiener = decon.deconvolucion3D(intenBorrosaRuido, psf, metodo='wiener')
    
    # 5. Calcular métricas
    metodos = ['Borroso + Ruido', 'Richardson-Lucy', 'Fourier', 'Wiener']
    imagenes = [intenBorrosaRuido, inten_rl, inten_fourier, inten_wiener]
    
    rmses = []
    ssims = []
    
    print("\n--- Resultados de las Métricas ---")
    for nombre, img in zip(metodos, imagenes):
        rmse_val, ssim_val = calcular_metricas(intensidad, img)
        rmses.append(rmse_val)
        ssims.append(ssim_val)
        print(f"{nombre} -> RMSE: {rmse_val:.4f}, SSIM: {ssim_val:.4f}")
        
    # 6. Visualización de Resultados
    import os
    import matplotlib.colors as mcolors
    
    os.makedirs('output/exper', exist_ok=True)
    
    idx_z = intensidad.shape[0] // 2  # Usamos un slice central para visualizar
    
    print("\nGenerando visualizaciones y guardando en 'output/'...")
    
    # 6.1 Gráfica 1: Original y Borrosa
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes1[0].imshow(intensidad[idx_z], cmap=sns.color_palette("rocket", as_cmap=True))
    axes1[0].axis('off')
    fig1.colorbar(im1, ax=axes1[0], fraction=0.046, pad=0.04)
    
    im2 = axes1[1].imshow(intenBorrosaRuido[idx_z], cmap=sns.color_palette("rocket", as_cmap=True))
    axes1[1].axis('off')
    fig1.colorbar(im2, ax=axes1[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('output/exper/comparacion_intensidad_1_original_borrosa.png')

    # 6.2 Gráfica 2: 3 Deconvoluciones
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    im3 = axes2[0].imshow(inten_rl[idx_z], cmap=sns.color_palette("rocket", as_cmap=True))
    axes2[0].axis('off')
    fig2.colorbar(im3, ax=axes2[0], fraction=0.046, pad=0.04)
    
    im4 = axes2[1].imshow(inten_fourier[idx_z], cmap=sns.color_palette("rocket", as_cmap=True))
    axes2[1].axis('off')
    fig2.colorbar(im4, ax=axes2[1], fraction=0.046, pad=0.04)
    
    im5 = axes2[2].imshow(inten_wiener[idx_z], cmap=sns.color_palette("rocket", as_cmap=True))
    axes2[2].axis('off')
    fig2.colorbar(im5, ax=axes2[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('output/exper/comparacion_intensidad_2_deconvoluciones.png')
    
    # 6.3 Gráfica 3: Barras
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.barplot(x=metodos, y=ssims, ax=ax3, hue=metodos, palette="viridis", legend=False)
    ax3.set_xlabel('Método de Deconvolución', fontweight='bold')
    ax3.set_ylabel('SSIM', fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/exper/comparacion_intensidad_3_barras.png')

    # 6.4 Gráfica 4: PSF (escala logarítmica)
    from matplotlib.colors import LogNorm
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    im6 = ax4.imshow(psf, cmap=sns.color_palette("mako", as_cmap=True), norm=LogNorm(vmin=1e-5, vmax=psf.max()))
    ax4.axis('off')
    fig4.colorbar(im6, ax=ax4, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('output/exper/comparacion_intensidad_4_psf.png')
    
    # 6.5 Gráfica 5: Original y Borrosa (Zoom)
    print("Generando visualizaciones con zoom y guardando en 'output/'...")
    y_ini, y_fin = 140, 235
    x_ini, x_fin = 163, 255
    
    fig5, axes5 = plt.subplots(1, 2, figsize=(10, 5))
    im1_z = axes5[0].imshow(intensidad[idx_z, y_ini:y_fin, x_ini:x_fin], cmap=sns.color_palette("rocket", as_cmap=True))
    axes5[0].axis('off')
    fig5.colorbar(im1_z, ax=axes5[0], fraction=0.046, pad=0.04)
    
    im2_z = axes5[1].imshow(intenBorrosaRuido[idx_z, y_ini:y_fin, x_ini:x_fin], cmap=sns.color_palette("rocket", as_cmap=True))
    axes5[1].axis('off')
    fig5.colorbar(im2_z, ax=axes5[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('output/exper/comparacion_intensidad_5_original_borrosa_zoom.png')
    
    # 6.6 Gráfica 6: 3 Deconvoluciones (Zoom)
    fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5))
    
    im3_z = axes6[0].imshow(inten_rl[idx_z, y_ini:y_fin, x_ini:x_fin], cmap=sns.color_palette("rocket", as_cmap=True))
    axes6[0].axis('off')
    fig6.colorbar(im3_z, ax=axes6[0], fraction=0.046, pad=0.04)
    
    im4_z = axes6[1].imshow(inten_fourier[idx_z, y_ini:y_fin, x_ini:x_fin], cmap=sns.color_palette("rocket", as_cmap=True))
    axes6[1].axis('off')
    fig6.colorbar(im4_z, ax=axes6[1], fraction=0.046, pad=0.04)
    
    im5_z = axes6[2].imshow(inten_wiener[idx_z, y_ini:y_fin, x_ini:x_fin], cmap=sns.color_palette("rocket", as_cmap=True))
    axes6[2].axis('off')
    fig6.colorbar(im5_z, ax=axes6[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('output/exper/comparacion_intensidad_6_deconvoluciones_zoom.png')

    # Mostrar todas las gráficas generadas
    plt.close('all')

def experimento_rl_pasos():
    print("Iniciando experimento de barrido de pasos para Richardson-Lucy...")
    
    # 1. Cargar datos
    datos_cargados = np.load('data/datos_sunspot.npz')
    data = datos_cargados['stokes']
    
    # Solo usar lambda 0 de la intensidad
    intensidad = data[0, 0, :, :]
    
    # 2. Generar PSF
    psf = decon.generar_psf_airy(tamano_matriz=31, radio_piz=3)
    
    # Agregar dimensión para que sean 3D (requerido por deconvolucion3D)
    intensidad_3d = np.expand_dims(intensidad, axis=0)
    
    # 3. Degradar la imagen (Borroso + Ruido)
    print("Aplicando degradación (PSF + Ruido)...")
    intensidad_borrosa_3d = decon.convolucion3D(intensidad_3d, psf)
    
    dummy_compV = np.zeros_like(intensidad_3d)
    intensidad_borrosa_ruido_3d, _ = simular_ruido_telescopio_porcentaje(intensidad_borrosa_3d, dummy_compV, porcentaje_ruido=2.0)
    
    intensidad_borrosa_ruido = intensidad_borrosa_ruido_3d[0]
    
    rmse_br, ssim_br = calcular_metricas(intensidad, intensidad_borrosa_ruido)
    print(f'Borroso y Ruidoso -> RMSE={rmse_br:.4f}, SSIM={ssim_br:.4f}')

    # 4. Barrido de pasos
    pasos_arr = [1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100]
    rmses = []
    ssims = []
    
    for p in pasos_arr:
        print(f"Calculando RL con {p} pasos...")
        intensidad_decoRL = decon.deconvolucion3D(intensidad_borrosa_ruido_3d, psf, metodo='rl', pasos=p)[0]
        rmse_val, ssim_val = calcular_metricas(intensidad, intensidad_decoRL)
        rmses.append(rmse_val)
        ssims.append(ssim_val)
        print(f"  -> RMSE={rmse_val:.4f}, SSIM={ssim_val:.4f}")

    # 5. Visualización
    import os
    os.makedirs('output', exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Número de pasos (Richardson-Lucy)', fontweight='bold')
    ax1.set_ylabel('RMSE', color=color, fontweight='bold')
    ax1.plot(pasos_arr, rmses, marker='o', color=color, label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('SSIM', color=color, fontweight='bold')  
    ax2.plot(pasos_arr, ssims, marker='s', color=color, label='SSIM')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
#     plt.title('Evolución de RMSE y SSIM en función de los pasos de RL', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Añadir leyendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.savefig('output/experimento_rl_pasos.png')
    plt.show()


def experimento_comparativa_rl_wk_ruido():
    print("Iniciando comparativa ampliada (RL vs Wiener vs Borroso) según el nivel de ruido...")
    
    # 1. Cargar datos
    datos_cargados = np.load('data/datos_sunspot.npz')
    data = datos_cargados['stokes']
    
    # Solo usar lambda 0 de la intensidad
    intensidad = data[0, 0, :, :]
    
    # 2. Generar PSF
    psf = decon.generar_psf_airy(tamano_matriz=31, radio_piz=3)
    intensidad_3d = np.expand_dims(intensidad, axis=0)
    
    # Imagen borrosa base (sin ruido aún)
    intensidad_borrosa_3d = decon.convolucion3D(intensidad_3d, psf)
    dummy_compV = np.zeros_like(intensidad_3d)

    niveles_ruido = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0, 12.0, 15.0]
    pasos_arr = [1, 2, 3, 4, 5, 10, 20, 50, 100]
    
    ssims_rl_optimo = []
    ssims_wk = []
    ssims_borroso = []
    
    import os
    out_dir = '/home/pabloore/conjuntoV/universidad/TFG/output/exper'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"{'Ruido (%)':<10} | {'SSIM Borroso':<15} | {'SSIM Wiener':<15} | {'SSIM RL (Pasos)':<20} | {'¿Wiener Válido?':<15}")
    print("-" * 85)
    
    for i, ruido in enumerate(niveles_ruido):
        # Añadir ruido
        intensidad_borrosa_ruido_3d, _ = simular_ruido_telescopio_porcentaje(
            intensidad_borrosa_3d, dummy_compV, porcentaje_ruido=ruido
        )
        
        # 0. Evaluar Borroso (Línea base)
        intensidad_borrosa_ruido = intensidad_borrosa_ruido_3d[0]
        _, ssim_borroso_val = calcular_metricas(intensidad, intensidad_borrosa_ruido)
        ssims_borroso.append(ssim_borroso_val)
        
        # 1. Evaluar Wiener (wk)
        intensidad_decoWK = decon.deconvolucion3D(intensidad_borrosa_ruido_3d, psf, metodo='wk')[0]
        _, ssim_wk_val = calcular_metricas(intensidad, intensidad_decoWK)
        ssims_wk.append(ssim_wk_val)
        
        # 2. Evaluar RL para buscar el óptimo
        mejor_ssim_rl = -1
        mejor_paso = -1
        
        for p in pasos_arr:
            intensidad_decoRL = decon.deconvolucion3D(intensidad_borrosa_ruido_3d, psf, metodo='rl', pasos=p)[0]
            _, ssim_val = calcular_metricas(intensidad, intensidad_decoRL)
            
            if ssim_val > mejor_ssim_rl:
                mejor_ssim_rl = ssim_val
                mejor_paso = p
                
        ssims_rl_optimo.append(mejor_ssim_rl)
        
        valido = "SÍ" if ssim_wk_val > ssim_borroso_val else "NO"
        print(f"{ruido:<10.1f} | {ssim_borroso_val:<15.4f} | {ssim_wk_val:<15.4f} | {mejor_ssim_rl:.4f} ({mejor_paso:<3})     | {valido:<15}")

    # 3. Visualización de la comparativa
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(niveles_ruido, ssims_borroso, marker='x', linestyle=':', color='gray', linewidth=2, label='Borroso (Sin deconvolucionar)')
    ax.plot(niveles_ruido, ssims_wk, marker='s', linestyle='--', color='tab:blue', linewidth=2, label='Wiener (wk)')
    ax.plot(niveles_ruido, ssims_rl_optimo, marker='o', linestyle='-', color='tab:red', linewidth=2, label='Richardson-Lucy (Óptimo)')

    # Encontrar el punto de cruce donde Wiener deja de ser válido
    for i in range(len(niveles_ruido) - 1):
        if ssims_wk[i] > ssims_borroso[i] and ssims_wk[i+1] <= ssims_borroso[i+1]:
            ax.axvline(x=niveles_ruido[i+1], color='orange', linestyle='-.', alpha=0.7)
            ax.text(niveles_ruido[i+1], ax.get_ylim()[1]*0.9, 'Límite validez WK', color='orange', rotation=90, verticalalignment='top')
            break

    ax.set_xlabel('Nivel de Ruido (%)', fontweight='bold')
    ax.set_ylabel('SSIM', fontweight='bold')
#     ax.set_title('Comparativa SSIM: RL vs Wiener vs Borroso Original', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Eje X lineal o log? Para ver el límite mejor usamos lineal
    ax.set_xticks(niveles_ruido)
    ax.set_xticklabels([str(r) for r in niveles_ruido])
    ax.legend()
    
    plt.tight_layout()
    path_salida = os.path.join(out_dir, 'limite_validez_wk_ruido.png')
    plt.savefig(path_salida)
    print(f"\nGráfica guardada en: {path_salida}")
    plt.show()

def experimento_wiener_compV_ruido():
    print("Iniciando experimento de Wiener (Propio) sobre la componente Stokes V...")
    
    # 1. Cargar datos
    datos_cargados = np.load('data/datos_sunspot.npz')
    data = datos_cargados['stokes']
    
    # Usar lambda 0
    intensidad = data[0, 0, :, :]
    compV = data[0, 1, :, :]
    
    # 2. Generar PSF
    psf = decon.generar_psf_airy(tamano_matriz=31, radio_piz=3)
    intensidad_3d = np.expand_dims(intensidad, axis=0)
    compV_3d = np.expand_dims(compV, axis=0)
    
    # Imagen borrosa base (sin ruido aún)
    intensidad_borrosa_3d = decon.convolucion3D(intensidad_3d, psf)
    compV_borrosa_3d = decon.convolucion3D(compV_3d, psf)

    niveles_ruido = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0, 12.0, 15.0]
    
    ssims_wiener = []
    ssims_borroso = []
    
    import os
    out_dir = 'output/exper'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"{'Ruido (%)':<10} | {'SSIM Borroso V':<15} | {'SSIM Mi Wiener V':<15} | {'¿Wiener Válido?':<15}")
    print("-" * 65)
    
    # Variables para guardar un ejemplo de ruido alto para visualizar
    ejemplo_ruido = 2.0
    compV_ejemplo_original = compV
    compV_ejemplo_borroso = None
    compV_ejemplo_wiener = None
    
    for i, ruido in enumerate(niveles_ruido):
        # Añadir ruido
        _, compV_borrosa_ruido_3d = simular_ruido_telescopio_porcentaje(
            intensidad_borrosa_3d, compV_borrosa_3d, porcentaje_ruido=ruido
        )
        
        # 0. Evaluar Borroso (Línea base) para compV
        compV_borrosa_ruido = compV_borrosa_ruido_3d[0]
        _, ssim_borroso_val = calcular_metricas(compV, compV_borrosa_ruido)
        ssims_borroso.append(ssim_borroso_val)
        
        # 1. Evaluar Wiener propio para compV
        compV_deco_wiener = decon.deconvolucion3D(compV_borrosa_ruido_3d, psf, metodo='wiener')[0]
        _, ssim_wiener_val = calcular_metricas(compV, compV_deco_wiener)
        ssims_wiener.append(ssim_wiener_val)
        
        # Guardar ejemplo
        if abs(ruido - ejemplo_ruido) < 0.01:
            compV_ejemplo_borroso = compV_borrosa_ruido
            compV_ejemplo_wiener = compV_deco_wiener
        
        valido = "SÍ" if ssim_wiener_val > ssim_borroso_val else "NO"
        print(f"{ruido:<10.1f} | {ssim_borroso_val:<15.4f} | {ssim_wiener_val:<15.4f} | {valido:<15}")

    # 3. Visualización de la comparativa de SSIM
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(niveles_ruido, ssims_borroso, marker='x', linestyle=':', color='gray', linewidth=2, label='Stokes V Borroso (Sin deconvolucionar)')
    ax.plot(niveles_ruido, ssims_wiener, marker='s', linestyle='--', color='tab:blue', linewidth=2, label='Stokes V Mi Wiener')

    ax.set_xlabel('Nivel de Ruido (%)', fontweight='bold')
    ax.set_ylabel('SSIM', fontweight='bold')
#     ax.set_title('Comparativa SSIM (Stokes V): Mi Wiener vs Borroso Original', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_xticks(niveles_ruido)
    ax.set_xticklabels([str(r) for r in niveles_ruido])
    ax.legend()
    
    plt.tight_layout()
    path_salida_1 = os.path.join(out_dir, 'experimento_stokesV_1_limite_ruido.png')
    plt.savefig(path_salida_1)
    
    # 4. Visualización directa de mapas
    if compV_ejemplo_borroso is not None:
        # Para mejorar el contraste, usamos un percentil (99.0) en lugar del máximo absoluto y subimos la saturación de 60 a 85
        cmap_v = sns.diverging_palette(145, 300, s=85, l=50, center="light", as_cmap=True)
        v_max = np.percentile(np.abs(compV_ejemplo_original), 99.0)
        
        fig_v, axes_v = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = axes_v[0].imshow(compV_ejemplo_original, cmap=cmap_v, vmin=-v_max, vmax=v_max)
#         axes_v[0].set_title('Stokes V Original', fontweight='bold')
        axes_v[0].axis('off')
        fig_v.colorbar(im1, ax=axes_v[0], fraction=0.046, pad=0.04)

        im2 = axes_v[1].imshow(compV_ejemplo_borroso, cmap=cmap_v, vmin=-v_max, vmax=v_max)
#         axes_v[1].set_title(f'Stokes V Borroso (Ruido {ejemplo_ruido}%)', fontweight='bold')
        axes_v[1].axis('off')
        fig_v.colorbar(im2, ax=axes_v[1], fraction=0.046, pad=0.04)

        im3 = axes_v[2].imshow(compV_ejemplo_wiener, cmap=cmap_v, vmin=-v_max, vmax=v_max)
#         axes_v[2].set_title('Stokes V Deconvolucionado (Mi Wiener)', fontweight='bold')
        axes_v[2].axis('off')
        fig_v.colorbar(im3, ax=axes_v[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        path_salida_2 = os.path.join(out_dir, 'experimento_stokesV_2_visualizacion.png')
        plt.savefig(path_salida_2)
        
    print(f"\nGráficas guardadas en {out_dir}")
    plt.close('all')

def experimento_stokesV_shift_ruido():
    print("\nIniciando experimento de deconvolución de Stokes V con shift positivo (RL, Fourier, Wiener)...")
    
    # 1. Cargar datos
    datos_cargados = np.load('data/datos_sunspot.npz')
    data = datos_cargados['stokes']
    
    compV = data[0, 1, :, :]
    
    # 2. Generar PSF
    psf = decon.generar_psf_airy(tamano_matriz=31, radio_piz=3)
    compV_3d = np.expand_dims(compV, axis=0)
    
    compV_borrosa_3d = decon.convolucion3D(compV_3d, psf)

    niveles_ruido = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0, 12.0, 15.0]
    
    ssims_borroso = []
    ssims_rl = []
    ssims_fourier = []
    ssims_wk = []
    ssims_wiener = []
    
    import os
    out_dir = 'output/exper'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"{'Ruido (%)':<10} | {'Borroso':<10} | {'RL (Shift)':<12} | {'Fourier':<10} | {'Wiener (wk)':<12} | {'Mi Wiener':<10}")
    print("-" * 75)
    
    for ruido in niveles_ruido:
        intensidad = data[0, 0, :, :]
        intensidad_3d = np.expand_dims(intensidad, axis=0)
        intensidad_borrosa_3d = decon.convolucion3D(intensidad_3d, psf)
        
        _, compV_borrosa_ruido_3d = simular_ruido_telescopio_porcentaje(
            intensidad_borrosa_3d, compV_borrosa_3d, porcentaje_ruido=ruido
        )
        
        compV_borrosa_ruido = compV_borrosa_ruido_3d[0]
        
        # 0. Evaluar Borroso
        _, ssim_borroso_val = calcular_metricas(compV, compV_borrosa_ruido)
        ssims_borroso.append(ssim_borroso_val)
        
        # --- SHIFT POSITIVO SOLO PARA RL ---
        min_val = np.min(compV_borrosa_ruido_3d)
        shift = abs(min_val) + 0.1 if min_val < 0 else 0
        compV_borrosa_ruido_3d_shifted = compV_borrosa_ruido_3d + shift
        
        # 1. RL (Shifted)
        compV_rl_shifted = decon.deconvolucion3D(compV_borrosa_ruido_3d_shifted, psf, metodo='rl')[0]
        compV_rl = compV_rl_shifted - shift
        _, ssim_rl_val = calcular_metricas(compV, compV_rl)
        ssims_rl.append(ssim_rl_val)
        
        # 2. Fourier (Normal)
        compV_fourier = decon.deconvolucion3D(compV_borrosa_ruido_3d, psf, metodo='fourier')[0]
        _, ssim_fourier_val = calcular_metricas(compV, compV_fourier)
        ssims_fourier.append(ssim_fourier_val)
        
        # 3. Wiener (wk) (Normal)
        compV_wk = decon.deconvolucion3D(compV_borrosa_ruido_3d, psf, metodo='wk')[0]
        _, ssim_wk_val = calcular_metricas(compV, compV_wk)
        ssims_wk.append(ssim_wk_val)
        
        # 4. Mi Wiener (Normal)
        compV_wiener = decon.deconvolucion3D(compV_borrosa_ruido_3d, psf, metodo='wiener')[0]
        _, ssim_wiener_val = calcular_metricas(compV, compV_wiener)
        ssims_wiener.append(ssim_wiener_val)
        
        print(f"{ruido:<10.1f} | {ssim_borroso_val:<10.4f} | {ssim_rl_val:<12.4f} | {ssim_fourier_val:<10.4f} | {ssim_wk_val:<12.4f} | {ssim_wiener_val:<10.4f}")

    # Visualización
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(niveles_ruido, ssims_borroso, marker='x', linestyle=':', color='gray', linewidth=2, label='Stokes V Borroso')
    ax.plot(niveles_ruido, ssims_rl, marker='o', linestyle='-', color='tab:orange', linewidth=2, label='RL (Shifted)')
    
    # Clip Fourier SSIM for plotting
    ssims_fourier_clipped = np.clip(ssims_fourier, -0.1, 1.0)
    ax.plot(niveles_ruido, ssims_fourier_clipped, marker='^', linestyle='-.', color='tab:green', linewidth=2, label='Fourier (Normal)')
    
    ax.plot(niveles_ruido, ssims_wk, marker='s', linestyle='--', color='tab:blue', linewidth=2, label='Wiener wk (Normal)')
    ax.plot(niveles_ruido, ssims_wiener, marker='d', linestyle='-', color='tab:purple', linewidth=2, label='Mi Wiener (Normal)')

    ax.set_ylim([-0.1, 1.05])
    ax.set_xlabel('Nivel de Ruido (%)', fontweight='bold')
    ax.set_ylabel('SSIM', fontweight='bold')
#     ax.set_title('Comparativa Métodos (Stokes V): RL con Shift vs Resto Normal', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_xticks(niveles_ruido)
    ax.set_xticklabels([str(r) for r in niveles_ruido])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    path_salida = os.path.join(out_dir, 'experimento_stokesV_3_comparativa_shift.png')
    plt.savefig(path_salida)
    print(f"\nGráfica guardada en {path_salida}")
    plt.close('all')

if __name__ == "__main__":
    # Puedes descomentar la siguiente línea si quieres ejecutar el experimento del Algoritmo de Noor
    experimento_algoritmo_noor()
    
    experimento_comparacion_intensidad()
    
    experimento_rl_pasos()
    
    # experimento_rl_ruido_pasos()
    
    # experimento_comparativa_rl_wk_ruido()
    
    experimento_wiener_compV_ruido()
    
    experimento_stokesV_shift_ruido()
