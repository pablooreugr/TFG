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
    datos_cargados = np.load('data/datos_sunspot.npz')

    data = datos_cargados['stokes']
    lambdas = datos_cargados['lam']

    intensidad = data[:, 0, :, :]
    compV = data[:, 1, :, :]

    psf = decon.generar_psf_airy(
        tamano_matriz=31,
        radio_piz=3
    )

    lambdas_absolutas = 6173.0 + (lambdas / 1000.0)

    campoMagnetico, _ = mag.calcularCampoMagnetico(intensidad, compV, lambdas_absolutas)

    # Imagen degradada
    # 1. Imagen degradada por la difracción (PSF)
    intenBorrosa = decon.convolucion3D(intensidad, psf)
    compVborrosa = decon.convolucion3D(compV, psf)

    # 2. Añadimos el ruido del sensor (ej. 0.1% de ruido)
    intenBorrosaRuido, compVborrosaRuido = simular_ruido_telescopio_porcentaje(intenBorrosa, compVborrosa, porcentaje_ruido=0.1)

    # 3. Calculamos las métricas del campo borroso y ruidoso
    campoBorroso, _ = mag.calcularCampoMagnetico(intenBorrosaRuido, compVborrosaRuido, lambdas_absolutas)


    rmse_campo_borroso, ssim_campo_borroso = calcular_metricas(
        campoMagnetico,
        campoBorroso
    )

    print(f'Borroso -> RMSE={rmse_campo_borroso:.4f}, SSIM={ssim_campo_borroso:.4f}')
    
    campoMagDeco = mag.algoritmoDeNoor(intenBorrosaRuido, compVborrosaRuido, lambdas_absolutas, psf, pasos=20, trabajadores=-1, pasosFor=2, relLim=1e-30, lambdaReg=1e-5)

    rmse_campo_deco, ssim_campo_deco = calcular_metricas(
        campoMagnetico,
        campoMagDeco
    )

    print(f'Deconvolucionado -> RMSE={rmse_campo_deco:.4f}, SSIM={ssim_campo_deco:.4f}')    

    vis.mostrar_n_arrays(
        [campoMagnetico, campoBorroso, campoMagDeco],
        ['Campo Magnético Original', 'Campo Magnético Borroso y Ruidoso', 'Campo Magnético Deconvolucionado'],
        cmap='viridis'
    )


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
    
    os.makedirs('output', exist_ok=True)
    
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
    plt.savefig('output/comparacion_intensidad_1_original_borrosa.png')

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
    plt.savefig('output/comparacion_intensidad_2_deconvoluciones.png')
    
    # 6.3 Gráfica 3: Barras
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.barplot(x=metodos, y=ssims, ax=ax3, hue=metodos, palette="viridis", legend=False)
    ax3.set_xlabel('Método de Deconvolución', fontweight='bold')
    ax3.set_ylabel('SSIM', fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/comparacion_intensidad_3_barras.png')

    # 6.4 Gráfica 4: PSF (sin logaritmo)
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    im6 = ax4.imshow(psf, cmap=sns.color_palette("mako", as_cmap=True))
    ax4.axis('off')
    fig4.colorbar(im6, ax=ax4, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('output/comparacion_intensidad_4_psf.png')
    
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
    plt.savefig('output/comparacion_intensidad_5_original_borrosa_zoom.png')
    
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
    plt.savefig('output/comparacion_intensidad_6_deconvoluciones_zoom.png')

    # Mostrar todas las gráficas generadas
    plt.show(block=True)

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
    plt.title('Evolución de RMSE y SSIM en función de los pasos de RL', fontweight='bold')
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
    ax.set_title('Comparativa SSIM: RL vs Wiener vs Borroso Original', fontweight='bold')
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

def experimento_wk_compV_ruido():
    print("Iniciando experimento de Wiener (wk) sobre la componente Stokes V...")
    
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
    
    ssims_wk = []
    ssims_borroso = []
    
    import os
    out_dir = '/home/pabloore/conjuntoV/universidad/TFG/output/exper'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"{'Ruido (%)':<10} | {'SSIM Borroso V':<15} | {'SSIM Wiener V':<15} | {'¿Wiener Válido?':<15}")
    print("-" * 60)
    
    for i, ruido in enumerate(niveles_ruido):
        # Añadir ruido
        _, compV_borrosa_ruido_3d = simular_ruido_telescopio_porcentaje(
            intensidad_borrosa_3d, compV_borrosa_3d, porcentaje_ruido=ruido
        )
        
        # 0. Evaluar Borroso (Línea base) para compV
        compV_borrosa_ruido = compV_borrosa_ruido_3d[0]
        _, ssim_borroso_val = calcular_metricas(compV, compV_borrosa_ruido)
        ssims_borroso.append(ssim_borroso_val)
        
        # 1. Evaluar Wiener (wk) para compV
        compV_decoWK = decon.deconvolucion3D(compV_borrosa_ruido_3d, psf, metodo='wk')[0]
        _, ssim_wk_val = calcular_metricas(compV, compV_decoWK)
        ssims_wk.append(ssim_wk_val)
        
        valido = "SÍ" if ssim_wk_val > ssim_borroso_val else "NO"
        print(f"{ruido:<10.1f} | {ssim_borroso_val:<15.4f} | {ssim_wk_val:<15.4f} | {valido:<15}")

    # 3. Visualización de la comparativa
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(niveles_ruido, ssims_borroso, marker='x', linestyle=':', color='gray', linewidth=2, label='Stokes V Borroso (Sin deconvolucionar)')
    ax.plot(niveles_ruido, ssims_wk, marker='s', linestyle='--', color='tab:blue', linewidth=2, label='Stokes V Wiener (wk)')

    ax.set_xlabel('Nivel de Ruido (%)', fontweight='bold')
    ax.set_ylabel('SSIM', fontweight='bold')
    ax.set_title('Comparativa SSIM (Stokes V): Wiener vs Borroso Original', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Eje X lineal
    ax.set_xticks(niveles_ruido)
    ax.set_xticklabels([str(r) for r in niveles_ruido])
    ax.legend()
    
    plt.tight_layout()
    path_salida = os.path.join(out_dir, 'limite_validez_wk_compV_ruido.png')
    plt.savefig(path_salida)
    print(f"\nGráfica guardada en: {path_salida}")
    plt.show()

if __name__ == "__main__":
    # Puedes descomentar la siguiente línea si quieres ejecutar el experimento del Algoritmo de Noor
    # experimento_algoritmo_noor()
    
    # experimiento_comparacion_intensidad()
    
    # experimento_rl_pasos()
    
    # experimento_rl_ruido_pasos()
    
    # experimento_comparativa_rl_wk_ruido()
    
    experimento_wk_compV_ruido()
