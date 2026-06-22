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

    # 2. Añadimos el ruido del sensor (ej. SNR = 1000)
    intenBorrosaRuido, compVborrosaRuido = simular_ruido_telescopio(intenBorrosa, compVborrosa, snr=1000)

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
    intenBorrosaRuido, _ = simular_ruido_telescopio(intenBorrosa, dummy_compV, snr=1000)
    
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

if __name__ == "__main__":
    # Puedes descomentar la siguiente línea si quieres ejecutar el experimento del Algoritmo de Noor
    # experimento_algoritmo_noor()
    
    experimento_comparacion_intensidad()

