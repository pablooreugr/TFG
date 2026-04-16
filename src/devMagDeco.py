from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import deconvolucion as decon


g_eff = 1.75 # Linea del magnesio I
constanteFormula = 4.67e-13 

def decWienerAxis0(imagen, psf):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionWiener(imagen[i], psf, k)

    return imagen

def decWienerAxis0Multi(imagen, psf):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionWienerMulti(imagen[i], psf, k)

    return imagen

def decRLAxis0(imagen, psf, iteraciones=500, epsilon=1):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionRL(imagen[i], psf, iteraciones, k, epsilon)

    return imagen

def decRLAxis0Multi(imagen, psf, iteraciones=500, epsilon=1):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionRLMulti(imagen[i], psf, iteraciones, k, epsilon)

    return imagen

def decFourierAxis0(imagen, psf):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionFourier(imagen[i], psf, k)

    return imagen

def decFourierAxis0Multi(imagen, psf):
    for i in range(imagen.shape[0]):
        print(f"Deconvolución Fourier (Multi) - Lambda {i}...")
        imagen[i] = decon.deconvolucionFourierMulti(imagen[i], psf, k)

    return imagen

def decAxis0Multi(imagen, psf=None, metDecon='w', iteraciones=30, epsilon=1, zernikes=None, k=1e-3):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionMulti(imagen[i], psf=psf, metodo=metDecon, iteraciones=iteraciones, k=k, epsilon=epsilon, zernikes=zernikes)
    return imagen


def magnetismoDirectamente(imagen, psf, lambdas, metDecon='wiener', iteraciones=1000, epsilon=1, k=1e-3):
    imagenIntensidad = imagen[:, 0, :, :]
    imagenV = imagen[:, 3, :, :]

    if metDecon == 'wiener':
        imagenIntensidad = decWienerAxis0(imagenIntensidad, psf, k)
        imagenV = decWienerAxis0(imagenV, psf, k)
    elif metDecon == 'fourier':
        imagenIntensidad = decFourierAxis0(imagenIntensidad, psf, k)
        imagenV = decFourierAxis0(imagenV, psf, k)
    elif metDecon == 'rl':
        imagenIntensidad = decRLAxis0(imagenIntensidad, psf, iteraciones, k, epsilon)
        imagenV = decRLAxis0(imagenV, psf, iteraciones, k, epsilon)

    lambdas3D = lambdas[:, np.newaxis, np.newaxis]

    gradIntensidad = np.gradient(imagenIntensidad, lambdas, axis=0) * (lambdas3D**2)

    # A partir de ahora calculamos la regresionLineal
    m = np.sum(gradIntensidad*imagenV, axis=0)/np.sum(gradIntensidad**2, axis=0)

    v_predicho = gradIntensidad * m

    mediaDatosV = np.mean(imagenV, axis=0)
    
    # Numerador: Suma de los residuos al cuadrado
    numerador = np.sum((imagenV - v_predicho)**2, axis=0)
    
    # Denominador: Suma de la varianza total
    denominador = np.sum((imagenV - mediaDatosV)**2, axis=0)
    
    # Calculamos R^2
    mapa_r_cuadrado = 1 - (numerador / denominador)

    campoMagnetico = -m*(1/(g_eff*constanteFormula))

    return campoMagnetico, mapa_r_cuadrado


def magnetismoDirectamenteMulti(imagen, psf, lambdas, metDecon='wiener', iteraciones=1000, epsilon=1): # En este calculamos la deconvolucion de V y de I directamente, luego sacamos por regresion lineal el magnetismo
    imagenIntensidad = imagen[:, 0, :, :]
    imagenV = imagen[:, 3, :, :]

    if metDecon == 'wiener':
        imagenIntensidad = decWienerAxis0Multi(imagenIntensidad, psf, k)
        imagenV = decWienerAxis0Multi(imagenV, psf, k)
    elif metDecon == 'fourier':
        imagenIntensidad = decFourierAxis0Multi(imagenIntensidad, psf, k)
        imagenV = decFourierAxis0Multi(imagenV, psf, k)
    elif metDecon == 'rl':
        imagenIntensidad = decRLAxis0Multi(imagenIntensidad, psf, iteraciones, k, epsilon)
        imagenV = decRLAxis0Multi(imagenV, psf, iteraciones, k, epsilon)

    lambdas3D = lambdas[:, np.newaxis, np.newaxis]

    gradIntensidad = np.gradient(imagenIntensidad, lambdas, axis=0) * (lambdas3D**2)

    # A partir de ahora calculamos la regresionLineal
    m = np.sum(gradIntensidad*imagenV, axis=0)/np.sum(gradIntensidad**2, axis=0)

    v_predicho = gradIntensidad * m

    mediaDatosV = np.mean(imagenV, axis=0)
    
    # Numerador: Suma de los residuos al cuadrado
    numerador = np.sum((imagenV - v_predicho)**2, axis=0)
    
    # Denominador: Suma de la varianza total
    denominador = np.sum((imagenV - mediaDatosV)**2, axis=0)
    
    # Calculamos R^2 (NumPy resta el 1 a cada píxel automáticamente)
    mapa_r_cuadrado = 1 - (numerador / denominador)

    campoMagnetico = -m*(1/(g_eff*constanteFormula))

    return campoMagnetico, mapa_r_cuadrado

def calcularMagnetismo(imagenIntensidad, imagenV, lambdas):
    lambdas3D = lambdas[:, np.newaxis, np.newaxis]

    gradIntensidad = np.gradient(imagenIntensidad, lambdas, axis=0) * (lambdas3D**2)

    # A partir de ahora calculamos la regresionLineal
    m = np.sum(gradIntensidad*imagenV, axis=0)/np.sum(gradIntensidad**2, axis=0)

    v_predicho = gradIntensidad * m

    mediaDatosV = np.mean(imagenV, axis=0)
    
    # Numerador: Suma de los residuos al cuadrado
    numerador = np.sum((imagenV - v_predicho)**2, axis=0)
    
    # Denominador: Suma de la varianza total
    denominador = np.sum((imagenV - mediaDatosV)**2, axis=0)
    
    # Calculamos R^2 (NumPy resta el 1 a cada píxel automáticamente)
    mapa_r_cuadrado = 1 - (numerador / denominador)

    campoMagnetico = -m*(1/(g_eff*constanteFormula))

    return campoMagnetico, mapa_r_cuadrado



def dibujarMagYR(campoMagnetico, mapa_r_cuadrado):
    # Representacion
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- PRIMER RECUADRO (ax1): El Magnetograma ---
    im1 = ax1.imshow(campoMagnetico, cmap='RdBu_r') 
    fig.colorbar(im1, ax=ax1, label='Valor del campo magnético paralelo G (Gauss)') 

    # --- SEGUNDO RECUADRO (ax2): El mapa de R^2 ---
    im2 = ax2.imshow(mapa_r_cuadrado, vmin=0, vmax=1, cmap='viridis')
    ax2.set_title('Mapa de Fiabilidad (R^2)')
    fig.colorbar(im2, ax=ax2, label='R^2')


    plt.tight_layout()  #ajusta los márgenes automáticamente para que los títulos y las barras de color no se superpongan entre sí.

    plt.show()


def dibujarComparacionPSF(psf_cargada, psf_airy):
    """
    Compara la PSF cargada con la PSF de Airy en un mismo plot.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # --- PSF cargada ---
    im1 = ax1.imshow(psf_cargada, cmap='viridis')
    ax1.set_title('PSF Cargada')
    fig.colorbar(im1, ax=ax1, label='Intensidad')

    # --- PSF de Airy ---
    im2 = ax2.imshow(psf_airy, cmap='viridis')
    ax2.set_title('PSF de Airy')
    fig.colorbar(im2, ax=ax2, label='Intensidad')

    # --- Diferencia ---
    im3 = ax3.imshow(psf_cargada - psf_airy, cmap='RdBu_r')
    ax3.set_title('Diferencia (Cargada - Airy)')
    fig.colorbar(im3, ax=ax3, label='Diferencia')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Cargar la imagen FITS
    # 1. Cargamos los datos
    ruta_archivo = 'data/prueba.fits'
    with fits.open(ruta_archivo) as hdul:
        datos = hdul[0].data
        cabecera = hdul[0].header

    # Extraemos el eje lambda de la cabecera
    eje_lambda = np.array([cabecera[f'L_{i}'] for i in range(datos.shape[0])])

    psf = np.load("data/PSF_517_1600_x_1600_px.npy")
    print(f"Tamaño de PSF: {psf.shape}")
    print(f"Tamaño de imagen original: {datos.shape}")

    psf = psf / np.sum(psf)  # Normalizamos la PSF para que su suma sea 1
    
    # Recortar la imagen a 1600x1600 (centrada)
    target_size = psf.shape[0]  # 1600
    start = (datos.shape[2] - target_size) // 2
    datos = datos[:, :, start:start+target_size, start:start+target_size]
    print(f"Tamaño de imagen después de recorte: {datos.shape}")

    intensidad_orig = datos[:, 0, :, :]
    V_orig = datos[:, 3, :, :]

    # 1. Generamos la PSF de Airy con las dimensiones recortadas
    psf_airy = decon.psfAiry(intensidad_orig[0])

    print("--- 1. Original (Sin Deconvolución) ---")
    campo_orig, r2_orig = calcularMagnetismo(intensidad_orig, V_orig, eje_lambda)

    print("--- 2. Calculando con Fourier ---")
    intensidad_fourier = decAxis0Multi(intensidad_orig.copy(), psf, metDecon='f')
    V_fourier = decAxis0Multi(V_orig.copy(), psf, metDecon='f')
    campo_fourier, r2_fourier = calcularMagnetismo(intensidad_fourier, V_fourier, eje_lambda)

    print("--- 3. Calculando con Wiener Básico (PSF cargada) ---")
    intensidad_basico = decAxis0Multi(intensidad_orig.copy(), psf, metDecon='w')
    V_basico = decAxis0Multi(V_orig.copy(), psf, metDecon='w')
    campo_basico, r2_basico = calcularMagnetismo(intensidad_basico, V_basico, eje_lambda)
    
    # === METODO DE FRAN OMITIDO ===
    # print("--- Calculando con Wiener Fran (Zernikes) ---")
    
    print("--- 4. Calculando con Wiener Scikit-Image (Manual) ---")
    intensidad_scikit = decAxis0Multi(intensidad_orig.copy(), psf, metDecon='w_skimage')
    V_scikit = decAxis0Multi(V_orig.copy(), psf, metDecon='w_skimage')
    campo_scikit, r2_scikit = calcularMagnetismo(intensidad_scikit, V_scikit, eje_lambda)

    print("--- 5. Calculando con Richardson-Lucy ---")
    intensidad_rl = decAxis0Multi(intensidad_orig.copy(), psf, metDecon='rl', iteraciones=30)
    V_rl = decAxis0Multi(V_orig.copy(), psf, metDecon='rl', iteraciones=30, k=1e-10)
    campo_rl, r2_rl = calcularMagnetismo(intensidad_rl, V_rl, eje_lambda)

    print("--- Generando plots comparativos ---")
    fig, axs = plt.subplots(5, 3, figsize=(18, 25))
    fig.canvas.manager.set_window_title('Comparativa Métodos de Deconvolución')

    ylim_recorte = (500, 700)
    xlim_recorte = (600, 800)

    def plot_row(row_idx, int_img, mag_img, r2_img, title_prefix):
        im = axs[row_idx, 0].imshow(int_img[0], cmap='hot', origin='lower')
        axs[row_idx, 0].set_title(f'Intensidad ({title_prefix})')
        axs[row_idx, 0].set_ylim(ylim_recorte)
        axs[row_idx, 0].set_xlim(xlim_recorte)
        fig.colorbar(im, ax=axs[row_idx, 0])

        im = axs[row_idx, 1].imshow(mag_img, cmap='RdBu_r')
        axs[row_idx, 1].set_title(f'Campo Magnético ({title_prefix})')
        axs[row_idx, 1].set_ylim(ylim_recorte)
        axs[row_idx, 1].set_xlim(xlim_recorte)
        fig.colorbar(im, ax=axs[row_idx, 1])

        im = axs[row_idx, 2].imshow(r2_img, vmin=0, vmax=1, cmap='viridis')
        axs[row_idx, 2].set_title(f'R^2 ({title_prefix})')
        axs[row_idx, 2].set_ylim(ylim_recorte)
        axs[row_idx, 2].set_xlim(xlim_recorte)
        fig.colorbar(im, ax=axs[row_idx, 2])

    plot_row(0, intensidad_orig, campo_orig, r2_orig, 'Original')
    plot_row(1, intensidad_fourier, campo_fourier, r2_fourier, 'Fourier')
    plot_row(2, intensidad_basico, campo_basico, r2_basico, 'Wiener Bás.')
    plot_row(3, intensidad_scikit, campo_scikit, r2_scikit, 'Scikit')
    plot_row(4, intensidad_rl, campo_rl, r2_rl, 'Richardson-Lucy')

    fig.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
    plt.savefig('output/comparativa_deconvolucion.png')

