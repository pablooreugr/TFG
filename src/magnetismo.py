import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import deconvolucion as decon
import visualizacion as vis

g_eff = 1.75 # Linea del magnesio I
constanteFormula = 4.67e-13 

def calcularMagnetismo(imagenIntensidad, imagenV, lambdas):
    """
    Calcula el campo magnético a partir de las imágenes de Intensidad y Stokes V.
    """
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

def calcularMagnetismoConDeconvolucion(imagen, psf, lambdas, metDecon='w', iteraciones=1000, epsilon=1, k=1e-3, zernikes=None, workers=-1):
    """
    Realiza la deconvolución de las componentes I y V de la imagen, y luego calcula el campo magnético.
    """
    imagenIntensidad = imagen[:, 0, :, :]
    imagenV = imagen[:, 3, :, :]

    # Usamos una copia para no sobreescribir la imagen original en memoria
    imagenIntensidad_dec = decon.aplicar_deconvolucion_3d(imagenIntensidad.copy(), psf=psf, metodo=metDecon, iteraciones=iteraciones, epsilon=epsilon, k=k, zernikes=zernikes, workers=workers)
    imagenV_dec = decon.aplicar_deconvolucion_3d(imagenV.copy(), psf=psf, metodo=metDecon, iteraciones=iteraciones, epsilon=epsilon, k=k, zernikes=zernikes, workers=workers)

    campoMagnetico, mapa_r_cuadrado = calcularMagnetismo(imagenIntensidad_dec, imagenV_dec, lambdas)

    return campoMagnetico, mapa_r_cuadrado

if __name__ == "__main__":
    # Cargar la imagen FITS
    ruta_archivo = 'data/prueba.fits'
    with fits.open(ruta_archivo) as hdul:
        datos = hdul[0].data
        cabecera = hdul[0].header

    # Extraemos el eje lambda de la cabecera
    eje_lambda = np.array([cabecera[f'L_{i}'] for i in range(datos.shape[0])])

    try:
        psf_cargada = np.load("data/PSF_517_1600_x_1600_px.npy")
        print(f"Tamaño de PSF cargada: {psf_cargada.shape}")
        print(f"Tamaño de imagen original: {datos.shape}")

        psf_cargada = psf_cargada / np.sum(psf_cargada)  # Normalizamos la PSF para que su suma sea 1
        
        # Recortar la imagen a 1600x1600 (centrada)
        target_size = psf_cargada.shape[0]  # 1600
        start = (datos.shape[2] - target_size) // 2
        datos = datos[:, :, start:start+target_size, start:start+target_size]
        
        intensidad_orig = datos[:, 0, :, :]
        V_orig = datos[:, 3, :, :]

        psf = psf_cargada

        # Aplicamos deconvolución
        intensidad_dec = decon.aplicar_deconvolucion_3d(intensidad_orig, psf=psf, metodo='rl', iteraciones=30, workers=-1, k=1e-7)
        V_dec = decon.aplicar_deconvolucion_3d(V_orig, psf=psf, metodo='rl', iteraciones=30, workers=-1, k=1e-7)

        campoMagnetico, mapa_r_cuadrado = calcularMagnetismo(intensidad_dec, V_dec, eje_lambda)

        vis.dibujarMagYR(campoMagnetico, mapa_r_cuadrado)
    except FileNotFoundError:
        print("Aviso: No se encontró 'data/PSF_517_1600_x_1600_px.npy'. La prueba no se puede ejecutar por completo.")