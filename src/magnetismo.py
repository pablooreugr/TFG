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


def cargar_datos_y_psf(ruta_fits='data/prueba.fits', ruta_psf='data/PSF_517_1600_x_1600_px.npy'):
    """Carga datos desde un FITS y la PSF, recorta la imagen para ajustar al tamaño de la PSF y devuelve elementos útiles."""
    with fits.open(ruta_fits) as hdul:
        datos = hdul[0].data
        cabecera = hdul[0].header

    eje_lambda = np.array([cabecera[f'L_{i}'] for i in range(datos.shape[0])])
    psf_cargada = np.load(ruta_psf)
    print(f"Tamaño de PSF cargada: {psf_cargada.shape}")
    print(f"Tamaño de imagen original: {datos.shape}")

    psf_cargada = psf_cargada / np.sum(psf_cargada)
    target_size = psf_cargada.shape[0]
    start = (datos.shape[2] - target_size) // 2
    datos_recortados = datos[:, :, start:start+target_size, start:start+target_size]

    intensidad_orig = datos_recortados[:, 0, :, :]
    V_orig = datos_recortados[:, 3, :, :]
    psf_fran = psf_cargada

    return datos_recortados, cabecera, eje_lambda, intensidad_orig, V_orig, psf_fran

if __name__ == "__main__":

    datos, cabecera, eje_lambda, intensidad_orig, V_orig, psf_fran = cargar_datos_y_psf()

    #campoMagneticoSD, mapa_r_cuadradoSD = calcularMagnetismo(intensidad_orig, V_orig, eje_lambda)
    #campoMagneticoDec, mapa_r_cuadradoDec = calcularMagnetismoConDeconvolucion(datos, psf_fran, eje_lambda, metDecon='rl', workers=-1, iteraciones=50)

    intesidadDecon = decon.deconvolucionMulti(intensidad_orig[1, :, :], psf_fran, metodo='rl', iteraciones=100, workers=-1)


    
    vis.compararMagnetogramas(intensidad_orig[1, :, :], intesidadDecon)