from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import deconvolucion as decon


g_eff = 1.75 # Linea del magnesio I
constanteFormula = 4.67e-13 

def decWienerAxis0(imagen, psf, k=1e-3):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionWiener(imagen[i], psf, k)

    return imagen

def decWienerAxis0Multi(imagen, psf, k=1e-3):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionWienerMulti(imagen[i], psf, k)

    return imagen

def decRLAxis0(imagen, psf, iteraciones=500, k=1e-3, epsilon=1):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionRL(imagen[i], psf, iteraciones, k, epsilon)

    return imagen

def decRLAxis0Multi(imagen, psf, iteraciones=500, k=1e-3, epsilon=1):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionRLMulti(imagen[i], psf, iteraciones, k, epsilon)

    return imagen

def decFourierAxis0(imagen, psf, k=1e-3):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionFourier(imagen[i], psf, k)

    return imagen

def decFourierAxis0Multi(imagen, psf, k=1e-3):
    for i in range(imagen.shape[0]):
        print(f"Deconvolución Fourier (Multi) - Lambda {i}...")
        imagen[i] = decon.deconvolucionFourierMulti(imagen[i], psf, k)

    return imagen

def decAxis0Multi(imagen, psf, metDecon='w', iteraciones=30, k=1e-3, epsilon=1):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionMulti(imagen[i], psf, metDecon, iteraciones, k, epsilon)


def magnetismoDirectamente(imagen, psf, lambdas, metDecon='wiener', iteraciones=1000, k=1e-3, epsilon=1):
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


def magnetismoDirectamenteMulti(imagen, psf, lambdas, metDecon='wiener', iteraciones=1000, k=1e-3, epsilon=1): # En este calculamos la deconvolucion de V y de I directamente, luego sacamos por regresion lineal el magnetismo
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

    intensidad = datos[:, 0, :, :]
    V = datos[:, 3, :, :]

    intensidad = decAxis0Multi(intensidad, psf, metDecon='w', k=1e-3)
    V = decAxis0Multi(V, psf, metDecon='w', k=1e-3)

    campoMagnetico, mapa_r_cuadrado = calcularMagnetismo(intensidad, V, eje_lambda)

    

    dibujarMagYR(campoMagnetico, mapa_r_cuadrado)

