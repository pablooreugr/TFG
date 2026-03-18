from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as sp_fft
import numexpr as ne

def recogerLosDatos(rutaArchivo):
    with fits.open(rutaArchivo) as hdul:
        datos = hdul[0].data
        cabecera = hdul[0].header
    
    return datos, cabecera

def psfGaussiana(datos, sigma=3.0):
    # 1. Obtenemos las dimensiones espaciales correctas (x e y)
    nx = datos.shape[3]
    ny = datos.shape[2]
    
    # 2. Creamos los vectores centrados usando división entera //
    # Esto genera, por ejemplo, de -800 a 799 para un tamaño de 1600
    ejeX = np.arange(-nx // 2, nx // 2)
    ejeY = np.arange(-ny // 2, ny // 2)
    
    # 3. Creamos la malla 2D (el lienzo)
    X, Y = np.meshgrid(ejeX, ejeY, indexing='ij')
    
    # 4. Calculamos la campana de Gauss
    psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # 5. Normalizamos la PSF (importante: que la suma de toda la luz sea 1)
    psf /= np.sum(psf)
    
    return psf

def deconvolucionFourier_paralela(imagen, psf, epsilon=1e-15):
    # Centramos la PSF
    psf_preparada = np.fft.ifftshift(psf)
    
    # 1. FFT de la PSF (paralelizada en todos los núcleos)
    H = sp_fft.fft2(psf_preparada, workers=-1)
    H_4D = H[np.newaxis, np.newaxis, :, :]
    
    # 2. FFT de la imagen 4D (paralelizada en todos los núcleos)
    # resultado_complejoworkers=-1 le dice que use el 100% de tu CPU
    G = sp_fft.fft2
    # 3. División matemática (paralelizada con NumExpr)
    # Ne.evaluate compila la fórmula y la divide entre los hilos de la CPU
    X_fourier = ne.evaluate("G / (H_4D + epsilon)")
    
    # 4. Inversa de Fourier (paralelizada en todos los núcleos)
    resultado_complejo = sp_fft.ifft2(X_fourier, axes=(2, 3), workers=-1)
    
    return np.real(resultado_complejo)


def prepararFourier(imagen, psf):
    psf_preparada = np.fft.ifftshift(psf) # Se supone que esto lo arregla
    psf_Fourier = np.fft.fft2(psf_preparada)

    imagenFourier = np.fft.fft2(imagen, axes=(2, 3))

    #Preparamos la psf fourier
    H_4D = psf_Fourier[np.newaxis, np.newaxis, :, :]

    return imagenFourier, H_4D


def deconvolucionFourier(imagen, psf):
    # Hay que preparar la psf porque resulta que aunque la PSF esta centrada en cero
    # el algoritmo de fft no lo toma como en cero, sino que tiene que tomarlo a la izquierda del todo
    imagenFourier, H_4D = prepararFourier(imagen, psf)

    epsilon = 1e-15
    #epsilon = 0.05
    X_fourier = imagenFourier / (H_4D + epsilon)

    resultado_complejo = np.fft.ifft2(X_fourier, axes=(2, 3))

    return np.real(resultado_complejo)

def deconvolucionWiener(imagen, psf, k=1e-4):
    imagenFourier, H_4D = prepararFourier(imagen, psf)

    # A partir de ahora construyo el filtro de Weiner
    numerador = np.conjugate(H_4D)

    denominador = np.abs(H_4D)**2 + k

    filtro = numerador/denominador

    X_fourier = imagenFourier * filtro

    resultado_complejo = np.fft.ifft2(X_fourier, axes=(2, 3))

    return np.real(resultado_complejo)
    



# --- A partir de aquí probamos si el código funciona ---

ruta = 'data/prueba.fits'

# 1. Recogemos los datos
datos, cabecera = recogerLosDatos(ruta)

# 2. Extraemos la imagen original 2D (Asumiendo que es intensidad)
imagenIntensidad = datos[0, 0, :, :]

# 3. Calculamos la PSF y la GUARDAMOS en una variable
# Le pasamos los datos para que coja el tamaño automáticamente
mi_psf = psfGaussiana(datos, sigma=3.0) 

# 4. Hacemos la deconvolución pasándole los datos y la PSF que acabamos de guardar
#datosArreglados = deconvolucionFourier(datos, mi_psf)
datosArreglados = deconvolucionWiener(datos, mi_psf, 1e-3)

# 5. Extraemos la imagen arreglada 2D para poder dibujarla
datosArregladosInt = datosArreglados[0, 0, :, :]

# --- SECCIÓN DE DIBUJO ---

# Creamos un lienzo grande con 1 fila y 3 columnas
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Marco 1: Imagen Original
# Usamos vmin y vmax opcionales si quieres ajustar el contraste luego, por ahora lo dejamos por defecto
axs[0].imshow(imagenIntensidad, cmap='hot', origin='lower')
axs[0].set_title('Imagen Original (Borrosa)')

# Marco 2: La PSF
# Nota del profe: Como la PSF es enorme (ej. 1644x1644) y la campana es pequeñita,
# quizás solo veas un puntito brillante en el centro. ¡Es normal!
axs[1].imshow(mi_psf, cmap='hot', origin='lower')
axs[1].set_title('PSF (Nuestra Lente)')

# Marco 3: Resultado
axs[2].imshow(datosArregladosInt, cmap='hot', origin='lower')
axs[2].set_title('Imagen Deconvolucionada')

# Ajustamos un poco el espacio para que no se pisen los títulos y mostramos
plt.tight_layout()
plt.show()