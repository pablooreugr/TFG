import numpy as np
from scipy.special import j1
from scipy import signal, fft


def generar_psf_airy(tamano_matriz, radio_piz):
    """
    Genera una PSF de Airy bidimensional.
    
    Parámetros:
    tamano_matriz (int): Dimensión de la imagen de salida (ej. 256 para 256x256).
    radio_piz (float): Radio del disco central medido en píxeles.
    """
    # 1. Crear el eje de coordenadas centrado en cero
    centro = (tamano_matriz - 1) / 2
    x = np.arange(tamano_matriz) - centro
    y = np.arange(tamano_matriz) - centro
    X, Y = np.meshgrid(x, y)
    
    # 2. Calcular la distancia radial 'r' en píxeles desde el centro
    R = np.sqrt(X**2 + Y**2)
    
    # 3. Factor de escala: el primer cero de J1 ocurre en approx 3.8317
    # Queremos que cuando R == radio_piz, el argumento sea 3.8317
    cero_airy = 3.8317059702075123
    factor_escala = cero_airy / radio_piz
    
    # Argumento de la función de Airy
    argumento = R * factor_escala
    
    # 4. Calcular la PSF evitando la división por cero en el centro (R=0)
    # Usamos np.where para manejar el centro de forma segura
    psf = np.zeros_like(argumento)
    
    # Donde el argumento es casi cero, la intensidad tiende a 1.0 (L'Hôpital)
    mascara_centro = argumento == 0
    psf[mascara_centro] = 1.0
    
    # Para el resto de píxeles, aplicamos la fórmula estándar
    psf[~mascara_centro] = (2 * j1(argumento[~mascara_centro]) / argumento[~mascara_centro])**2

    psf_normalizada = psf / np.sum(psf)
    
    return psf_normalizada

def girarPSF(psf):
    return psf[::-1, ::-1]

def hacerPadding(imagen, psf):
    pad_size = psf.shape[0] // 2
    return np.pad(imagen, pad_size, mode='reflect'), pad_size


def convolucionarDosMapas(imagen, psf, trabajadores=-1, usar_padding=True):
    if usar_padding:
        imagen_padded, pad_size = hacerPadding(imagen, psf)
        with fft.set_workers(trabajadores):
            resultado_padded = signal.fftconvolve(imagen_padded, psf, mode='same')
        return resultado_padded[pad_size:-pad_size, pad_size:-pad_size]
    else:
        with fft.set_workers(trabajadores):
            # mode='same' recorta el resultado al tamaño de la imagen original
            return signal.fftconvolve(imagen, psf, mode='same')
    
def convolucion3D(imagen, psf, trabajadores=-1, usar_padding=True):

    resultado = np.zeros_like(imagen)
    for i in range(imagen.shape[0]):
        resultado[i] = convolucionarDosMapas(imagen[i], psf, trabajadores, usar_padding=usar_padding)
    return resultado

def lucyRichason(imagen, psf, pasos=30, trabajadores=-1, eps=1e-12):
    psf_inv = girarPSF(psf)
    u_padded, pad_size = hacerPadding(imagen.astype(float).copy(), psf)
    imagen_padded, _ = hacerPadding(imagen, psf)

    for _ in range(pasos):
        denom = convolucionarDosMapas(u_padded, psf, trabajadores=trabajadores, usar_padding=False) + eps
        ratio = imagen_padded / denom
        correccion = convolucionarDosMapas(ratio, psf_inv, trabajadores=trabajadores, usar_padding=False)
        u_padded *= correccion
    
    u = u_padded[pad_size:-pad_size, pad_size:-pad_size]
    return np.clip(u, 0, None)

def alinear_psf_fft(psf, shape):
    psf_padded = np.zeros(shape)
    psf_padded[:psf.shape[0], :psf.shape[1]] = psf
    return np.roll(psf_padded, shift=(-psf.shape[0]//2, -psf.shape[1]//2), axis=(0, 1))

def decoFourier(imagen, psf, eps=1e-12, trabajadores=-1):
    imagen_fft = fft.fft2(imagen, workers=trabajadores)
    psf_alineada = alinear_psf_fft(psf, imagen.shape)
    psf_fft = fft.fft2(psf_alineada, workers=trabajadores)

    imagen_deco_fft = imagen_fft / (psf_fft + eps)
    imagen_deco = np.real(fft.ifft2(imagen_deco_fft, workers=trabajadores))
    return np.clip(imagen_deco, 0, None)

def decoWiener(imagen, psf, snr, eps=1e-12, trabajadores=-1):
    # Hacemos el padding una sola vez (asegúrate de que hacerPadding devuelve el modo 'reflect')
    imagen_padded, pad_size = hacerPadding(imagen.astype(float), psf)

    # 1. FFT de la imagen
    imagen_fft = fft.fft2(imagen_padded, workers=trabajadores)
    
    # 2. Alinear y preparar la PSF
    psf_alineada = alinear_psf_fft(psf, imagen_padded.shape)
    
    # 3. FFT de la PSF (ya alineada por la función anterior)
    psf_fft = fft.fft2(psf_alineada, workers=trabajadores)

    # 4. Cálculo del filtro de Wiener
    psf_conj = np.conj(psf_fft)
    wiener_filter = psf_conj / (np.abs(psf_fft)**2 + (1 / (snr**2)) + eps)

    # 5. Aplicar filtro y volver al dominio espacial
    imagen_deco_fft = imagen_fft * wiener_filter
    imagen_deco_padded = np.real(fft.ifft2(imagen_deco_fft, workers=trabajadores))
    
    # 6. Recortar el padding y asegurar que no hay valores negativos
    imagen_deco = imagen_deco_padded[pad_size:-pad_size, pad_size:-pad_size]
    
    return np.clip(imagen_deco, 0, None)


def estimate_snr_empirical(image):
    # Estimación del ruido usando la desviación estándar de la diferencia local.
    # Esto reduce gran parte de la estructura de baja frecuencia.
    image = np.asarray(image, dtype=float)

    if image.ndim < 2 or image.shape[1] < 2:
        raise ValueError("La imagen debe tener al menos 2 columnas para estimar el ruido.")

    diff = image[:, 1:] - image[:, :-1]
    sigma_noise = np.std(diff) / np.sqrt(2)

    # Señal media de la imagen
    mean_signal = np.mean(image)

    snr = mean_signal / (sigma_noise + 1e-12)
    return snr, sigma_noise


def deconvolucion(imagen, psf, metodo='rl', pasos=30, trabajadores=-1):
    if metodo == 'rl':
        return lucyRichason(imagen, psf, pasos=pasos, trabajadores=trabajadores)
    elif metodo == 'fourier':
        return decoFourier(imagen, psf, trabajadores=trabajadores)
    elif metodo == 'wiener':
        snr, _ = estimate_snr_empirical(imagen)
        return decoWiener(imagen, psf, snr, trabajadores=trabajadores)
    else:
        print('El método no existe')


def deconvolucion3D(imagen, psf, metodo='rl', pasos=30, trabajadores=-1):

    imagenDecon = np.zeros_like(imagen)
    for i in range(imagen.shape[0]):
        #print(f'paso {i}')
        imagenDecon[i] = deconvolucion(imagen[i], psf, metodo=metodo, pasos=pasos, trabajadores=trabajadores)
    return imagenDecon


if __name__ == "__main__":
    print('hola')