from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as sp_fft
from scipy.signal import fftconvolve
from scipy.special import j1
import numexpr as ne
import mod.shift_func
import mod.zernike
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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

def psfAiry(datos, escala=1.32/3):
    # 1. Dimensiones y creación de la malla (igual que en la gaussiana)
    nx = datos.shape[1]
    ny = datos.shape[0]
    ejeX = np.arange(-nx // 2, nx // 2)
    ejeY = np.arange(-ny // 2, ny // 2)
    X, Y = np.meshgrid(ejeX, ejeY, indexing='ij')

    R = np.sqrt(X**2 + Y**2) * escala

    termino_interno = np.ones_like(R, dtype=float)
    np.divide(2 * j1(R), R, out=termino_interno, where=R != 0)
    
    # 4. Intensidad (al cuadrado)
    psf = termino_interno**2
    
    psf = psf - psf.min() # Forzamos que el fondo sea 0 absoluto
    psf = psf / np.sum(psf)
    
    return psf



def psfNiandrejo(datos, escala=1.0):
    # 1. Obtenemos las dimensiones espaciales correctas
    nx = datos.shape[1]
    ny = datos.shape[0]
    
    # 2. Creamos los vectores centrados
    ejeX = np.arange(-nx // 2, nx // 2)
    ejeY = np.arange(-ny // 2, ny // 2)
    
    # 3. Creamos la malla 2D
    X, Y = np.meshgrid(ejeX, ejeY, indexing='ij')
    
    # 4. Calculamos la distancia al cuadrado
    distancia_cuadrada = (X**2 + Y**2)
    
    # 5. Aplicamos el perfil Lorentziano usando 'escala' para ensanchar el montículo
    # Al hacer esto, ya NO necesitamos usar np.where ni corregir el centro a mano,
    # porque el centro da exactamente 1.0 de forma natural.
    psf = 1 / (1 + (distancia_cuadrada / escala**2))

    psf = np.where(distancia_cuadrada == 0, 1, 1 / distancia_cuadrada)
    
    # 6. Normalizamos la PSF (que la suma de toda la luz sea 1)
    psf /= np.sum(psf)
    
    return psf


def deconvolucionFourier_paralela(imagen, psf, epsilon=1e-15):
    # Centramos la PSF
    psf_preparada = np.fft.ifftshift(psf)
    
    # 1. FFT de la PSF (paralelizada en todos los núcleos)
    H = sp_fft.fft2(psf_preparada, workers=-1)
    
    # 2. FFT de la imagen 4D (paralelizada en todos los núcleos)
    # resultado_complejoworkers=-1 le dice que use el 100% de tu CPU
    G = sp_fft.fft2
    # 3. División matemática (paralelizada con NumExpr)
    # Ne.evaluate compila la fórmula y la divide entre los hilos de la CPU
    X_fourier = ne.evaluate("G / (H + epsilon)")
    
    # 4. Inversa de Fourier (paralelizada en todos los núcleos)
    resultado_complejo = sp_fft.ifft2(X_fourier, workers=-1)
    
    return np.real(resultado_complejo)


def prepararFourier(imagen, psf):
    psf_preparada = np.fft.ifftshift(psf) # Se supone que esto lo arregla
    psf_Fourier = np.fft.fft2(psf_preparada)

    imagenFourier = np.fft.fft2(imagen)

    return imagenFourier, psf_Fourier

def prepararFourierMulti(imagen, psf, workers=-1):
    psf_preparada = np.fft.ifftshift(psf)
    psf_furier = sp_fft.fft2(psf_preparada, workers=workers)

    imagenFourier = sp_fft.fft2(imagen, workers=workers)

    return imagenFourier, psf_furier


def deconvolucionFourier(imagen, psf, k=1e-3):
    # Hay que preparar la psf porque resulta que aunque la PSF esta centrada en cero
    # el algoritmo de fft no lo toma como en cero, sino que tiene que tomarlo a la izquierda del todo
    imagenFourier, psfFurier = prepararFourier(imagen, psf)

    epsilon = k
    #epsilon = 0.05
    X_fourier = imagenFourier / (psfFurier + epsilon)

    resultado_complejo = np.fft.ifft2(X_fourier)

    return np.real(resultado_complejo)

def deconvolucionFourierMulti(imagen, psf, k=1e-3, workers=-1):
    # Hay que preparar la psf porque resulta que aunque la PSF esta centrada en cero
    # el algoritmo de fft no lo toma como en cero, sino que tiene que tomarlo a la izquierda del todo
    imagenFourier, psfFurier = prepararFourierMulti(imagen, psf, workers=workers)

    epsilon = k
    #epsilon = 0.05
    X_fourier = ne.evaluate("imagenFourier / (psfFurier + epsilon)")

    resultado_complejo = sp_fft.ifft2(X_fourier, workers=workers)

    return np.real(resultado_complejo)


def deconvolucionWiener(imagen, psf, k=1e-3):
    imagenFourier, psfFourier = prepararFourier(imagen, psf)

    # A partir de ahora construyo el filtro de Weiner
    numerador = np.conjugate(psfFourier)

    denominador = np.abs(psfFourier)**2 + k

    filtro = numerador/denominador

    X_fourier = imagenFourier * filtro

    resultado_complejo = np.fft.ifft2(X_fourier)

    return np.real(resultado_complejo)

def deconvolucionWienerMulti(imagen, psf, k=1e-3, workers=-1):
    imagenFourier, psfFourier = prepararFourierMulti(imagen, psf, workers=workers)

    # A partir de ahora construyo el filtro de Weiner
    numerador = np.conjugate(psfFourier)

    denominador = ne.evaluate('abs(psfFourier)**2 + k')

    filtro = ne.evaluate('numerador/denominador')

    X_fourier = ne.evaluate('imagenFourier * filtro')

    resultado_complejo = sp_fft.ifft2(X_fourier, workers=workers)
    
    return np.real(resultado_complejo)

def deconvolucionWienerFran(imagen, zernikes):
    """
    Deconvolución de Wiener avanzada propuesta por Fran.
    Llama a restore_ima del módulo pd_functions_v22 usando los coeficientes de Zernike.
    """
    import sys
    import os
    # Aseguramos que el módulo se pueda importar correctamente
    dir_path = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.join(dir_path, 'mod')
    if mod_path not in sys.path:
        sys.path.append(mod_path)
    
    try:
        import mod.pd_functions_v22 as pdf
    except ModuleNotFoundError as e:
        print(f"\\n⚠️ AVISO: No se puede ejecutar el método de Fran. Falta una dependencia de su código personal: {e}")
        print("Asegúrate de pedirle a Fran los archivos 'shift_func.py' y 'zernike.py' y ponlos en la carpeta 'src/mod/'.")
        print("Devolviendo imagen en blanco (ceros) para no interrumpir el resto de procesos...")
        return np.zeros_like(imagen)
    
    try:
        # La función devuelve la imagen restaurada y el filtro de ruido empleado.
        # Solo necesitamos la imagen reconstruida.
        imagen_restaurada, filtro = pdf.restore_ima(imagen, zernikes)
        return np.real(imagen_restaurada)
    except AttributeError as e:
        # Caso habitual: se ha importado el paquete pip "zernike" en lugar
        # del módulo local esperado por el código de Fran (función zernike()).
        if "module 'zernike' has no attribute 'zernike'" in str(e):
            print("\n⚠️ AVISO: Incompatibilidad con el módulo 'zernike'.")
            print("El método de Fran espera un archivo local 'zernike.py' (en src/mod/) con la función 'zernike'.")
            print("El paquete instalado por pip llamado 'zernike' no expone esa función con ese nombre.")
            print("Devolviendo imagen en blanco (ceros) para no interrumpir el resto de procesos...")
            return np.zeros_like(imagen)
        raise

def deconvolucionWienerFranMulti(imagen, zernikes, workers=-1):
    """
    Deconvolución de Wiener avanzada propuesta por Fran.
    Versión multinúcleo empleando monkey-patching para forzar workers.
    Llama a restore_ima del módulo pd_functions_v22 usando los coeficientes de Zernike.
    """
    import sys
    import os
    import scipy.fft as sp_fft
    import scipy.fftpack as sp_fftpack
    from functools import partial

    # Aseguramos que el módulo se pueda importar correctamente
    dir_path = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.join(dir_path, 'mod')
    if mod_path not in sys.path:
        sys.path.append(mod_path)
    
    try:
        import mod.pd_functions_v22 as pdf
        import mod.math_func2 as mf
    except ModuleNotFoundError as e:
        print(f"\n⚠️ AVISO: No se puede ejecutar el método de Fran. Falta una dependencia de su código personal: {e}")
        return np.zeros_like(imagen)

    # 1. Guardamos las funciones originales
    original_np_fft2 = np.fft.fft2
    original_np_ifft2 = np.fft.ifft2
    original_sp_fft2 = sp_fftpack.fft2
    original_sp_ifft2 = sp_fftpack.ifft2
    original_pdf_fft2 = pdf.fft2
    original_pdf_ifft2 = pdf.ifft2
    original_mf_fft2 = mf.fft2
    original_mf_ifft2 = mf.ifft2

    # 2. Creamos funciones parciales forzando multinúcleo o número específico de hilos
    multi_fft2 = partial(sp_fft.fft2, workers=workers)
    multi_ifft2 = partial(sp_fft.ifft2, workers=workers)

    try:
        # 3. Aplicamos el Monkey-patch
        np.fft.fft2 = multi_fft2
        np.fft.ifft2 = multi_ifft2
        sp_fftpack.fft2 = multi_fft2
        sp_fftpack.ifft2 = multi_ifft2
        pdf.fft2 = multi_fft2
        pdf.ifft2 = multi_ifft2
        mf.fft2 = multi_fft2
        mf.ifft2 = multi_ifft2

        # 4. Ejecutamos la función original de Fran
        try:
            imagen_restaurada, filtro = pdf.restore_ima(imagen, zernikes)
            return np.real(imagen_restaurada)
        except AttributeError as e:
            if "module 'zernike' has no attribute 'zernike'" in str(e):
                print("\n⚠️ AVISO: Incompatibilidad con el módulo 'zernike'.")
                return np.zeros_like(imagen)
            raise
    finally:
        # 5. Restauramos las funciones originales, incluso si hay error
        np.fft.fft2 = original_np_fft2
        np.fft.ifft2 = original_np_ifft2
        sp_fftpack.fft2 = original_sp_fft2
        sp_fftpack.ifft2 = original_sp_ifft2
        pdf.fft2 = original_pdf_fft2
        pdf.ifft2 = original_pdf_ifft2
        mf.fft2 = original_mf_fft2
        mf.ifft2 = original_mf_ifft2

def deconvolucionWienerScikit(imagen, psf, balance=1e-3):
    """
    Implementación nativa oficial de la librería scikit-image del filtro de Wiener,
    utilizando el módulo recién instalado por el usuario.
    """
    from skimage import restoration
    
    # 1. NORMALIZACIÓN DE ENERGÍA (Por qué se dispara a 35000):
    # scikit-image evalúa la frecuencia DC (frecuencia 0, intensidad media) del filtro. 
    # Por defecto, skimage asume que la integral de luz de la PSF es exactamente 1.0. 
    # Si nuestra tu PSF suma un valor muy pequeño (ej. 0.025), la librería asume que 
    # la óptica cortó ese diferencial de luz y multiplica toda la foto final por casi x40!
    # Solución: Normalizamos la suma ponderada del núcleo a 1.0:
    psf_norm = psf / np.sum(psf)
    
    # IMPORTANTE: skimage usa un filtro Laplaciano (corte agresivo de altas frecuencias) 
    # como regularizador por defecto, no una constante K plana como tu método Básico. 
    # Un balance de 1e-3 en el Laplaciano emborrona la imagen muchísimo más porque ataca más rápido las siluetas.
    # Le pasamos la PSF limpia y normalizada:
    imagen_restaurada = restoration.wiener(imagen, psf_norm, balance=balance, clip=False)
    
    return imagen_restaurada


def deconvolucionRL(imagen, psf, pasos = 1000, k=1e-3, epsilon=1):
    psf_preparada = np.fft.ifftshift(psf)
    psfFourier = np.fft.fft2(psf_preparada)

    psfInvertido = np.flip(psf_preparada)
    psfInvFurier = np.fft.fft2(psfInvertido)
    
    o_ene = imagen

    for i in range(pasos): #sustituyo para ahorrar el espacio
        #print(f'Paso {i}')
        o_ene = np.fft.fft2(o_ene)
        denominador = o_ene * psfFourier
        denominador = np.real(np.fft.ifft2(denominador))
        o_ene = np.real(np.fft.ifft2(o_ene))

        fraccion = imagen/(denominador + k)
        del denominador

        fraccion = np.fft.fft2(fraccion)
        fraccion = fraccion * psfInvFurier
        fraccion = np.real(np.fft.ifft2(fraccion))

        o_ene1 = o_ene * fraccion
        del fraccion

        valor = np.mean((o_ene1 - o_ene)**2)
        #print(valor)
        o_ene = o_ene1
        del o_ene1

        if (valor <= epsilon):
            print(f'Finalizado prematuramente, i={i}')
            break
    
    return o_ene

def deconvolucionRLMulti(imagen, psf, pasos=1000, k=1e-3, epsilon=1, workers=-1):
    # Usamos sp_fft para las preparaciones iniciales
    psf_preparada = sp_fft.ifftshift(psf)
    psfFourier = sp_fft.fft2(psf_preparada, workers=workers)

    psfInvertido = np.flip(psf_preparada)
    psfInvFurier = sp_fft.fft2(psfInvertido, workers=workers)
    
    # Hacemos una copia para no alterar la imagen original por referencia
    o_ene = np.copy(imagen)

    for i in range(pasos):
        #print(f'Paso {i}')
        
        # 1. Transformamos o_ene sin sobrescribir la variable original espacial
        o_ene_fourier = sp_fft.fft2(o_ene, workers=workers)
        
        # 2. Calculamos el denominador en frecuencia y volvemos al dominio espacial
        denominador_fourier = ne.evaluate('o_ene_fourier * psfFourier')
        denominador = np.real(sp_fft.ifft2(denominador_fourier, workers=workers))

        # 3. Calculamos la fracción usando numexpr
        fraccion = ne.evaluate('imagen / (denominador + k)')
        del denominador

        # 4. Pasamos la fracción a frecuencia, multiplicamos y volvemos al espacio
        fraccion_fourier = sp_fft.fft2(fraccion, workers=workers)
        fraccion_fourier = ne.evaluate('fraccion_fourier * psfInvFurier')
        fraccion = np.real(sp_fft.ifft2(fraccion_fourier, workers=workers))

        # 5. Calculamos la nueva estimación de la imagen
        o_ene1 = ne.evaluate('o_ene * fraccion')
        del fraccion

        # 6. Calculamos el error cuadrático medio (MSE) con numexpr para la resta
        # np.mean es muy eficiente por sí solo, pero vectorizamos la resta y la potencia
        diferencia_cuadrada = ne.evaluate('(o_ene1 - o_ene)**2')
        denominador = ne.evaluate('(o_ene)**2')

        valor = np.sum(diferencia_cuadrada)/np.sum(denominador)
        #print(valor)
        
        o_ene = o_ene1
        del o_ene1

        if valor <= epsilon:
            print(f'Finalizado prematuramente, i={i}, valor = {valor}')
            break

    print(f'Finalizado a los {pasos} pasos')
    
    return o_ene

def deconvolucionMulti(imagen, psf=None, metodo='rl', k=1e-3, iteraciones=30, epsilon=1, zernikes=None, workers=-1):
    # Aceptamos alias de nombre largo para evitar incompatibilidades entre módulos.
    mapa_metodos = {
        'wiener': 'w',
        'fourier': 'f',
        'rl': 'rl',
        'richardson-lucy': 'rl',
        'richardson_lucy': 'rl',
        'w': 'w',
        'f': 'f',
        'w_fran': 'w_fran',
        'w_skimage': 'w_skimage',
    }
    metodo = mapa_metodos.get(str(metodo).lower(), metodo)

    if (imagen < 0).any():
        imagenPos = np.where(imagen <= 0, 0, imagen)
        imagenNeg = np.abs(np.where(imagen < 0, imagen, 0))

        if metodo == 'rl':
            imagenPos = deconvolucionRLMulti(imagenPos, psf, iteraciones, k, epsilon, workers=workers)
            imagenNeg = deconvolucionRLMulti(imagenNeg, psf, iteraciones, k, epsilon, workers=workers)
        elif metodo == 'f':
            imagenPos = deconvolucionFourierMulti(imagenPos, psf, k, workers=workers)
            imagenNeg = deconvolucionFourierMulti(imagenNeg, psf, k, workers=workers)
        elif metodo == 'w':
            imagenPos = deconvolucionWienerMulti(imagenPos, psf, k, workers=workers)
            imagenNeg = deconvolucionWienerMulti(imagenNeg, psf, k, workers=workers)
        elif metodo == 'w_fran' and zernikes is not None:
            imagenPos = deconvolucionWienerFranMulti(imagenPos, zernikes, workers=workers)
            imagenNeg = deconvolucionWienerFranMulti(imagenNeg, zernikes, workers=workers)
        elif metodo == 'w_skimage' and psf is not None:
            imagenPos = deconvolucionWienerScikit(imagenPos, psf, balance=k)
            imagenNeg = deconvolucionWienerScikit(imagenNeg, psf, balance=k)
        else:
            raise ValueError(f"Método de deconvolución no soportado: {metodo}")

        return imagenPos - imagenNeg
    else:
        if metodo == 'rl':
            return deconvolucionRLMulti(imagen, psf, iteraciones, k, epsilon, workers=workers)
        elif metodo == 'f':
            return deconvolucionFourierMulti(imagen, psf, k, workers=workers)
        elif metodo == 'w':
            return deconvolucionWienerMulti(imagen, psf, k, workers=workers)
        elif metodo == 'w_fran' and zernikes is not None:
            return deconvolucionWienerFranMulti(imagen, zernikes, workers=workers)
        elif metodo == 'w_skimage' and psf is not None:
            return deconvolucionWienerScikit(imagen, psf, balance=k)
        else:
            raise ValueError(f"Método de deconvolución no soportado: {metodo}")
    
def _procesar_multi_generico(i_img, psf, metodo, iteraciones, epsilon, zernikes, k, workers):
    i, img = i_img
    return i, deconvolucionMulti(img, psf=psf, metodo=metodo, iteraciones=iteraciones, k=k, epsilon=epsilon, zernikes=zernikes, workers=workers)

def aplicar_deconvolucion_3d(imagen_3d, psf=None, metodo='w', iteraciones=30, epsilon=1, zernikes = np.array([0,0,0,
        0.5765,
        0.5391,
        -0.1163,
        0.121,
        0.1504,
        -0.1154,
        0.2591,
        -0.3103,
        0.1108,
        -0.1963,
        -0.0431,
        -0.2591,
        -0.5599,
        0.0904,
        0.2754,
        0.0715,
        0.0006,
        0.0862]), k=1e-3, workers=-1):
    """
    Aplica deconvolución a lo largo del eje lambda (eje 0) de un cubo 3D.
    """
    executor_workers = None if workers == -1 else workers

    if metodo in ['w_fran', 'w_fran_multi']:
        img_list = list(enumerate(imagen_3d))
        func = partial(_procesar_multi_generico, psf=psf, metodo=metodo, iteraciones=iteraciones, epsilon=epsilon, zernikes=zernikes, k=k, workers=workers)
        with ProcessPoolExecutor(max_workers=executor_workers) as executor:
            for i, res in executor.map(func, img_list):
                imagen_3d[i] = res
    else:
        for i in range(imagen_3d.shape[0]):
            imagen_3d[i] = deconvolucionMulti(imagen_3d[i], psf=psf, metodo=metodo, iteraciones=iteraciones, k=k, epsilon=epsilon, zernikes=zernikes, workers=workers)
    return imagen_3d


def probar_deconvolucion(sigma, k, tipo_psf='airy', metodo='wiener', ruta='data/prueba.fits'):
    """
    Realiza una prueba de deconvolución sobre una imagen FITS.
    
    Parámetros:
    - sigma: float, valor de sigma para la PSF.
    - k: float, constante para el filtro de deconvolución.
    - tipo_psf: str, 'gaussiana' o 'airy' (por defecto 'airy').
    - metodo: str, 'wiener' o 'fourier' (por defecto 'wiener').
    - ruta: str, ruta al archivo FITS.
    """
    
    # 1. Recogemos los datos
    datos, cabecera = recogerLosDatos(ruta)

    # 2. Extraemos la imagen original 2D (Asumiendo que es intensidad)
    imagenIntensidad = datos[0, 0, :, :]

    # 3. Calculamos la PSF basándonos en la elección del usuario
    tipo_psf = tipo_psf.lower()
    if tipo_psf == 'gaussiana':
        mi_psf = psfGaussiana(datos, sigma)
    elif tipo_psf == 'airy':
        escala = 1.37 / sigma  # La escala equivalente 
        mi_psf = psfAiry(datos, escala)
    else:
        raise ValueError("⚠️ Error: El parámetro tipo_psf debe ser 'gaussiana' o 'airy'")

    # 4. Hacemos la deconvolución según el método elegido
    metodo = metodo.lower()
    if metodo == 'wiener':
        datosArreglados = deconvolucionWiener(datos, mi_psf, k)
    elif metodo == 'fourier':
        datosArreglados = deconvolucionFourier(datos, mi_psf, k)
    else:
        raise ValueError("⚠️ Error: El parámetro metodo debe ser 'wiener' o 'fourier'")

    # 5. Extraemos la imagen arreglada 2D para poder dibujarla
    datosArregladosInt = datosArreglados[0, 0, :, :]

    # --- SECCIÓN DE DIBUJO ---
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Marco 1: Imagen Original
    axs[0].imshow(imagenIntensidad, cmap='hot', origin='lower')
    axs[0].set_title('Imagen Original (Borrosa)')
    axs[0].set_xlim(600, 800)
    axs[0].set_ylim(500, 700)

    # Marco 2: La PSF
    axs[1].imshow(mi_psf, cmap='hot', origin='lower')
    axs[1].set_title(f'PSF ({tipo_psf.capitalize()})')
    ny, nx = mi_psf.shape
    cy, cx = ny // 2, nx // 2
    axs[1].set_xlim(cx - 100, cx + 100)
    axs[1].set_ylim(cy - 100, cy + 100)

    # Marco 3: Resultado
    axs[2].imshow(datosArregladosInt, cmap='hot', origin='lower')
    axs[2].set_title(f'Deconvolución ({metodo.capitalize()})')
    axs[2].set_xlim(600, 800)
    axs[2].set_ylim(500, 700)

    plt.tight_layout()
    
    # 6. Guardamos la figura directamente en output con todos los parámetros en el nombre
    nombre_archivo = f'output/deconvolucion_{metodo}_{tipo_psf}_sigma{sigma}_k{k}.png'
    plt.savefig(nombre_archivo)
    
    # Cerramos la figura para liberar memoria
    plt.close(fig)
    
    print(f"✅ Imagen guardada en: {nombre_archivo}")
    return datosArreglados

#Para probar el algoritmo de deconvolucion
#sigma = [1, 3, 5]
#valorK = [1e-5, 1e-3, 1e-1, 1]
#tipoPsf = ['airy', 'gaussiana']
#metodo = ['fourier', 'wiener']
#
#for s in sigma:
#    for k in valorK:
#        for t in tipoPsf:
#            for m in metodo:
#                probar_deconvolucion(s, k, t, m)
#                print(f'Calculado el s:{s}, k: {k}, t: {t}, m: {m}')


if __name__ == "__main__":
    ruta_archivo = 'data/prueba.fits'

    with fits.open(ruta_archivo) as hdul:
        datos = hdul[0].data
        cabecera = hdul[0].header
    
    imagenInt = datos[0, 3, :, :]
    
    # Generamos la PSF
    miPsf = psfAiry(imagenInt)
    
    # Calculamos las 3 deconvoluciones
    print("Calculando Fourier...")
    img_fourier = deconvolucionFourierMulti(imagenInt, miPsf)
    
    print("Calculando Wiener...")
    img_wiener = deconvolucionWienerMulti(imagenInt, miPsf)
    
    print("Calculando Richardson-Lucy (esto puede tardar unos segundos)...")
    # He puesto pasos=20 para agilizar la prueba, súbelo a 100 o 1000 si necesitas más iteraciones
    img_rl = deconvolucionRLMulti(imagenInt, miPsf, epsilon=10, pasos=10) 
    
    # Ampliamos a 5 columnas y ajustamos el tamaño de la figura
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
    
    # Marco 1: Imagen Original
    axs[0].imshow(imagenInt, cmap='hot', origin='lower')
    axs[0].set_title('Imagen Original (Borrosa)')
    axs[0].set_ylim(600, 800)
    axs[0].set_xlim(500, 700)
    
    # Marco 2: La PSF
    axs[1].imshow(miPsf, cmap='hot', origin='lower')
    axs[1].set_title('PSF')
    ny, nx = miPsf.shape
    cy, cx = ny // 2, nx // 2
    axs[1].set_xlim(cx - 100, cx + 100)
    axs[1].set_ylim(cy - 100, cy + 100)
    
    # Marco 3: Deconvolución Fourier
    axs[2].imshow(img_fourier, cmap='hot', origin='lower')
    axs[2].set_title('Deconvolución (Fourier)')
    axs[2].set_ylim(600, 800)
    axs[2].set_xlim(500, 700)
    
    # Marco 4: Deconvolución Wiener
    axs[3].imshow(img_wiener, cmap='hot', origin='lower')
    axs[3].set_title('Deconvolución (Wiener)')
    axs[3].set_ylim(600, 800)
    axs[3].set_xlim(500, 700)
    
    # Marco 5: Deconvolución Richardson-Lucy
    axs[4].imshow(img_rl, cmap='hot', origin='lower')
    axs[4].set_title('Deconvolución (RL)')
    axs[4].set_ylim(600, 800)
    axs[4].set_xlim(500, 700)
    
    plt.tight_layout()
    plt.show()