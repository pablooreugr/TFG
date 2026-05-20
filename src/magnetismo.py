import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import deconvolucion as decon
from mod.pd_functions_v22 import PSF
import visualizacion as vis
from scipy.sparse.linalg import LinearOperator, cg
import time

g_eff = 1.75 # Linea del magnesio I
constanteFormula = 4.67e-13 

class MonitorKrylov:
    def __init__(self):
        self.iteracion = 0
        self.t_inicio = time.time()
        self.x_anterior = None
        self.historial_pasos = [] # Por si luego quieres graficarlo

    def __call__(self, xk):
        self.iteracion += 1
        t_actual = time.time() - self.t_inicio
        
        # Calculamos cuánto ha cambiado el mapa respecto a la iteración anterior
        if self.x_anterior is not None:
            cambio = np.linalg.norm(xk - self.x_anterior)
            self.historial_pasos.append(cambio)
            
            # Imprimimos en la terminal sobrescribiendo la misma línea
            print(f"Iteración {self.iteracion:03d} | Tiempo: {t_actual:.1f} s | Magnitud del paso: {cambio:.2e}", end='\r')
        else:
            print(f"Iteración {self.iteracion:03d} | Tiempo: {t_actual:.1f} s | Magnitud del paso: Calculando...", end='\r')
            
        # Guardamos el estado actual para la siguiente iteración
        self.x_anterior = xk.copy()

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


# Este método es el metodo del que me basar en el paper de Van noot
def metodoForw(intensidad, V, lambdas, psf, trabajadores=-1):
    nx = intensidad.shape[2]
    ny = intensidad.shape[1]
    n_lambda = intensidad.shape[0]

    n_total = nx * ny

    # Primero vamos a intentar preparar los datos antes del método
    campoMagneticoInicial = calcularMagnetismo(intensidad, V, lambdas)[0]

    intensidad_decon = decon.aplicar_deconvolucion_3d(intensidad, psf=psf, metodo='rl', workers=-1)

    derivadaI = np.gradient(intensidad_decon, lambdas, axis=0)
    K_cubo = -constanteFormula * g_eff * derivadaI * (lambdas[:, np.newaxis, np.newaxis]**2)

    psf_tilde = psf[::-1, ::-1]

    # Calculo la V inicial
    V_inicial = K_cubo * campoMagneticoInicial[np.newaxis, :, :]
    deltaV = V - V_inicial

    # Definimos los operadores lineales que usaremos.
    def J(dB_2D):
        v_ideal = K_cubo * dB_2D[np.newaxis, :, :]
        # Creamos una matriz vacía nueva para no pisar la anterior
        v_degradado = np.zeros_like(v_ideal)

        for i in range(n_lambda):
            v_degradado[i, :, :] = decon.convolucion(v_ideal[i, :, :], psf, trabajadores=trabajadores)
        
        return v_degradado
    
    # Operador adjunto
    def JT(residuos3d):
        for i in range(n_lambda):
            residuos3d[i, :, :] = decon.convolucion(residuos3d[i, :, :], psf_tilde, trabajadores=trabajadores)

        residuos3d = residuos3d * K_cubo

        dB_2D = np.sum(residuos3d, axis=0)

        return dB_2D
    
    def funcionA(x_1D):
        x_2D = x_1D.reshape((nx, ny))

        #Aplicamos la cadena de operadores J^T(J(X))
        efecto_telescopio = J(x_2D)
        correccion_estimada = JT(efecto_telescopio)

        return correccion_estimada.flatten()
    
    # Ahora preparamos el sistema para que aparezca el A*x = b
    b_2D = JT(deltaV)

    b_1D = b_2D.flatten()

    #Y creamos la matriz con el operador
    matrizA = LinearOperator((n_total, n_total), matvec=funcionA)

    # A partir de aqui es donde ocurre la solucion del sistema

    print('Iniciando sistema de inversion')

    monitor = MonitorKrylov()

    dB_1d_solucion, info = cg(matrizA, b_1D, rtol=1e-5, callback=monitor)

    if info == 0:
        print("¡Convergencia exitosa!")
    elif info > 0:
        print(f"Alcanzado límite de iteraciones ({info}) sin converger totalmente.")
    else:
        print("Error numérico durante la iteración.")

    deltaB_final = dB_1d_solucion.reshape((nx, ny))

    return campoMagneticoInicial + deltaB_final

if __name__ == "__main__":

    datos, cabecera, eje_lambda, intensidad_orig, V_orig, psf_fran = cargar_datos_y_psf()

    campoMagneticoSD, mapa_r_cuadradoSD = calcularMagnetismo(intensidad_orig, V_orig, eje_lambda)
    #campoMagneticoDec, mapa_r_cuadradoDec = calcularMagnetismoConDeconvolucion(datos, psf_fran, eje_lambda, metDecon='w_fran', workers=-1)
    campoMagneticoFM = metodoForw(intensidad_orig, V_orig, eje_lambda, psf_fran)
    
    vis.compararMagnetogramas(campoMagneticoSD, campoMagneticoFM)