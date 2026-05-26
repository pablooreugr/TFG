import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import deconvolucion as decon
from mod.pd_functions_v22 import PSF
import visualizacion as vis
from scipy.sparse.linalg import LinearOperator, cg
import time
import scipy.fft as sp_fft
import scipy.signal as sp_signal
from matplotlib.colors import LogNorm

g_eff = 3 # Linea del magnesio I
constanteFormula = 4.67e-13 

class MonitorKrylov:
    def __init__(self, plot_live=True):
        self.iteracion = 0
        self.t_inicio = time.time()
        self.x_anterior = None
        self.historial_pasos = [] # Por si luego quieres graficarlo
        self.plot_live = plot_live
        
        if self.plot_live:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 5))
            self.line, = self.ax.plot([], [], 'b.-')
            self.ax.set_title("Convergencia de Gradiente Conjugado")
            self.ax.set_xlabel("Iteración")
            self.ax.set_ylabel("Magnitud del paso ($||x_k - x_{k-1}||$)")
            self.ax.set_yscale('log')
            self.ax.grid(True)
            self.fig.show()

    def __call__(self, xk):
        self.iteracion += 1
        t_actual = time.time() - self.t_inicio
        
        # Calculamos cuánto ha cambiado el mapa respecto a la iteración anterior
        if self.x_anterior is not None:
            cambio = np.linalg.norm(xk - self.x_anterior)
            self.historial_pasos.append(cambio)
            
            # Imprimimos en la terminal sobrescribiendo la misma línea
            print(f"Iteración {self.iteracion:03d} | Tiempo: {t_actual:.1f} s | Magnitud del paso: {cambio:.2e}      ", end='\r')
            
            # Actualizamos la gráfica
            if self.plot_live and len(self.historial_pasos) > 0:
                self.line.set_data(range(1, len(self.historial_pasos) + 1), self.historial_pasos)
                self.ax.relim()
                self.ax.autoscale_view()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
        else:
            print(f"Iteración {self.iteracion:03d} | Tiempo: {t_actual:.1f} s | Magnitud del paso: Calculando...      ", end='\r')
            
        # Guardamos el estado actual para la siguiente iteración
        self.x_anterior = xk.copy()
        
    def finalizar_grafica(self):
        if self.plot_live:
            plt.ioff()
            # Mostramos la gráfica final y bloqueamos hasta que se cierre (opcional)
            # plt.show()

def visualizar_psf_log(psf, titulo="PSF"):
    """
    Visualiza una PSF en 2D y su corte transversal en 1D usando escala logarítmica.
    """
    # Recortamos los valores minúsculos o ceros absolutos para que LogNorm no lance errores
    psf_segura = np.clip(psf, a_min=1e-12, a_max=None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 1. Imagen 2D Logarítmica ---
    # Usamos cmap='magma' o 'inferno' porque resaltan de maravilla las variaciones tenues en el fondo
    im = ax1.imshow(psf_segura, norm=LogNorm(vmin=1e-6, vmax=psf_segura.max()), cmap='magma')
    ax1.set_title(f"Vista 2D Log - {titulo}")
    fig.colorbar(im, ax=ax1, label='Intensidad (Log)')

    # --- 2. Perfil 1D (Corte transversal por el centro) ---
    centro_y = psf.shape[0] // 2
    ax2.plot(psf_segura[centro_y, :], color='red', lw=2)
    ax2.set_yscale('log')
    ax2.set_title("Corte Transversal 1D (Centro)")
    ax2.set_xlabel("Píxel X")
    ax2.set_ylabel("Intensidad (Log)")
    
    # Añadimos una cuadrícula densa para leer bien las caídas de órdenes de magnitud
    ax2.grid(True, which="major", color='black', alpha=0.3)
    ax2.grid(True, which="minor", linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.show()


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

def metodoForw(intensidad, V, lambdas, psf, recorte=100, trabajadores=-1, relLim=1e-3, pasos=1000):
    nx = intensidad.shape[2]
    ny = intensidad.shape[1]
    n_lambda = intensidad.shape[0]

    n_total = nx * ny

    # Primero vamos a intentar preparar los datos antes del método
    campoMagneticoInicial = calcularMagnetismo(intensidad, V, lambdas)[0]

    # 1. Deconvolución de I usando la PSF GRANDE completa
    intensidad_decon = decon.aplicar_deconvolucion_3d(intensidad, psf=psf, metodo='rl', workers=-1)

    # --- 2. RECORTAR, APODIZAR Y NORMALIZAR LA PSF ---
    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
    mitad = recorte // 2
    
    # 2.1 Extraemos el parche central bruto
    psf_recortada = psf[cy - mitad : cy + mitad + 1, cx - mitad : cx + mitad + 1]
    
    visualizar_psf_log(psf_recortada, titulo="PSF Apodizada (Filtro Gaussiano)")
    # 2.2 Creamos el filtro Gaussiano 2D (Apodización)
    # Generamos los ejes de coordenadas centrados en cero
    y, x = np.ogrid[-mitad:mitad+1, -mitad:mitad+1]
    
    # Definimos la anchura de la campana (sigma). 
    # Un divisor de 2.5 o 3 asegura que la función caiga a casi cero en los bordes del recorte.
    sigma = mitad / 2.5 
    ventana_gaussiana = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Aplicamos el filtro al parche original
    psf_suavizada = psf_recortada * ventana_gaussiana
    
    # 2.3 OBLIGATORIO: Normalizamos el resultado final (suma = 1.0) 
    psf_pequena = psf_suavizada / np.sum(psf_suavizada)
    
    # Simetría del Adjunto: Usamos la PSF pequeña invertida espacialmente 
    psf_espejo = psf_pequena[::-1, ::-1]
    # --------------------------------------------------------------------
    visualizar_psf_log(psf_pequena, titulo="PSF Apodizada (Filtro Gaussiano)")
    derivadaI = np.gradient(intensidad_decon, lambdas, axis=0)
    K_cubo = -constanteFormula * g_eff * derivadaI * (lambdas[:, np.newaxis, np.newaxis]**2)

    # Definimos los operadores lineales que usaremos.
    def J(dB_2D):
        v_ideal = K_cubo * dB_2D[np.newaxis, :, :]
        # Creamos una matriz vacía nueva para no pisar la anterior
        v_degradado = np.zeros_like(v_ideal)

        with sp_fft.set_workers(trabajadores):
            for i in range(n_lambda):
                # APLICAMOS LA PSF PEQUEÑA
                v_degradado[i, :, :] = sp_signal.fftconvolve(v_ideal[i, :, :], psf_pequena, mode='same')
        return v_degradado

    # Calculo la V inicial (usando el operador forward correctamente)
    V_inicial = J(campoMagneticoInicial)
    deltaV = V - V_inicial

    # Operador adjunto
    def JT(residuos3d):
        # Evitamos modificar 'residuos3d' in-place creando una matriz nueva
        residuos_conv = np.zeros_like(residuos3d)
        with sp_fft.set_workers(trabajadores):
            for i in range(n_lambda):
                # APLICAMOS LA PSF PEQUEÑA INVERTIDA (ESPEJO)
                residuos_conv[i, :, :] = sp_signal.fftconvolve(residuos3d[i, :, :], psf_espejo, mode='same')

        residuos_multi = residuos_conv * K_cubo

        dB_2D = np.sum(residuos_multi, axis=0)

        return dB_2D
    
    # Parámetro de regularización Tikhonov
    lambda_reg = 5e-2

    def funcionA(x_1D):
        x_2D = x_1D.reshape((nx, ny))

        # Aplicamos la cadena de operadores J^T(J(X))
        efecto_telescopio = J(x_2D)
        correccion_estimada = JT(efecto_telescopio)

        # Añadimos regularización Tikhonov
        resultado = correccion_estimada + lambda_reg * x_2D

        return resultado.flatten()
    
    # Ahora preparamos el sistema para que aparezca el A*x = b
    b_2D = JT(deltaV)

    b_1D = b_2D.flatten()

    #Y creamos la matriz con el operador
    matrizA = LinearOperator((n_total, n_total), matvec=funcionA)

    # A partir de aqui es donde ocurre la solucion del sistema

    print('Iniciando sistema de inversion')

    monitor = MonitorKrylov(plot_live=True)

    dB_1d_solucion, info = cg(matrizA, b_1D, rtol=relLim, maxiter=pasos, callback=monitor)
    print() # Para no sobreescribir la última línea en la terminal

    if info == 0:
        print("¡Convergencia exitosa!")
    elif info > 0:
        print(f"Alcanzado límite de iteraciones ({info}) sin converger totalmente.")
    else:
        print("Error numérico durante la iteración.")
        
    # Dejamos de actualizar en tiempo real
    monitor.finalizar_grafica()

    deltaB_final = dB_1d_solucion.reshape((nx, ny))

    return campoMagneticoInicial + deltaB_final

if __name__ == "__main__":

    datos, cabecera, eje_lambda, intensidad_orig, V_orig, psf_fran = cargar_datos_y_psf(ruta_fits='data/prueba.fits')

    campoMagneticoSD, mapa_r_cuadradoSD = calcularMagnetismo(intensidad_orig, V_orig, eje_lambda)
    #campoMagneticoDec, mapa_r_cuadradoDec = calcularMagnetismoConDeconvolucion(datos, psf_fran, eje_lambda, metDecon='w_fran', workers=-1)
    campoMagneticoFM = metodoForw(intensidad_orig, V_orig, eje_lambda, psf_fran)
    
    vis.compararMagnetogramas(campoMagneticoSD, campoMagneticoFM)