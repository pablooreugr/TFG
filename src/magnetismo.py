import numpy as np
import deconvolucion as deco
import visualizacion as vis
from scipy.sparse.linalg import LinearOperator, cg
from scipy.signal import convolve2d


constanteFormula = 4.67e-13 # A^-1 G^-1

def calcularCampoMagnetico(intensidad, V, lambdas, g=3):
    #Preparamos las componentes para la regresión lineal

    lambdas3D = lambdas[:, np.newaxis, np.newaxis]

    derI = np.gradient(intensidad, lambdas, axis=0) * (lambdas3D**2)

    # Vamos a hacer a partir la relacion lineal V = K dI/dlambda * lambda^2

    numerador_pendiente = np.sum(derI * V, axis=0)
    denominador_pendiente = np.sum(derI**2, axis=0)
    pendiente = np.divide(numerador_pendiente, denominador_pendiente, out=np.zeros_like(numerador_pendiente), where=denominador_pendiente!=0)

    campoMagnetico = -pendiente/(constanteFormula*g)

    # Ahora lo que hacemos es calcular el R^2

    mediaDatosV = np.mean(V, axis=0)
    vPredicho = derI * pendiente

    numerador = np.sum((V - vPredicho)**2, axis=0)
    denominador = np.sum((V - mediaDatosV)**2, axis=0)

    rCuadrado = 1 - np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)

    return campoMagnetico, rCuadrado

def algoritmoDeNoor(intensidad, V, lambdas, psf, g=3, pasos=30, trabajadores=-1, metodo='rl', lambdaReg=1e-6, relLim=1e-30, pasosFor=30, cg_auto_close=False):
        nx = intensidad.shape[2]
        ny = intensidad.shape[1]
        n_lambda = intensidad.shape[0]

        n_total = nx * ny

        

        #Primero vamos a obtener una primera aproximacion del campo Magnetico
        campoMagneticoInicial, _ = calcularCampoMagnetico(intensidad, V, lambdas, g=g)

        #campoMagneticoInicial = np.zeros_like(campoMagneticoInicial)
        intensidad = deco.deconvolucion3D(intensidad, psf, pasos = 20)


        #intensidadDecon = deco.deconvolucion3D(intensidad, psf, metodo=metodo, pasos=pasos, trabajadores=trabajadores)
        derivadaI = np.gradient(intensidad, lambdas, axis=0)

        constanteK = - constanteFormula * g * derivadaI * (lambdas[:, np.newaxis, np.newaxis]**2)

        # Escalado del kernel K
        k_max = np.abs(constanteK).max()
        if k_max == 0:
            k_max = 1.0  # seguridad
        K_scaled = constanteK / k_max

        # A partir de aqui calcularemos los distintos operadores

        def opK(campoB):
            return K_scaled * campoB[np.newaxis, :, :]
        
        def opKt(campoV):
            return np.sum(K_scaled * campoV, axis=0)
        
        def opP(campoV, psf):
            pad_size = psf.shape[0] // 2
            res = np.pad(campoV, ((0,0), (pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
            return res, pad_size
        
        def opC(campoV, psf):
            return deco.convolucion3D(campoV, psf, usar_padding=False)
        
        def opR(campoV, pad_size):
            return campoV[:, pad_size:-pad_size, pad_size:-pad_size]
        
        def opCt(campoV, psf):
            psfGirada = deco.girarPSF(psf)
            res = deco.convolucion3D(campoV, psfGirada, usar_padding=False)
            return res
        
        def opPt(imagen_extendida, psf):
            pad_size = psf.shape[0] // 2
            """
            P^T: Transpuesta del padding por reflexión para arreglo 3D.
            """
            y = imagen_extendida.copy()
            p = pad_size

            # Eje Y (axis 1)
            y[:, p+1 : 2*p+1, :] += y[:, 0:p, :][:, ::-1, :]
            y[:, -2*p-1 : -p-1, :] += y[:, -p:, :][:, ::-1, :]

            # Eje X (axis 2)
            y[:, :, p+1 : 2*p+1] += y[:, :, 0:p][:, :, ::-1]
            y[:, :, -2*p-1 : -p-1] += y[:, :, -p:][:, :, ::-1]

            return y[:, p:-p, p:-p]
        
        def opRt(imagen, psf):
            pad_size = psf.shape[0] // 2
            return np.pad(imagen, pad_width=((0,0), (pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)

        # A partir de aqui creamos los padding
        def J(dB_2D):
            compV = opK(dB_2D)
            padding, pad_size = opP(compV, psf)
            convolucion = opC(padding, psf)
            recorte = opR(convolucion, pad_size)
            return recorte
        
        # Calculamos J^T => aclara
        
        def JT(res_3D):
            recorte = opRt(res_3D, psf)
            convolucion = opCt(recorte, psf)
            paddinTras = opPt(convolucion, psf)
            res = opKt(paddinTras)

            return res

        laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float)

        def apply_laplacian_twice(mapa):
            lap1 = convolve2d(mapa, laplacian_kernel, mode='same', boundary='symm')
            lap2 = convolve2d(lap1, laplacian_kernel, mode='same', boundary='symm')
            return lap2

        lambda_scaled = lambdaReg / (k_max**2)

        def matrizA(x1D):
            x2D = x1D.reshape((ny, nx))
            correccion = JT(J(x2D))
            reg_term = apply_laplacian_twice(x2D)
            return (correccion + lambda_scaled * reg_term).flatten()
        
        # Ahora preparamos los datos de A ΔB = J^T ΔV => Ax = β

        A_op = LinearOperator((n_total, n_total), matvec=matrizA)

        V_inicial = J(campoMagneticoInicial)
        deltaV = (V/k_max) - V_inicial

        beta_data = JT(deltaV)
        beta_reg = apply_laplacian_twice(campoMagneticoInicial)
        
        beta1D = (beta_data - lambda_scaled * beta_reg).flatten()

        # Y a partir de aquí calculamos el descenso del gradiente
        grafica = vis.GraficaConvergencia(titulo="Convergencia del Algoritmo de Noor", auto_close=cg_auto_close)

        def callback_cg(xk):
            residuo = beta1D - A_op.matvec(xk)
            norma_residuo = np.linalg.norm(residuo)
            grafica.actualizar(norma_residuo)

        dB1D, info = cg(A_op, beta1D, rtol=relLim, maxiter=pasosFor, callback=callback_cg)
        grafica.finalizar()

        if info == 0:
            print("¡Convergencia exitosa!")
        elif info > 0:
            print(f"Alcanzado límite de iteraciones ({info}) sin converger totalmente.")
        else:
            print("Error numérico durante la iteración.")

        deltaB_final = dB1D.reshape((ny, nx))

        #deltaB_actual = deltaB_final / k_max

        return campoMagneticoInicial + deltaB_final

    





def calcularCampoMagneticoRL(intensidad, V, lambdas, psf, g=3, pasos=30, trabajadores=-1):
    """
    Calcula el campo magnético deconvolucionando I y V por separado con Richardson-Lucy.

    Para la deconvolución de V se aplica un shift al mínimo:
      1. Se desplaza V hacia arriba restando su mínimo (para que sea ≥ 0, requisito de RL).
      2. Se deconvoluciona V desplazada con RL.
      3. Se deshace el shift restando el mismo valor desplazado.
    Por último se calcula el campo magnético con calcularCampoMagnetico.

    ADVERTENCIA: Este método empírico de shift rompe la antisimetría natural del
    parámetro de Stokes V y aplasta (reduce) la amplitud del campo magnético
    reconstruido, además de levantar el fondo continuo. Solo debe usarse como
    referencia o *baseline* comparativo.

    Parámetros
    ----------
    intensidad : np.ndarray  (n_lambda, ny, nx)
    V          : np.ndarray  (n_lambda, ny, nx)
    lambdas    : np.ndarray  (n_lambda,)
    psf        : np.ndarray  (ky, kx)
    g          : float, factor de Landé (por defecto 3)
    pasos      : int, número de iteraciones de RL (por defecto 30)
    trabajadores : int, workers FFT (por defecto -1 → todos)

    Retorna
    -------
    campoMagnetico : np.ndarray (ny, nx)
    rCuadrado      : np.ndarray (ny, nx)
    """
    # --- Deconvolución de I ---
    intensidad_deco = deco.deconvolucion3D(
        intensidad, psf, metodo='rl', pasos=pasos, trabajadores=trabajadores
    )

    # --- Deconvolución de V con shift al mínimo ---
    # Shift: desplazamos V hacia arriba para que sea ≥ 0 (RL requiere datos positivos)
    v_min = V.min()
    shift = -v_min if v_min < 0 else 0.0   # solo shiftamos si hay valores negativos
    V_shifted = V + shift

    V_shifted_deco = deco.deconvolucion3D(
        V_shifted, psf, metodo='rl', pasos=pasos, trabajadores=trabajadores
    )

    # Deshacemos el shift para volver al rango original
    V_deco = V_shifted_deco - shift

    # --- Campo magnético ---
    campoMagnetico, rCuadrado = calcularCampoMagnetico(
        intensidad_deco, V_deco, lambdas, g=g
    )

    return campoMagnetico, rCuadrado


if __name__ == "__main__":
    print('deco.py')