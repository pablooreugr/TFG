import numpy as np
import deconvolucion as deco
import visualizacion as vis
from scipy.sparse.linalg import LinearOperator, cg
from scipy.ndimage import laplace


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

        intensidad = deco.deconvolucion3D(intensidad, psf, pasos = 20)

        #Primero vamos a obtener una primera aproximacion del campo Magnetico
        campoMagneticoInicial, _ = calcularCampoMagnetico(intensidad, V, lambdas, g=g)

        #campoMagneticoInicial = np.zeros_like(campoMagneticoInicial)


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

        def matrizA(x1D, lambda_reg = lambdaReg):
            lambda_scaled = lambdaReg / (k_max**2)   # porque ΔB = ΔB' / k_max
            x2D = x1D.reshape((ny, nx))

            correccion = JT(J(x2D))

            suavidad = - lambda_scaled * laplace(x2D)

            resultado = correccion + suavidad

            return resultado.flatten()
        
        # Ahora preparamos los datos de A ΔB = J^T ΔV => Ax = β

        A_op = LinearOperator((n_total, n_total), matvec=matrizA)

        V_inicial = J(campoMagneticoInicial)
        deltaV = (V/k_max) - V_inicial

        beta = JT(deltaV)
        beta1D = beta.flatten()

        # Y a partir de aquí calculamos el descenso del gradiente
        grafica = vis.GraficaConvergencia(titulo="Convergencia del Algoritmo de Noor", auto_close=cg_auto_close)
        
        historial_convergencia = []
        estado_previo = [np.zeros_like(beta1D)]

        def callback_cg(xk):
            # Diferencia iterativa entre iteraciones
            diff = np.linalg.norm(xk - estado_previo[0])
            historial_convergencia.append(diff)
            estado_previo[0] = xk.copy()
            
            # Para la gráfica interactiva podemos seguir usando la norma del residuo
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

        return campoMagneticoInicial + deltaB_final, historial_convergencia

    





if __name__ == "__main__":
    print('deco.py')