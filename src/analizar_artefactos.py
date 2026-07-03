import numpy as np
import sys
import os
from scipy.ndimage import laplace

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import magnetismo
import deconvolucion as deco
from scipy.sparse.linalg import LinearOperator, cg

def algoritmoDeNoor_custom(intensidad, V, lambdas, psf, g=3, lambdaReg=1e-6, relLim=1e-30, pasosFor=30, skip_i_deco=False):
    nx = intensidad.shape[2]
    ny = intensidad.shape[1]
    n_total = nx * ny
    campoMagneticoInicial, _ = magnetismo.calcularCampoMagnetico(intensidad, V, lambdas, g=g)
    if not skip_i_deco:
        intensidad_uso = deco.deconvolucion3D(intensidad, psf, pasos=20)
    else:
        intensidad_uso = intensidad.copy()
    derivadaI = np.gradient(intensidad_uso, lambdas, axis=0)
    constanteFormula = 4.67e-13
    constanteK = - constanteFormula * g * derivadaI * (lambdas[:, np.newaxis, np.newaxis]**2)
    k_max = np.abs(constanteK).max()
    if k_max == 0: k_max = 1.0
    K_scaled = constanteK / k_max

    def opK(campoB): return K_scaled * campoB[np.newaxis, :, :]
    def opKt(campoV): return np.sum(K_scaled * campoV, axis=0)
    def opP(campoV, psf):
        pad_size = psf.shape[0] // 2
        return np.pad(campoV, ((0,0), (pad_size, pad_size), (pad_size, pad_size)), mode='reflect'), pad_size
    def opC(campoV, psf): return deco.convolucion3D(campoV, psf, usar_padding=False)
    def opR(campoV, pad_size): return campoV[:, pad_size:-pad_size, pad_size:-pad_size]
    def opCt(campoV, psf):
        return deco.convolucion3D(campoV, deco.girarPSF(psf), usar_padding=False)
    def opPt(imagen_extendida, psf):
        pad_size = psf.shape[0] // 2
        y = imagen_extendida.copy()
        p = pad_size
        y[:, p+1 : 2*p+1, :] += y[:, 0:p, :][:, ::-1, :]
        y[:, -2*p-1 : -p-1, :] += y[:, -p:, :][:, ::-1, :]
        y[:, :, p+1 : 2*p+1] += y[:, :, 0:p][:, :, ::-1]
        y[:, :, -2*p-1 : -p-1] += y[:, :, -p:][:, :, ::-1]
        return y[:, p:-p, p:-p]
    def opRt(imagen, psf):
        pad_size = psf.shape[0] // 2
        return np.pad(imagen, pad_width=((0,0), (pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)

    def J(dB_2D):
        return opR(opC(opP(opK(dB_2D), psf)[0], psf), psf.shape[0] // 2)
    def JT(res_3D):
        return opKt(opPt(opCt(opRt(res_3D, psf), psf), psf))
    def matrizA(x1D):
        lambda_scaled = lambdaReg / (k_max**2)
        x2D = x1D.reshape((ny, nx))
        correccion = JT(J(x2D))
        return (correccion + lambda_scaled * x2D).flatten()
        
    A_op = LinearOperator((n_total, n_total), matvec=matrizA)
    V_inicial = J(campoMagneticoInicial)
    deltaV = (V/k_max) - V_inicial
    beta1D = JT(deltaV).flatten()

    dB1D, info = cg(A_op, beta1D, rtol=relLim, maxiter=pasosFor)
    return campoMagneticoInicial + dB1D.reshape((ny, nx))

nx, ny = 64, 64
B_real = np.zeros((ny, nx))
y, x = np.ogrid[-ny//2:ny//2, -nx//2:nx//2]
B_real += 1000 * np.exp(-((x-10)**2 + (y-10)**2) / (8**2))
B_real -= 800 * np.exp(-((x+15)**2 + (y+15)**2) / (6**2))

lambdas = np.linspace(6300, 6303, 60)
lambda0 = 6301.5
I0 = 1.0 - 0.5 * np.exp(-((lambdas - lambda0)/0.2)**2)
I_3D = np.tile(I0[:, np.newaxis, np.newaxis], (1, ny, nx))

dI0_dl = np.gradient(I0, lambdas)
dI_3D = np.tile(dI0_dl[:, np.newaxis, np.newaxis], (1, ny, nx))
lambdas3D = lambdas[:, np.newaxis, np.newaxis]
constanteFormula = 4.67e-13
g = 3
V_3D = -constanteFormula * g * (lambdas3D**2) * B_real[np.newaxis, :, :] * dI_3D

psf = deco.generar_psf_airy(15, 3.0)
I_conv = deco.convolucion3D(I_3D, psf, usar_padding=True)
V_conv = deco.convolucion3D(V_3D, psf, usar_padding=True)

np.random.seed(42)
# Usemos un ruido mayor para amplificar los artefactos del kernel, como 5%
I_obs = I_conv + np.random.normal(0, 5e-3, size=I_conv.shape)
V_obs = V_conv + np.random.normal(0, 5e-4, size=V_conv.shape)

B_rec_A = algoritmoDeNoor_custom(I_obs, V_obs, lambdas, psf, g=g, skip_i_deco=False)
B_rec_B = algoritmoDeNoor_custom(I_3D, V_obs, lambdas, psf, g=g, skip_i_deco=True)

lap_A = np.abs(laplace(B_rec_A))
lap_B = np.abs(laplace(B_rec_B))

print("Energia de alta frecuencia (Ruido/Pixeles sueltos) Prueba A:", lap_A.sum())
print("Energia de alta frecuencia (Ruido/Pixeles sueltos) Prueba B:", lap_B.sum())
print("Maximo pico de ruido local Prueba A:", lap_A.max())
print("Maximo pico de ruido local Prueba B:", lap_B.max())
