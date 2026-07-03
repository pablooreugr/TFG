import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.sparse.linalg import LinearOperator, cg

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import magnetismo
import deconvolucion as deco

# Definimos la versión modificada de Noor para el experimento
def algoritmoDeNoor_custom(intensidad, V, lambdas, psf, g=3, lambdaReg=1e-6, relLim=1e-30, pasosFor=30, skip_i_deco=False):
    nx = intensidad.shape[2]
    ny = intensidad.shape[1]
    n_total = nx * ny
    
    # Aproximación inicial
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

    # Callback para guardar residual final
    norma_final = [np.linalg.norm(beta1D)] # Valor inicial por si no itera
    def callback_cg(xk):
        norma_final[0] = np.linalg.norm(beta1D - A_op.matvec(xk))

    dB1D, info = cg(A_op, beta1D, rtol=relLim, maxiter=pasosFor, callback=callback_cg)
    
    deltaB_final = dB1D.reshape((ny, nx))
    return campoMagneticoInicial + deltaB_final, norma_final[0]

# --- 1. Datos Sintéticos ---
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

# --- 2. Degradación ---
psf = deco.generar_psf_airy(15, 3.0)
print("Convolucionando datos limpios...")
I_conv = deco.convolucion3D(I_3D, psf, usar_padding=True)
V_conv = deco.convolucion3D(V_3D, psf, usar_padding=True)

np.random.seed(42)
I_obs = I_conv + np.random.normal(0, 1e-3, size=I_conv.shape)
V_obs = V_conv + np.random.normal(0, 1e-4, size=V_conv.shape)

# --- 3. Prueba A (Noor Actual) ---
print("Ejecutando Prueba A (Noor Actual)...")
B_rec_A, res_A = algoritmoDeNoor_custom(I_obs, V_obs, lambdas, psf, g=g, skip_i_deco=False)
rmse_A = np.sqrt(np.mean((B_real - B_rec_A)**2))

# --- 4. Prueba B (Noor Idealizado) ---
print("Ejecutando Prueba B (Noor Idealizado)...")
# A la prueba B le pasamos la intensidad LIMPIA (I_3D) para que calcule el Kernel perfecto
# Y le decimos que no deconvolucione la intensidad (skip_i_deco=True)
B_rec_B, res_B = algoritmoDeNoor_custom(I_3D, V_obs, lambdas, psf, g=g, skip_i_deco=True)
rmse_B = np.sqrt(np.mean((B_real - B_rec_B)**2))

# --- 5. Resultados ---
print("\n--- RESULTADOS EXPERIMENTO KERNEL ---")
print(f"RMSE Prueba A (Noor Actual):   {rmse_A:.4f} G")
print(f"RMSE Prueba B (Noor Idealizado): {rmse_B:.4f} G")
print(f"Mejora del RMSE: {(rmse_A - rmse_B):.4f} G")
print("")
print(f"Residuo Final Prueba A: {res_A:.4e}")
print(f"Residuo Final Prueba B: {res_B:.4e}")
print("-------------------------------------\n")

# --- 6. Visualización ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title(f"Prueba A (Noor Actual)\nRMSE: {rmse_A:.2f} G")
plt.imshow(B_rec_A, cmap='coolwarm', origin='lower', vmin=-1000, vmax=1000)
plt.colorbar(label="Gauss")

plt.subplot(1, 2, 2)
plt.title(f"Prueba B (Noor Idealizado)\nRMSE: {rmse_B:.2f} G")
plt.imshow(B_rec_B, cmap='coolwarm', origin='lower', vmin=-1000, vmax=1000)
plt.colorbar(label="Gauss")

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "resultado_kernel.png")
plt.savefig(output_path)
print(f"Gráfica guardada en {output_path}")
