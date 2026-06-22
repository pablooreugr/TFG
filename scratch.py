import numpy as np
import deconvolucion as deco
import magnetismo as mag

datos_cargados = np.load('data/datos_sunspot.npz')
data = datos_cargados['stokes']
lambdas = datos_cargados['lam']
intensidad = data[:, 0, :, :]
compV = data[:, 3, :, :]
psf = deco.generar_psf_airy(tamano_matriz=31, radio_piz=3)
lambdas_absolutas = 6173.0 + (lambdas / 1000.0)

# Simulate blur
int_b = deco.convolucion3D(intensidad, psf)
v_b = deco.convolucion3D(compV, psf)

# Compute K and check norms
g=3
constanteFormula = 4.67e-13
derivadaI = np.gradient(int_b, lambdas_absolutas, axis=0)
K = - constanteFormula * g * derivadaI * (lambdas_absolutas[:, np.newaxis, np.newaxis]**2)

print(f"Max K: {np.max(np.abs(K))}")
print(f"Mean K: {np.mean(np.abs(K))}")
K_squared_sum = np.sum(K**2, axis=0)
print(f"Max K^2 sum: {np.max(K_squared_sum)}")
print(f"Mean K^2 sum: {np.mean(K_squared_sum)}")

