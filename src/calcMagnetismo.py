from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


ruta_archivo = 'data/prueba.fits'

g_eff = 1.5 # Linea del magnesio I
constanteFormula = 4.67e-13 #

# 3. Abrimos el archivo FITS
# Usamos 'with' para asegurarnos de que el archivo se cierra de forma segura
# en la memoria una vez que terminamos de leerlo.
with fits.open(ruta_archivo) as hdul:
    datos = hdul[0].data
    cabecera = hdul[0].header


datos_int_lambda = datos[:, 0, :, :]
datos_v_lambda = datos[:, 3, :, :]


numLongitudesOnda = datos.shape[0] #Me ayuda a saber el numero de datos que hay
eje_lambda = np.array([cabecera[f'L_{i}'] for i in range(numLongitudesOnda)]) #Extraigo los valores de la longitud de onda de la cabezera [A

lambda_cuadrado = eje_lambda**2
lambda_cuadrado_3d = lambda_cuadrado[:, np.newaxis, np.newaxis] #Esto me convierte [lambda, 1, 1] en 3D, para poder multiplicar todos los valores uno a uno

derivada = np.gradient(datos_int_lambda, eje_lambda, axis=0) 
nuevaDerivada = derivada * lambda_cuadrado_3d

m = np.sum(nuevaDerivada*datos_v_lambda, axis=0)/np.sum(nuevaDerivada**2, axis=0)

v_predicho = nuevaDerivada * m

mediaDatosV = np.mean(datos_v_lambda, axis=0)

# Numerador: Suma de los residuos al cuadrado
numerador = np.sum((datos_v_lambda - v_predicho)**2, axis=0)

# Denominador: Suma de la varianza total
denominador = np.sum((datos_v_lambda - mediaDatosV)**2, axis=0)

# Calculamos R^2 (NumPy resta el 1 a cada píxel automáticamente)
mapa_r_cuadrado = 1 - (numerador / denominador)

campoMagnetico = -m*(1/(g_eff*constanteFormula))


# Representacion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- PRIMER RECUADRO (ax1): El Magnetograma ---
im1 = ax1.imshow(campoMagnetico, cmap='RdBu_r') 
ax1.set_title('Mapa del Campo Magnético')
fig.colorbar(im1, ax=ax1, label='Valor del campo magnético paralelo G (Gauss)') 

# --- SEGUNDO RECUADRO (ax2): El mapa de R^2 ---
im2 = ax2.imshow(mapa_r_cuadrado, vmin=0, vmax=1, cmap='viridis')
ax2.set_title('Mapa de Fiabilidad (R^2)')
fig.colorbar(im2, ax=ax2, label='R^2')


plt.tight_layout()  #ajusta los márgenes automáticamente para que los títulos y las barras de color no se superpongan entre sí.

plt.show()
