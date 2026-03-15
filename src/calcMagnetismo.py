from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


ruta_archivo = 'data/prueba.fits'

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

derivada = np.gradient(datos_int_lambda, eje_lambda, axis=0)

m = np.sum(derivada*datos_v_lambda, axis=0)/np.sum(derivada**2, axis=0)

# Creamos la imagen
plt.imshow(m, cmap='RdBu_r') # Usamos un mapa de color 'Red-Blue' (típico para campos magnéticos)
plt.colorbar(label='Pendiente (proporcional a B)')
plt.title('Mapa del Campo Magnético (Magnetograma)')
plt.show()

