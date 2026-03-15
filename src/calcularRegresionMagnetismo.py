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

datos_int_lambda = datos[:, 0, 200, 1000]

numLongitudesOnda = datos.shape[0] #Me ayuda a saber el numero de datos que hay
eje_lambda = np.array([cabecera[f'L_{i}'] for i in range(numLongitudesOnda)]) #Extraigo los valores de la longitud de onda de la cabezera [A

derivada_inten_lambda = np.gradient(datos_int_lambda, eje_lambda)






# Ahora ya puedes graficar Stokes I (datos_int_lambda) frente a eje_lambda
plt.figure(figsize=(8, 5))
plt.plot(eje_lambda, datos_int_lambda, marker='o', linestyle='-', color='black')
plt.plot(eje_lambda, derivada_inten_lambda, marker='o', linestyle='-', color='red')

plt.xlabel(r'Longitud de onda $\lambda$ ($\text{\AA}$)')
plt.ylabel('Intensidad (Stokes I)')
plt.title('Perfil de la línea de Mg I (5172 \AA)')
plt.grid(True, alpha=0.3)
plt.show()



