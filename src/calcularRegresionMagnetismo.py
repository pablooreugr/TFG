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


pixelx = 300
pixely = 600
datos_int_lambda = datos[:, 0, pixely, pixelx]
datos_V_lambda = datos[:, 3, pixely, pixelx]

numLongitudesOnda = datos.shape[0] #Me ayuda a saber el numero de datos que hay
eje_lambda = np.array([cabecera[f'L_{i}'] for i in range(numLongitudesOnda)]) #Extraigo los valores de la longitud de onda de la cabezera [A

derivada_inten_lambda = np.gradient(datos_int_lambda, eje_lambda) # El primer valor es el eje y, y el segundo el x


# A partir de aquí voy a intentar calcular el minimo cuadrado vectorizado
# Para calcular la pendiente

m = np.dot(datos_V_lambda, derivada_inten_lambda)/np.dot(derivada_inten_lambda, derivada_inten_lambda)

# Recta de mínimos cuadrados (pasa por el origen)
y_fit = m * derivada_inten_lambda

plt.figure(figsize=(8,5))

# datos reales
plt.scatter(derivada_inten_lambda, datos_V_lambda, color='black', label='Datos')

# recta ajustada
x_linea = np.linspace(derivada_inten_lambda.min(), derivada_inten_lambda.max(), 200)
plt.plot(x_linea, m*x_linea, color='red', label=f'Ajuste: y = {m:.3e} x')

plt.xlabel('Stokes I')
plt.ylabel('Stokes V')
plt.title('Ajuste lineal por mínimos cuadrados')
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()

