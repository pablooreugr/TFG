# 1. Importamos la librería necesaria para manejar archivos FITS
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# 2. Definimos la ruta de tu archivo FITS
ruta_archivo = 'data/prueba.fits'

# 3. Abrimos el archivo FITS
# Usamos 'with' para asegurarnos de que el archivo se cierra de forma segura
# en la memoria una vez que terminamos de leerlo.
with fits.open(ruta_archivo) as hdul:
    
    # hdul significa "Header Data Unit List" (Lista de bloques del FITS)
    # Vamos a imprimir qué contiene para entender su estructura
    hdul.info()
    
    # Normalmente, los datos científicos están en el bloque 0 (PrimaryHDU) o en el 1.
    # Supongamos que están en el 0. Vamos a extraer los datos y la cabecera.
    datos = hdul[0].data
    cabecera = hdul[0].header

# 4. Comprobamos la forma (shape) de nuestra matriz de datos
print("\nLa forma de los datos es:", datos.shape)
print(repr(cabecera))

imagen_intensidad = datos[0, 0, :, :] # El primer datos es la longitud de onda, la segunda el parametro de stokes, la tercera los pixeles del eje y, y la ultima los pixeles del eje x

plt.figure(figsize=(8, 8))
plt.imshow(imagen_intensidad, cmap='hot', origin='lower')
plt.show()

