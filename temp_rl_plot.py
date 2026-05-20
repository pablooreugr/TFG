import sys
import os

# Añadimos la carpeta src al path para poder importar deconvolucion
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from deconvolucion import deconvolucionRLMulti, psfAiry

def main():
    ruta_archivo = 'data/prueba.fits'
    
    # 1. Cargar datos
    print(f"Cargando datos de {ruta_archivo}...")
    with fits.open(ruta_archivo) as hdul:
        datos = hdul[0].data
    
    # Tomamos la imagen de intensidad (usamos [0, 0, :, :] como en probar_deconvolucion)
    # Si tu intensidad está en el índice 3, cámbialo a [0, 3, :, :]
    imagenInt = datos[0, 0, :, :] 
    
    # 2. Generar PSF
    print("Generando PSF de Airy...")
    miPsf = psfAiry(imagenInt)
    
    # 3. Definir los pasos a evaluar
    pasos_lista = [5, 10, 20, 30, 50]
    
    # 4. Preparar la figura
    # Tendremos un plot para la imagen original + uno para cada paso
    num_plots = len(pasos_lista) + 1
    fig, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(4 * num_plots, 5))
    
    # Rango de visualización deseado
    x_min, x_max = 600, 800
    y_min, y_max = 500, 700
    
    # Mostrar imagen original
    axs[0].imshow(imagenInt, cmap='hot', origin='lower')
    axs[0].set_title('Imagen Original')
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    
    # 5. Iterar sobre cada cantidad de pasos
    for i, pasos in enumerate(pasos_lista):
        print(f"Calculando Deconvolución Richardson-Lucy ({pasos} pasos)...")
        # Usamos epsilon=0 para asegurar que corra exactamente la cantidad de pasos indicada
        img_rl = deconvolucionRLMulti(imagenInt, miPsf, epsilon=-1, pasos=pasos)
        
        axs[i+1].imshow(img_rl, cmap='hot', origin='lower')
        axs[i+1].set_title(f'RL ({pasos} pasos)')
        axs[i+1].set_xlim(x_min, x_max)
        axs[i+1].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Guardar en output por si acaso
    if not os.path.exists('output'):
        os.makedirs('output')
    ruta_salida = 'output/comparativa_rl_pasos.png'
    plt.savefig(ruta_salida)
    print(f"✅ Imagen guardada en: {ruta_salida}")
    
    # Mostrar en pantalla
    plt.show()

if __name__ == "__main__":
    main()
