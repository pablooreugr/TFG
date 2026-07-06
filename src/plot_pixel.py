import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from astropy.io import fits
import sys
from pathlib import Path

# Configuracion de seaborn
sns.set_theme(style="whitegrid")

def cargar_datos_y_encontrar_maximo(ruta_fits='data/prueba.fits'):
    with fits.open(ruta_fits) as hdul:
        datos = hdul[0].data
        cabecera = hdul[0].header
        
    eje_lambda = np.array([cabecera[f'L_{i}'] for i in range(datos.shape[0])])
    
    target_size = 1600
    if datos.shape[2] > target_size:
        start = (datos.shape[2] - target_size) // 2
        datos_recortados = datos[:, :, start:start+target_size, start:start+target_size]
    else:
        datos_recortados = datos

    # Calcular campo magnetico como en buscarParametros.py para encontrar el máximo
    lambdas3D = eje_lambda[:, np.newaxis, np.newaxis]
    intensidad_full = datos_recortados[:, 0, :, :]
    V_full = datos_recortados[:, 3, :, :]
    gradIntensidad_full = np.gradient(intensidad_full, eje_lambda, axis=0) * (lambdas3D**2)

    # Solo necesitamos el numerador y denominador de m para el campo magnético aprox
    m = np.sum(gradIntensidad_full*V_full, axis=0)/np.sum(gradIntensidad_full**2, axis=0)
    g_eff = 1.75
    constanteFormula = 4.67e-13 
    campoMagnetico = -m*(1/(g_eff*constanteFormula))
    
    # Buscar el máximo local en la región que parece haber clicado el usuario
    # x ~ 1250, y ~ 1180. Buscamos en x:[1200, 1300], y:[1150, 1250]
    region = campoMagnetico[1150:1250, 1200:1300]
    # En la imagen se ve que el campo magnético es fuertemente positivo (rojo oscuro)
    y_local, x_local = np.unravel_index(np.argmax(region), region.shape)
    
    x_max = 1200 + x_local
    y_max = 1150 + y_local
    
    print(f"Píxel encontrado con máximo campo magnético en la región: x={x_max}, y={y_max}")

    intensidad = datos_recortados[:, 0, y_max, x_max]
    v_pix = datos_recortados[:, 3, y_max, x_max]
    grad_pix = gradIntensidad_full[:, y_max, x_max]
    
    return eje_lambda, intensidad, v_pix, grad_pix, x_max, y_max

def generar_plot(eje_lambda, i_pix, v_pix, grad_pix, output_file="pixel_plot.png"):
    v_max = np.max(np.abs(v_pix))
    if v_max == 0: v_max = 1.0
    v_rescaled = v_pix / v_max
    
    i_min, i_max_val = np.min(i_pix), np.max(i_pix)
    i_range = i_max_val - i_min
    if i_range == 0: i_range = 1.0
    i_rescaled = ((i_pix - i_min) / i_range) * 2 - 1.0
    
    grad_max = np.max(np.abs(grad_pix))
    if grad_max == 0: grad_max = 1.0
    grad_rescaled = grad_pix / grad_max

    df = pd.DataFrame({
        r'Longitud de Onda ($\lambda$)': eje_lambda,
        'Stokes V': v_rescaled,
        'Intensidad': i_rescaled,
        r'dI/d$\lambda \cdot \lambda^2$': grad_rescaled
    })

    df_melted = df.melt(r'Longitud de Onda ($\lambda$)', var_name='Componente', value_name='Amplitud')

    plt.figure(figsize=(10, 6))
    palette = {'Stokes V': 'black', 'Intensidad': 'green', r'dI/d$\lambda \cdot \lambda^2$': 'red'}
    
    sns.lineplot(data=df_melted, x=r'Longitud de Onda ($\lambda$)', y='Amplitud', 
                 hue='Componente', style='Componente', markers=True, dashes=False,
                 palette=palette, linewidth=2, markersize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    eje_lambda, i_pix, v_pix, grad_pix, x_pix, y_pix = cargar_datos_y_encontrar_maximo()
    output_dir = Path("docs/img")
    output_dir.mkdir(parents=True, exist_ok=True)
    generar_plot(eje_lambda, i_pix, v_pix, grad_pix, output_file=f"docs/img/pixel_sintetico_{x_pix}_{y_pix}.png")
