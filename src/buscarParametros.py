# Este script servira para poder buscar los parametros optimos para ver la deconvolución

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import magnetismo as mag
import deconvolucion as decon

def visualizar_interactivo(campoMagnetico, intensidad_orig, V_orig, gradIntensidad, eje_lambda):
    """
    Muestra una interfaz interactiva donde al hacer click en el campo magnético
    se muestran los perfiles de Stokes V, Intensidad y dI/dλ reescalados, además
    de la relación de linealidad para calcular el campo en ese píxel.
    """
    # Configuración de la figura interactiva
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, width_ratios=[1.2, 1])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_profiles = fig.add_subplot(gs[0, 1])
    ax_linearity = fig.add_subplot(gs[1, 1])

    # Visualización del campo magnético
    # Usando el mapa de colores de visualizacion.py ('RdBu_r' de -500 a 500)
    img_plot = ax_img.imshow(campoMagnetico, cmap='RdBu_r', vmin=-500, vmax=500, origin='lower')
    ax_img.set_title("Campo Magnético Calculado\n(Haz click en un píxel)")
    fig.colorbar(img_plot, ax=ax_img, shrink=0.8, label='Campo Magnético paralelo (Gauss)')
    
    # Marcador para el pixel seleccionado
    punto_click, = ax_img.plot([], [], 'ko', markeredgecolor='white', markersize=8)

    # Gráfica de perfiles con puntos ('marker')
    line_v, = ax_profiles.plot([], [], label='V', color='black', linewidth=2, marker='o', markersize=4)
    line_i, = ax_profiles.plot([], [], label='I (reescalado)', color='green', linestyle='--', marker='o', markersize=4)
    line_di, = ax_profiles.plot([], [], label='dI/dλ (reescalado)', color='red', linestyle='-.', marker='o', markersize=4)
    ax_profiles.set_title("Perfiles en función de λ")
    ax_profiles.set_xlabel("λ")
    ax_profiles.set_ylabel("Amplitud")
    ax_profiles.legend()
    ax_profiles.grid(True, alpha=0.3)

    # Gráfica de linealidad con puntos
    scatter_lin = ax_linearity.scatter([], [], s=25, color='black', zorder=5)
    line_fit, = ax_linearity.plot([], [], color='red', label='Ajuste lineal', linestyle='--', zorder=4)
    ax_linearity.set_title("Linealidad: V vs dI/dλ * λ²")
    ax_linearity.set_xlabel("dI/dλ * λ²")
    ax_linearity.set_ylabel("V")
    ax_linearity.legend()
    ax_linearity.grid(True, alpha=0.3)

    def onclick(event):
        if event.inaxes == ax_img:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            
            if 0 <= x < campoMagnetico.shape[1] and 0 <= y < campoMagnetico.shape[0]:
                # Actualizar el marcador en la imagen
                punto_click.set_data([x], [y])
                
                # Extraer perfiles del píxel
                v_pix = V_orig[:, y, x]
                i_pix = intensidad_orig[:, y, x]
                grad_pix = gradIntensidad[:, y, x]
                
                # Reescalar I y dI/dλ para que la forma sea comparable a V
                v_max = np.max(np.abs(v_pix))
                if v_max == 0: v_max = 1.0
                
                # I: escalamos para que ocupe el rango [-v_max, v_max]
                i_min, i_max_val = np.min(i_pix), np.max(i_pix)
                i_range = i_max_val - i_min
                if i_range == 0: i_range = 1.0
                i_rescaled = ((i_pix - i_min) / i_range) * 2 * v_max - v_max
                
                # dI/dλ: escalamos para que el máximo absoluto coincida con v_max
                grad_max = np.max(np.abs(grad_pix))
                if grad_max == 0: grad_max = 1.0
                grad_rescaled = grad_pix * (v_max / grad_max)
                
                # Actualizar datos en la gráfica de perfiles
                line_v.set_data(eje_lambda, v_pix)
                line_i.set_data(eje_lambda, i_rescaled)
                line_di.set_data(eje_lambda, grad_rescaled)
                
                ax_profiles.relim()
                ax_profiles.autoscale_view()
                
                # Actualizar gráfica de linealidad
                scatter_lin.set_offsets(np.c_[grad_pix, v_pix])
                
                # Calcular y dibujar la recta de ajuste
                sum_grad2 = np.sum(grad_pix**2)
                if sum_grad2 != 0:
                    m = np.sum(grad_pix * v_pix) / sum_grad2
                else:
                    m = 0
                
                g_min, g_max_plot = np.min(grad_pix), np.max(grad_pix)
                line_fit.set_data([g_min, g_max_plot], [m * g_min, m * g_max_plot])
                
                ax_linearity.relim()
                ax_linearity.autoscale_view()
                
                # Refrescar la figura
                fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    datos, cabecera, eje_lambda, intensidad_orig, V_orig, psf_fran = mag.cargar_datos_y_psf()

    # Deconvolución
    V_orig = decon.aplicar_deconvolucion_3d(V_orig, psf=psf_fran, metodo='rl', iteraciones=50, workers=-1)
    intensidad_orig = decon.aplicar_deconvolucion_3d(intensidad_orig, psf=psf_fran, metodo='rl', iteraciones=50, workers=-1)

    # calcularMagnetismo devuelve campoMagnetico y mapa_r_cuadrado
    campoMagnetico, r2 = mag.calcularMagnetismo(intensidad_orig, V_orig, eje_lambda)

    # Calculamos el gradiente que se usa para la linealidad (dI/dλ * λ²)
    lambdas3D = eje_lambda[:, np.newaxis, np.newaxis]
    gradIntensidad = np.gradient(intensidad_orig, eje_lambda, axis=0) * (lambdas3D**2)

    # Llamamos a la visualización interactiva
    visualizar_interactivo(campoMagnetico, intensidad_orig, V_orig, gradIntensidad, eje_lambda)