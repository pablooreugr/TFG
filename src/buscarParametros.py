# Este script servira para poder buscar los parametros optimos para ver la deconvolución

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import magnetismo as mag
import deconvolucion as decon
import scipy.optimize as opt
from joblib import Parallel, delayed
from pathlib import Path

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

def visualizar_interactivo_intensidad(campoMagnetico, intensidad_orig, eje_lambda, parametros_ajuste):
    """
    Muestra una interfaz interactiva donde al hacer click en el campo magnético
    se muestra el perfil de Intensidad con sus valores reales y el ajuste precalculado.
    """
    # Configuración de la figura interactiva
    fig, (ax_img, ax_profile) = plt.subplots(1, 2, figsize=(14, 6))

    # Visualización del campo magnético
    # Usando el mapa de colores de visualizacion.py ('RdBu_r' de -500 a 500)
    img_plot = ax_img.imshow(campoMagnetico, cmap='RdBu_r', vmin=-500, vmax=500, origin='lower')
    ax_img.set_title("Campo Magnético Calculado\n(Haz click en un píxel)")
    fig.colorbar(img_plot, ax=ax_img, shrink=0.8, label='Campo Magnético paralelo (Gauss)')
    
    # Marcador para el pixel seleccionado
    punto_click, = ax_img.plot([], [], 'ko', markeredgecolor='white', markersize=8)

    # Gráfica del perfil de intensidad
    line_i, = ax_profile.plot([], [], label='Intensidad', color='green', linewidth=2, marker='o', markersize=4)
    line_fit, = ax_profile.plot([], [], label='Ajuste Gaussiano', color='red', linestyle='--', linewidth=2)
    ax_profile.set_title("Perfil de Intensidad en función de λ")
    ax_profile.set_xlabel("λ")
    ax_profile.set_ylabel("Intensidad (valores reales)")
    ax_profile.legend()
    ax_profile.grid(True, alpha=0.3)

    def onclick(event):
        if event.inaxes == ax_img:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            
            if 0 <= x < campoMagnetico.shape[1] and 0 <= y < campoMagnetico.shape[0]:
                # Actualizar el marcador en la imagen
                punto_click.set_data([x], [y])
                
                # Extraer perfil del píxel
                i_pix = intensidad_orig[:, y, x]
                
                # Actualizar datos en la gráfica de perfiles (valores reales)
                line_i.set_data(eje_lambda, i_pix)
                
                # Obtener el ajuste gaussiano precalculado
                popt = parametros_ajuste[:, y, x]
                if not np.isnan(popt[0]):
                    # Crear un eje X más denso para que la curva se vea suave
                    eje_lambda_denso = np.linspace(eje_lambda[0], eje_lambda[-1], 200)
                    fit_y = gaussiana(eje_lambda_denso, *popt)
                    line_fit.set_data(eje_lambda_denso, fit_y)
                else:
                    line_fit.set_data([], [])
                
                ax_profile.relim()
                ax_profile.autoscale_view()
                
                # Refrescar la figura
                fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

# A partir de aqui voy a programar la parte de buscar la gaussiana de cada punto

def gaussiana(x, C, A, lambda0, sigma):
    return C - A * np.exp(-0.5 * ((x - lambda0) / sigma)**2)

def ajustar_gaussiana(eje_lambda, intensidad_pix):
    # Estimación inicial de parámetros
    C_init = np.max(intensidad_pix)
    A_init = C_init - np.min(intensidad_pix)
    lambda0_init = eje_lambda[np.argmin(intensidad_pix)]
    sigma_init = 0.1  # Valor inicial para el ancho de la gaussiana

    p0 = [C_init, A_init, lambda0_init, sigma_init]

    try:
        popt, _ = opt.curve_fit(gaussiana, eje_lambda, intensidad_pix, p0=p0)
        return popt  # Devuelve los parámetros ajustados
    except RuntimeError:
        return np.array([np.nan, np.nan, np.nan, np.nan])

def ajustar_todos_los_pixeles_apply(eje_lambda, intensidad_orig):
    forma_orig = intensidad_orig.shape
    n_lambda = forma_orig[0]
    
    # 1. Aplanamos las dimensiones espaciales para iterar de forma lineal.
    # Si tu array es (lambda, x, y), pasará a ser (lambda, x*y).
    # Esto facilita muchísimo la distribución del trabajo entre los núcleos.
    intensidad_plana = intensidad_orig.reshape(n_lambda, -1)
    num_pixeles = intensidad_plana.shape[1]
    
    # 2. Ejecutamos en paralelo. 
    # n_jobs=-1 le dice a joblib que use TODOS los hilos disponibles.
    resultados_planos = Parallel(n_jobs=-1)(
        delayed(ajustar_gaussiana)(eje_lambda, intensidad_plana[:, i]) 
        for i in range(num_pixeles)
    )
    
    # 3. resultados_planos es una lista. La convertimos a array de NumPy.
    # Usamos .T (transpuesta) para alinear las dimensiones correctamente.
    resultados_array = np.array(resultados_planos).T 
    
    # 4. Reconstruimos la forma espacial original.
    # Si ajustar_gaussiana devuelve N parámetros, la salida final será (N, x, y)
    n_params = resultados_array.shape[0]
    resultado_final = resultados_array.reshape((n_params,) + forma_orig[1:])
    
    return resultado_final

def visualizar_mapas_parametros(parametros_ajuste, mapa_r2):
    """
    Crea una figura con 5 subgráficos mostrando los mapas 2D de los
    parámetros ajustados (C, A, lambda0, sigma) y el R^2 del campo magnético.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Ocultamos el último gráfico (el 6º) porque solo tenemos 5 mapas
    axes[-1].axis('off')
    
    titulos = ['Continuo (C)', 'Amplitud (A)', r'Centro ($\lambda_0$)', r'Ancho ($\sigma$)', r'Fiabilidad ($R^2$)']
    cmaps = ['viridis', 'plasma', 'coolwarm', 'magma', 'inferno']
    mapas = [parametros_ajuste[0], parametros_ajuste[1], parametros_ajuste[2], parametros_ajuste[3], mapa_r2]
    
    for i in range(5):
        ax = axes[i]
        mapa = mapas[i]
        
        validos = mapa[~np.isnan(mapa)]
        if validos.size > 0:
            if i == 4:
                # Para el R^2, el rango ideal es [0, 1]
                vmin, vmax = 0, 1
            else:
                # Usamos percentiles para evitar que valores anómalos estropeen la escala de color
                vmin, vmax = np.percentile(validos, 2), np.percentile(validos, 98)
        else:
            vmin, vmax = 0, 1
            
        im = ax.imshow(mapa, cmap=cmaps[i], origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(titulos[i])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    fig.suptitle("Mapas de Parámetros de Ajuste Gaussiano y Fiabilidad", fontsize=16)
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

    # Para usar la original, puedes usar esta llamada:
    # visualizar_interactivo(campoMagnetico, intensidad_orig, V_orig, gradIntensidad, eje_lambda)

    # Nos aseguramos de que la carpeta output exista y guardamos ahí
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    archivo_parametros = output_dir / "parametros_ajuste.npy"
    if archivo_parametros.exists():
        print(f"Cargando ajustes gaussianos desde {archivo_parametros}...")
        parametros_ajuste = np.load(archivo_parametros)
    else:
        print("Calculando ajustes gaussianos para todos los píxeles...")
        parametros_ajuste = ajustar_todos_los_pixeles_apply(eje_lambda, intensidad_orig)
        print(f"Guardando ajustes calculados en {archivo_parametros}...")
        np.save(archivo_parametros, parametros_ajuste)
        
    print("Ajustes listos. Abriendo visualización...")

    # Mostrar los mapas de parámetros y el mapa de R^2
    visualizar_mapas_parametros(parametros_ajuste, r2)

    visualizar_interactivo(campoMagnetico, intensidad_orig, V_orig, gradIntensidad, eje_lambda)

    # Llamamos a la visualización interactiva solo con intensidad
    visualizar_interactivo_intensidad(campoMagnetico, intensidad_orig, eje_lambda, parametros_ajuste)