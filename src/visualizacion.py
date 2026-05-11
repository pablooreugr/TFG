import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import matplotlib.image as mpimg
import matplotlib.colors as colors

def explorar_resultados():
    # ==========================================
    # 1. CONFIGURACIÓN DE PARÁMETROS
    # ==========================================
    # Cambia esto si tus imágenes están en otra carpeta (ej. 'output/pruebaV2')
    DIRECTORIO_IMAGENES = 'output/pruebaV2' 

    lambdas = [5172.429, 5172.532, 5172.634, 5172.687, 5172.738, 
               5172.79,  5172.841, 5172.942, 5173.044, 5173.4]

    # Opciones disponibles según tu barrido
    metodos = ('fourier', 'wiener', 'rl')
    psfs = ('airy', 'gaussiana')
    params_psf = ('1.0', '3.0', '5.0')
    valores_k = ('1e-04', '1e-03', '1e-02') # Representación en string para los botones
    iteraciones_rl = ('15', '30', '50')
    epsilon_str = "1e-12" # Fijo según tu código

    # ==========================================
    # 2. CONFIGURACIÓN DE LA INTERFAZ
    # ==========================================
    fig, ax_img = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(left=0.35, bottom=0.15) # Dejamos espacio a la izquierda y abajo para los controles
    ax_img.axis('off')

    # --- Creación de Ejes para los Widgets ---
    # Coordenadas: [izquierda, abajo, ancho, alto]
    ax_metodo = plt.axes([0.02, 0.70, 0.12, 0.15], facecolor='lightgoldenrodyellow')
    ax_psf    = plt.axes([0.02, 0.50, 0.12, 0.15], facecolor='lightgoldenrodyellow')
    ax_param  = plt.axes([0.16, 0.50, 0.12, 0.15], facecolor='lightgoldenrodyellow')
    ax_k      = plt.axes([0.02, 0.30, 0.12, 0.15], facecolor='lightgoldenrodyellow')
    ax_iter   = plt.axes([0.16, 0.30, 0.12, 0.15], facecolor='lightgoldenrodyellow')
    ax_lambda = plt.axes([0.35, 0.05, 0.55, 0.03], facecolor='lightgoldenrodyellow')

    # --- Títulos de los selectores ---
    ax_metodo.set_title('Método Decon.')
    ax_psf.set_title('Tipo PSF')
    ax_param.set_title('Parámetro PSF')
    ax_k.set_title('Valor k')
    ax_iter.set_title('Iteraciones (Solo RL)')

    # --- Widgets ---
    radio_metodo = RadioButtons(ax_metodo, metodos)
    radio_psf    = RadioButtons(ax_psf, psfs)
    radio_param  = RadioButtons(ax_param, params_psf)
    radio_k      = RadioButtons(ax_k, valores_k)
    radio_iter   = RadioButtons(ax_iter, iteraciones_rl)
    slider_lam   = Slider(ax_lambda, 'Índice Lambda', 0, len(lambdas)-1, valinit=0, valstep=1, valfmt='%0.0f')

    # ==========================================
    # 3. FUNCIÓN DE ACTUALIZACIÓN
    # ==========================================
    def actualizar(val):
        # Recoger valores de los selectores
        met = radio_metodo.value_selected
        tipo_psf = radio_psf.value_selected
        p_psf = float(radio_param.value_selected)
        k_val = float(radio_k.value_selected)
        it_val = int(radio_iter.value_selected)
        lam_idx = int(slider_lam.val)
        
        # Lógica importante: Si no es RL, las iteraciones se guardaron como 0
        if met in ['fourier', 'wiener']:
            it_val = 0
            
        # Reconstruir el nombre del archivo exactamente como lo guarda tu script
        nombre_archivo = f"mag_{met}_{tipo_psf}_p{p_psf}_k{k_val:.1e}_eps{epsilon_str}_it{it_val}_lam{lam_idx:03d}.png"
        ruta_completa = os.path.join(DIRECTORIO_IMAGENES, nombre_archivo)
        
        # Limpiar el eje de la imagen
        ax_img.clear()
        ax_img.axis('off')
        
        # Intentar cargar la imagen
        if os.path.exists(ruta_completa):
            img = mpimg.imread(ruta_completa)
            ax_img.imshow(img)
            titulo = f"Longitud de onda: {lambdas[lam_idx]:.3f} Å\nArchivo: {nombre_archivo}"
            ax_img.set_title(titulo, fontsize=11, color='black')
        else:
            # Si la combinación no existe, mostrar un mensaje de error
            ax_img.text(0.5, 0.5, f"❌ IMAGEN NO ENCONTRADA\n\n{nombre_archivo}", 
                        horizontalalignment='center', verticalalignment='center', 
                        color='red', fontsize=14, transform=ax_img.transAxes)
            ax_img.set_title("Combinación sin datos", color='red')
            
        fig.canvas.draw_idle()

    # ==========================================
    # 4. CONECTAR EVENTOS E INICIAR
    # ==========================================
    radio_metodo.on_clicked(actualizar)
    radio_psf.on_clicked(actualizar)
    radio_param.on_clicked(actualizar)
    radio_k.on_clicked(actualizar)
    radio_iter.on_clicked(actualizar)
    slider_lam.on_changed(actualizar)

    # Llamada inicial para cargar la primera imagen al abrir
    actualizar(None)

    plt.show()


# ==========================================
# 5. OTRAS VISUALIZACIONES (Desde Magnetismo)
# ==========================================

def dibujarMagYR(campoMagnetico, mapa_r_cuadrado):
    # Representacion
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- PRIMER RECUADRO (ax1): El Magnetograma ---
    #im1 = ax1.imshow(campoMagnetico, cmap='RdBu_r') 
    im1 = ax1.imshow(campoMagnetico, cmap='RdBu_r', vmin=-500, vmax=500) 
    fig.colorbar(im1, ax=ax1, label='Valor del campo magnético paralelo G (Gauss)') 

    # --- SEGUNDO RECUADRO (ax2): El mapa de R^2 ---
    im2 = ax2.imshow(mapa_r_cuadrado, vmin=0, vmax=1, cmap='viridis')
    ax2.set_title('Mapa de Fiabilidad (R^2)')
    fig.colorbar(im2, ax=ax2, label='R^2')


    plt.tight_layout()  #ajusta los márgenes automáticamente para que los títulos y las barras de color no se superpongan entre sí.

    plt.show()


def dibujarComparacionPSF(psf_cargada, psf_airy):
    """
    Compara la PSF cargada con la PSF de Airy en un mismo plot.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # --- PSF cargada ---
    im1 = ax1.imshow(psf_cargada, cmap='viridis', norm=colors.LogNorm(vmin=1e-10))
    ax1.set_title('PSF Cargada')
    fig.colorbar(im1, ax=ax1, label='Intensidad')

    # --- PSF de Airy ---
    im2 = ax2.imshow(psf_airy, cmap='viridis', norm=colors.LogNorm(vmin=1e-10))
    ax2.set_title('PSF de Airy')
    fig.colorbar(im2, ax=ax2, label='Intensidad')

    # --- Diferencia ---
    im3 = ax3.imshow(psf_cargada - psf_airy, cmap='RdBu_r', norm=colors.LogNorm(vmin=1e-10))
    ax3.set_title('Diferencia (Cargada - Airy)')
    fig.colorbar(im3, ax=ax3, label='Diferencia')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    explorar_resultados()