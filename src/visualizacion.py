import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import matplotlib.image as mpimg

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