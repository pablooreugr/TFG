import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import matplotlib.image as mpimg
import os

# --- 1. DEFINICIÓN DE LOS PARÁMETROS EXPLORADOS ---
metodos = ['wiener', 'fourier', 'rl']
tipos_psf = ['airy', 'gaussiana']
escalas = [1.0, 3.0, 5.0]
valores_k = [1e-4, 1e-3, 0.1]
valores_epsilon = [1.0, 10.0, 50.0, 100.0]
lambdas = [5172.429, 5172.532, 5172.634, 5172.687, 5172.738, 5172.79, 5172.841, 5172.942, 5173.044, 5173.4]
iteraciones = 1000

# --- 2. CONFIGURACIÓN DE LA FIGURA ---
fig, ax_img = plt.subplots(figsize=(12, 9))
plt.subplots_adjust(bottom=0.45) 

# --- 3. CREACIÓN DE LOS EJES PARA LOS WIDGETS ---
ax_escala  = fig.add_axes([0.25, 0.35, 0.5, 0.03])
ax_k       = fig.add_axes([0.25, 0.30, 0.5, 0.03])
ax_epsilon = fig.add_axes([0.25, 0.25, 0.5, 0.03])
ax_lambda  = fig.add_axes([0.25, 0.20, 0.5, 0.03])

ax_radio_metodo = fig.add_axes([0.25, 0.05, 0.2, 0.12])
ax_radio_psf    = fig.add_axes([0.55, 0.05, 0.2, 0.12])

# --- 4. CREACIÓN DE LOS WIDGETS ---
# AL QUITAR 'valstep=1', LOS SLIDERS AHORA SE MUEVEN DE FORMA CONTINUA
slider_escala  = Slider(ax_escala, 'Escala / Sigma', 0, len(escalas)-1, valinit=1)
slider_k       = Slider(ax_k, 'Valor K', 0, len(valores_k)-1, valinit=1)
slider_epsilon = Slider(ax_epsilon, 'Epsilon (solo RL)', 0, len(valores_epsilon)-1, valinit=0)
slider_lambda  = Slider(ax_lambda, 'Índice Lambda', 0, len(lambdas)-1, valinit=0)

radio_metodo = RadioButtons(ax_radio_metodo, metodos)
radio_psf    = RadioButtons(ax_radio_psf, tipos_psf)

# --- 5. LÓGICA DE ACTUALIZACIÓN ---
def update(val):
    metodo = radio_metodo.value_selected
    psf = radio_psf.value_selected
    
    # REDONDEAMOS EL VALOR CONTINUO AL ÍNDICE ENTERO MÁS CERCANO
    idx_escala = int(round(slider_escala.val))
    idx_k      = int(round(slider_k.val))
    idx_eps    = int(round(slider_epsilon.val))
    idx_lam    = int(round(slider_lambda.val))
    
    escala = escalas[idx_escala]
    k = valores_k[idx_k]
    epsilon = valores_epsilon[idx_eps]
    
    # Modificamos el texto mostrado para ver el valor real saltando
    slider_escala.valtext.set_text(f"{escala}")
    slider_k.valtext.set_text(f"{k:.1e}")
    slider_epsilon.valtext.set_text(f"{epsilon}")
    slider_lambda.valtext.set_text(f"λ {idx_lam} ({lambdas[idx_lam]:.3f} Å)")

    # Lógica específica: Formato de Epsilon según el método (Tu corrección anterior)
    if metodo in ['wiener', 'fourier']:
        epsilon_str = "1" # Sin decimal para wiener/fourier
        ax_epsilon.set_alpha(0.3) 
    else:
        epsilon_str = str(epsilon) # Con decimal para RL (ej. "1.0")
        ax_epsilon.set_alpha(1.0)
        
    k_str = f"{k:.1e}"
    
    # Construir la ruta de la imagen
    filename = f"output/comprobacion/mag_{metodo}_{psf}_p{escala}_k{k_str}_eps{epsilon_str}_it{iteraciones}_lam{idx_lam:03d}.png"
    
    # Cargar y mostrar
    ax_img.clear()
    ax_img.axis('off')
    
    if os.path.exists(filename):
        img = mpimg.imread(filename)
        ax_img.imshow(img)
        ax_img.set_title(f"Mostrando: {filename}", fontsize=10)
    else:
        ax_img.text(0.5, 0.5, 'Imagen no encontrada\n\n' + filename, 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax_img.transAxes, color='red', fontsize=12)
        ax_img.set_title("Archivo no encontrado")
        
    fig.canvas.draw_idle()

# --- 6. VINCULAR EVENTOS ---
slider_escala.on_changed(update)
slider_k.on_changed(update)
slider_epsilon.on_changed(update)
slider_lambda.on_changed(update)
radio_metodo.on_clicked(update)
radio_psf.on_clicked(update)

# Llamada inicial
update(None)

plt.show()