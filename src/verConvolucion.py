import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, RadioButtons
import os

# 1. Definimos los valores exactos que tienes generados
sigmas = [1, 3, 5]
ks = [1e-5, 1e-3, 1e-1, 1]
tipos_psf = ['airy', 'gaussiana']
metodos = ['fourier', 'wiener']

# 2. Configuramos la figura principal
fig, ax = plt.subplots(figsize=(12, 8))
# Dejamos espacio en la parte inferior para colocar los controles
plt.subplots_adjust(left=0.1, bottom=0.4) 
ax.axis('off')

# 3. Creamos los ejes (espacios) para los controles (posición: [izq, abajo, ancho, alto])
ax_sigma = plt.axes([0.2, 0.25, 0.6, 0.03])
ax_k = plt.axes([0.2, 0.18, 0.6, 0.03])
ax_psf = plt.axes([0.2, 0.05, 0.2, 0.1])
ax_metodo = plt.axes([0.5, 0.05, 0.2, 0.1])

# 4. Creamos los controles (Sliders e índices)
# Como los valores de sigma y K saltan (no son continuos), el slider se moverá por el "índice" de tu lista
slider_sigma = Slider(ax_sigma, 'Sigma', 0, len(sigmas)-1, valinit=2, valstep=1)
slider_k = Slider(ax_k, 'Valor K', 0, len(ks)-1, valinit=1, valstep=1)

radio_psf = RadioButtons(ax_psf, tipos_psf, active=0) # Por defecto Airy
radio_metodo = RadioButtons(ax_metodo, metodos, active=1) # Por defecto Wiener

# 5. Función de actualización: se llama cada vez que tocas un control
def update(val):
    # Extraemos los valores reales basándonos en el índice del slider
    sigma_actual = sigmas[int(slider_sigma.val)]
    k_actual = ks[int(slider_k.val)]
    psf_actual = radio_psf.value_selected
    metodo_actual = radio_metodo.value_selected
    
    # Actualizamos el texto del slider para que muestre tu valor real (ej. 1e-5) y no el índice (0, 1, 2...)
    slider_sigma.valtext.set_text(str(sigma_actual))
    slider_k.valtext.set_text(str(k_actual))

    # Construimos el nombre del archivo igual que en tu generador
    nombre_archivo = f'output/deconvolucion/deconvolucion_{metodo_actual}_{psf_actual}_sigma{sigma_actual}_k{k_actual}.png'
    
    ax.clear()
    ax.axis('off')
    
    # Cargamos y mostramos la imagen si existe
    if os.path.exists(nombre_archivo):
        img = mpimg.imread(nombre_archivo)
        ax.imshow(img)
        ax.set_title(f'Mostrando: {nombre_archivo}', fontsize=12)
    else:
        # Si la combinación no existe o el nombre no cuadra, mostramos un aviso en rojo
        ax.text(0.5, 0.5, f'⚠️ Imagen no encontrada:\n{nombre_archivo}', 
                ha='center', va='center', fontsize=14, color='red')
        ax.set_title('Archivo no encontrado', color='red')
        
    fig.canvas.draw_idle()

# 6. Conectamos los controles a la función de actualización
slider_sigma.on_changed(update)
slider_k.on_changed(update)
radio_psf.on_clicked(update)
radio_metodo.on_clicked(update)

# Llamamos a update una vez manualmente para cargar la imagen inicial al abrir
update(None)

# Mostramos la ventana interactiva
plt.show()