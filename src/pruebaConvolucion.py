import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import deconvolucion
import scipy.fft as fft

# ----------------------------
# 1. GENERAR IMAGEN LIMPIA
# ----------------------------
def generar_imagen_sintetica(nx=512, ny=512, n_estrellas=20):
    imagen = np.zeros((nx, ny))
    
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for _ in range(n_estrellas):
        x0 = np.random.randint(0, nx)
        y0 = np.random.randint(0, ny)
        intensidad = np.random.uniform(1, 10)
        sigma = np.random.uniform(1, 3)
        
        estrella = intensidad * np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2))
        imagen += estrella
    
    return imagen



def generar_granulacion_solar(nx=1600, ny=1600):
    
    def campo_filtrado(escala):
        ruido = np.random.normal(0, 1, (nx, ny))
        ruido_fft = fft.fft2(ruido)
        
        kx = np.fft.fftfreq(nx)
        ky = np.fft.fftfreq(ny)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2
        
        filtro = np.exp(-K2 * (escala**2))
        return np.real(fft.ifft2(ruido_fft * filtro))
    
    # ------------------------
    # MULTIESCALA
    # ------------------------
    pequeña = campo_filtrado(escala=8)     # gránulos pequeños
    media   = campo_filtrado(escala=20)    # granulación típica
    grande  = campo_filtrado(escala=60)    # estructuras grandes
    
    imagen = 0.6 * media + 0.25 * pequeña + 0.15 * grande
    
    # ------------------------
    # NORMALIZACIÓN
    # ------------------------
    imagen -= imagen.min()
    imagen /= imagen.max()
    
    # ------------------------
    # NO LINEALIDAD (clave)
    # ------------------------
    imagen = imagen**1.8
    
    # ------------------------
    # ENFATIZAR BORDES (intergranular lanes)
    # ------------------------
    grad_x = np.gradient(imagen, axis=0)
    grad_y = np.gradient(imagen, axis=1)
    borde = np.sqrt(grad_x**2 + grad_y**2)
    
    imagen = imagen - 0.3 * borde
    
    # ------------------------
    # NORMALIZACIÓN FINAL
    # ------------------------
    imagen -= imagen.min()
    imagen /= imagen.max()
    
    return imagen


# ----------------------------
# 2. RUIDO
# ----------------------------
def añadir_ruido_poisson(imagen):
    imagen = imagen - np.min(imagen)
    imagen = imagen / np.max(imagen)
    
    escala = 1000
    imagen_escalada = imagen * escala
    
    ruido = np.random.poisson(imagen_escalada)
    
    return ruido / escala


# ----------------------------
# 3. MÉTRICAS
# ----------------------------
def mse(im1, im2):
    return np.mean((im1 - im2)**2)

def psnr(im1, im2):
    error = mse(im1, im2)
    if error == 0:
        return np.inf
    return 10 * np.log10(np.max(im1)**2 / error)


# ----------------------------
# 4. EXPERIMENTO
# ----------------------------
def experimento():
    
    # Imagen limpia
    imagen_limpia = generar_granulacion_solar()
    
    # PSF (usa la tuya)
    psf = deconvolucion.psfGaussiana(
        imagen_limpia.reshape(1,1,*imagen_limpia.shape), 
        sigma=2
    )
    
    # Degradación
    imagen_borrosa = fftconvolve(imagen_limpia, psf, mode='same')
    imagen_ruidosa = añadir_ruido_poisson(imagen_borrosa)
    
    # ------------------------
    # MÉTODOS
    # ------------------------
    
    rec_fourier = deconvolucion.deconvolucionFourierMulti(imagen_ruidosa, psf)
    rec_wiener = deconvolucion.deconvolucionWienerMulti(imagen_ruidosa, psf)
    rec_rl = deconvolucion.deconvolucionRLMulti(imagen_ruidosa, psf, pasos=50)
    
    # ------------------------
    # EVALUACIÓN
    # ------------------------
    
    metodos = {
        "Fourier": rec_fourier,
        "Wiener": rec_wiener,
        "Richardson-Lucy": rec_rl
    }
    
    print("\n📊 RESULTADOS:")
    for nombre, img in metodos.items():
        print(f"{nombre}:")
        print(f"   MSE  = {mse(imagen_limpia, img):.6e}")
        print(f"   PSNR = {psnr(imagen_limpia, img):.2f} dB\n")
    
    # ------------------------
    # VISUALIZACIÓN
    # ------------------------
    
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    
    axs[0].imshow(imagen_limpia, cmap='hot')
    axs[0].set_title("Original")
    
    axs[1].imshow(imagen_ruidosa, cmap='hot')
    axs[1].set_title("Ruidosa")
    
    axs[2].imshow(rec_fourier, cmap='hot')
    axs[2].set_title("Fourier")
    
    axs[3].imshow(rec_wiener, cmap='hot')
    axs[3].set_title("Wiener")
    
    axs[4].imshow(rec_rl, cmap='hot')
    axs[4].set_title("RL")
    
    for ax in axs:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# ----------------------------
# 5. RUN
# ----------------------------
if __name__ == "__main__":
    experimento()