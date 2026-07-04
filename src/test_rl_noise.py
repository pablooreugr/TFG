import numpy as np
import deconvolucion as decon
from skimage.metrics import structural_similarity as ssim

datos_cargados = np.load('../data/datos_sunspot.npz')
intensidad = datos_cargados['stokes'][:, 0, :, :]
psf = decon.generar_psf_airy(tamano_matriz=31, radio_piz=3)

intenBorrosa = decon.convolucion3D(intensidad, psf)
inten_rl_clean = decon.deconvolucion3D(intenBorrosa, psf, metodo='rl', pasos=20)
rango = intensidad.max() - intensidad.min()

ssims = [ssim(intensidad[i], inten_rl_clean[i], data_range=rango) for i in range(intensidad.shape[0])]
print(f"SSIM sin ruido (RL): {np.mean(ssims):.4f}")

ruido = np.random.normal(0, 0.02 * np.max(intensidad), intensidad.shape)
intenBorrosaRuido = intenBorrosa + ruido

inten_rl_noisy = decon.deconvolucion3D(intenBorrosaRuido, psf, metodo='rl', pasos=20)
ssims_noisy = [ssim(intensidad[i], inten_rl_noisy[i], data_range=rango) for i in range(intensidad.shape[0])]
print(f"SSIM con ruido (RL): {np.mean(ssims_noisy):.4f}")
