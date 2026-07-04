import numpy as np
from astropy.io import fits
import magnetismo as mag
import deconvolucion as decon

def get_kmax_synth():
    datos_cargados = np.load('data/datos_sunspot.npz')
    data = datos_cargados['stokes']
    lambdas = datos_cargados['lam']
    intensidad = data[:, 0, :, :]
    lambdas_absolutas = 6173.0 + (lambdas / 1000.0)
    derivadaI = np.gradient(intensidad, lambdas_absolutas, axis=0)
    constanteK = - 4.67e-13 * 3 * derivadaI * (lambdas_absolutas[:, np.newaxis, np.newaxis]**2)
    return np.abs(constanteK).max()

def get_kmax_real():
    fits_path = 'data/01_QSUN_TM_00_Mg1_10_10072024T131604_LV_1.0_v0.4.fits'
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        intensidad = data[:, 0, :, :]
    lambdas = 5170.0 + (np.arange(10) - 4.5) * 0.05
    derivadaI = np.gradient(intensidad, lambdas, axis=0)
    constanteK = - 4.67e-13 * 1.5 * derivadaI * (lambdas[:, np.newaxis, np.newaxis]**2)
    return np.abs(constanteK).max()

print("K_max Sintetico:", get_kmax_synth())
print("K_max Real:", get_kmax_real())
