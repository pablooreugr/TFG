import magnetismo as mag
import deconvolucion as deco
import numpy as np





if __name__ == "__main__":

    # Ruta en tu workspace
    path = "data/datos_sunspot.npz"  # o usa el enlace [data/datos_sunspot.npz](data/datos_sunspot.npz)
    
    with np.load(path, allow_pickle=True) as npz:
        lambdas = npz['lam']
        datos = npz['stokes']