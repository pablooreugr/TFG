import numpy as np
from deconvolucion import aplicar_deconvolucion_3d
img3d = np.random.rand(2, 10, 10)
aplicar_deconvolucion_3d(img3d, metodo='w_fran')
