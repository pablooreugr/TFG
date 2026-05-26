import magnetismo as mag
import deconvolucion as deco
import visualizacion as vis
import numpy as np





if __name__ == "__main__":

    # Ruta en tu workspace
    path = "data/datos_sunspot.npz"  # o usa el enlace [data/datos_sunspot.npz](data/datos_sunspot.npz)

    with np.load(path, allow_pickle=True) as npz:
        lambdas = npz['lam']
        datos = npz['stokes']

    intensidad = datos[:, 0, :, :]
    compV = datos[:, 3, :, :]

    # Convertir lambdas relativas (mÅ) a absolutas (Å)
    # Línea a 6173 Å
    lambdas_absolutas = 6173.0 + (lambdas / 1000.0)

    campoMagnetico, r_cuad = mag.calcularMagnetismo(intensidad, compV, lambdas_absolutas)

    #vis.dibujarMagYR(campoMagnetico, r_cuad)

    # A partir de aqui voy a convolucionar una psf con la intensidad, y la V.

    psf = deco.psfAiry(intensidad[0, :, :])
    
    # Preparamos arrays para guardar todas las convoluciones
    intensidad_conv = np.zeros_like(intensidad)
    compV_conv = np.zeros_like(compV)
    
    for i in range(lambdas_absolutas.size):
        print(f"Convolucionando para lambda = {lambdas_absolutas[i]:.3f} Å...")
        intensidad_conv[i] = deco.convolucion(intensidad[i], psf)
        compV_conv[i] = deco.convolucion(compV[i], psf)

    # Mostrar mapas interactivos de la intensidad y componente V convolucionadas
    #vis.visualizarIntensidadYCompVSlider(intensidad_conv, compV_conv, lambdas_absolutas)

    campoMagneticoDec = mag.metodoForw(intensidad_conv, compV_conv, lambdas_absolutas, psf, relLim=1e-50, pasos=50)

    vis.compararMagnetogramas(campoMagnetico, campoMagneticoDec)