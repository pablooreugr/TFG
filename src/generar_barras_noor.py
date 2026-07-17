import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import visualizacion as vis
import deconvolucion as decon
from skimage.metrics import structural_similarity as ssim
import magnetismo as mag
import os

sns.set_theme(style="whitegrid")
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output/presentacion'))
os.makedirs(out_dir, exist_ok=True)

def calcular_metricas(original, procesado):
    rmse_val = np.sqrt(np.mean((original - procesado)**2))
    rango_datos = original.max() - original.min()
    if original.ndim == 3:
        ssims = [ssim(original[i], procesado[i], data_range=rango_datos) for i in range(original.shape[0])]
        ssim_val = np.mean(ssims)
    else:
        ssim_val = ssim(original, procesado, data_range=rango_datos)
    return rmse_val, ssim_val

def simular_ruido_telescopio_porcentaje(intensidad, compV, porcentaje_ruido=1):
    I_max = np.max(intensidad)
    V_max = np.max(np.abs(compV))
    sigma_ruido_I = I_max * (porcentaje_ruido / 100.0)
    sigma_ruido_V = V_max * (porcentaje_ruido / 100.0)
    ruido_I = np.random.normal(loc=0.0, scale=sigma_ruido_I, size=intensidad.shape)
    ruido_V = np.random.normal(loc=0.0, scale=sigma_ruido_V, size=compV.shape)
    return intensidad + ruido_I, compV + ruido_V

def save_bar_plot(x, y, ylabel, palette, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=x, y=y, ax=ax, hue=x, palette=palette, legend=False)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Cargando datos y generando barras para Noor...")
    datos_cargados = np.load('data/datos_sunspot.npz')
    data = datos_cargados['stokes']
    lambdas = datos_cargados['lam']
    intensidad = data[:, 0, :, :]
    compV = data[:, 1, :, :]
    lambdas_absolutas = 6173.0 + (lambdas / 1000.0)
    
    psf = decon.generar_psf_airy(tamano_matriz=31, radio_piz=3)
    campoMagnetico, _ = mag.calcularCampoMagnetico(intensidad, compV, lambdas_absolutas)

    intenBorrosa = decon.convolucion3D(intensidad, psf)
    compVborrosa = decon.convolucion3D(compV, psf)
    intenBorrosaRuido, compVborrosaRuido = simular_ruido_telescopio_porcentaje(intenBorrosa, compVborrosa, porcentaje_ruido=0.5)
    
    campoBorroso, _ = mag.calcularCampoMagnetico(intenBorrosaRuido, compVborrosaRuido, lambdas_absolutas)
    
    k_max_actual = mag.calcular_k_max(intenBorrosaRuido, lambdas_absolutas, g=3)
    lambda_dinamico = 0.5 * (k_max_actual**2)
    campoMagDeco, _ = mag.algoritmoDeNoor(
        intenBorrosaRuido, compVborrosaRuido, lambdas_absolutas, psf, 
        pasos=20, trabajadores=-1, pasosFor=30, relLim=1e-30, lambdaReg=lambda_dinamico, cg_auto_close=True
    )
    
    rmse_b, ssim_b = calcular_metricas(campoMagnetico, campoBorroso)
    rmse_d, ssim_d = calcular_metricas(campoMagnetico, campoMagDeco)
    
    metodos = ['Borroso', 'Algoritmo de Noor']
    ssims = [ssim_b, ssim_d]
    rmses = [rmse_b, rmse_d]
    
    colores_viridis = sns.color_palette("viridis", 3)
    paleta_metodos = {
        'Borroso': colores_viridis[0],
        'Algoritmo de Noor': colores_viridis[2]
    }
    
    save_bar_plot(metodos, ssims, 'SSIM (Campo Magnético)', paleta_metodos, 'experimento_noor_barras_ssim.png')
    save_bar_plot(metodos, rmses, 'RMSE (Campo Magnético)', paleta_metodos, 'experimento_noor_barras_rmse.png')
    print("Gráficas guardadas: experimento_noor_barras_ssim.png y experimento_noor_barras_rmse.png")

if __name__ == '__main__':
    main()
