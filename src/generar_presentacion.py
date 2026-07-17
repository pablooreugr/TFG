import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import visualizacion as vis
import deconvolucion as decon
from skimage.metrics import structural_similarity as ssim
import magnetismo as mag
import os
from astropy.io import fits
from matplotlib.colors import LogNorm

sns.set_theme(style="whitegrid")
out_dir = '../output/presentacion'
# change out_dir to absolute
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

def save_single_plot(data, cmap, vmin, vmax, filename, is_log=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    if is_log:
        im = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def save_bar_plot(x, y, ylabel, palette, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=x, y=y, ax=ax, hue=x, palette=palette, legend=False)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def run_all():
    print("Cargando datos...")
    datos_cargados = np.load('data/datos_sunspot.npz')
    data = datos_cargados['stokes']
    lambdas = datos_cargados['lam']
    intensidad = data[:, 0, :, :]
    compV = data[:, 1, :, :]
    lambdas_absolutas = 6173.0 + (lambdas / 1000.0)
    psf = decon.generar_psf_airy(tamano_matriz=31, radio_piz=3)
    campoMagnetico, _ = mag.calcularCampoMagnetico(intensidad, compV, lambdas_absolutas)

    print("--- Experimento Noor ---")
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
    
    cmap_b = sns.color_palette("icefire", as_cmap=True)
    vmax_b = np.percentile(np.abs(campoMagnetico), 99.5)
    
    save_single_plot(campoMagnetico, cmap_b, -vmax_b, vmax_b, 'experimento_noor_2_comparativa_b_original.png')
    save_single_plot(campoBorroso, cmap_b, -vmax_b, vmax_b, 'experimento_noor_2_comparativa_b_borroso.png')
    save_single_plot(campoMagDeco, cmap_b, -vmax_b, vmax_b, 'experimento_noor_2_comparativa_b_deconvolucionado.png')
    
    residuos = np.abs(campoMagnetico - campoMagDeco)
    residuos_borroso = np.abs(campoMagnetico - campoBorroso)
    cmap_residuos = sns.color_palette("viridis", as_cmap=True)
    vmax_r = np.percentile(residuos_borroso, 99.5)
    
    save_single_plot(residuos_borroso, cmap_residuos, 0, vmax_r, 'experimento_noor_3_residuos_borroso.png')
    save_single_plot(residuos, cmap_residuos, 0, vmax_r, 'experimento_noor_3_residuos_deconvolucionado.png')
    
    y_ini, y_fin = 140, 235
    x_ini, x_fin = 163, 255
    save_single_plot(campoMagnetico[y_ini:y_fin, x_ini:x_fin], cmap_b, -vmax_b, vmax_b, 'experimento_noor_2b_comparativa_b_zoom_original.png')
    save_single_plot(campoBorroso[y_ini:y_fin, x_ini:x_fin], cmap_b, -vmax_b, vmax_b, 'experimento_noor_2b_comparativa_b_zoom_borroso.png')
    save_single_plot(campoMagDeco[y_ini:y_fin, x_ini:x_fin], cmap_b, -vmax_b, vmax_b, 'experimento_noor_2b_comparativa_b_zoom_deconvolucionado.png')
    
    save_single_plot(residuos_borroso[y_ini:y_fin, x_ini:x_fin], cmap_residuos, 0, vmax_r, 'experimento_noor_3b_residuos_zoom_borroso.png')
    save_single_plot(residuos[y_ini:y_fin, x_ini:x_fin], cmap_residuos, 0, vmax_r, 'experimento_noor_3b_residuos_zoom_deconvolucionado.png')

    print("--- Experimento Comparacion Intensidad ---")
    dummy_compV = np.zeros_like(intensidad)
    intenBorrosaRuido2, _ = simular_ruido_telescopio_porcentaje(intenBorrosa, dummy_compV, porcentaje_ruido=0.1)
    inten_rl = decon.deconvolucion3D(intenBorrosaRuido2, psf, metodo='rl', pasos=20)
    inten_fourier = decon.deconvolucion3D(intenBorrosaRuido2, psf, metodo='fourier')
    inten_wiener = decon.deconvolucion3D(intenBorrosaRuido2, psf, metodo='wiener')
    
    idx_z = intensidad.shape[0] // 2
    cmap_i = sns.color_palette("rocket", as_cmap=True)
    vmin_i = None; vmax_i = None
    save_single_plot(intensidad[idx_z], cmap_i, vmin_i, vmax_i, 'comparacion_intensidad_1_original.png')
    save_single_plot(intenBorrosaRuido2[idx_z], cmap_i, vmin_i, vmax_i, 'comparacion_intensidad_1_borrosa.png')
    
    save_single_plot(inten_rl[idx_z], cmap_i, vmin_i, vmax_i, 'comparacion_intensidad_2_deconvolucion_rl.png')
    save_single_plot(inten_fourier[idx_z], cmap_i, vmin_i, vmax_i, 'comparacion_intensidad_2_deconvolucion_fourier.png')
    save_single_plot(inten_wiener[idx_z], cmap_i, vmin_i, vmax_i, 'comparacion_intensidad_2_deconvolucion_wiener.png')
    
    metodos = ['Borroso + Ruido', 'Richardson-Lucy', 'Wiener', 'Fourier']
    imagenes = [intenBorrosaRuido2, inten_rl, inten_wiener, inten_fourier]
    rmses, ssims = [], []
    for img in imagenes:
        r, s = calcular_metricas(intensidad, img)
        rmses.append(r); ssims.append(s)
    
    colores_viridis = sns.color_palette("viridis", 3)
    paleta_metodos = {'Borroso + Ruido': colores_viridis[0], 'Richardson-Lucy': colores_viridis[1], 'Wiener': colores_viridis[2], 'Fourier': '#d62728'}
    save_bar_plot(metodos, ssims, 'SSIM', paleta_metodos, 'comparacion_intensidad_3_barras_ssim.png')
    save_bar_plot(metodos, rmses, 'RMSE', paleta_metodos, 'comparacion_intensidad_3_barras_rmse.png')
    
    save_single_plot(psf, sns.color_palette("mako", as_cmap=True), 1e-5, psf.max(), 'comparacion_intensidad_4_psf.png', is_log=True)
    
    save_single_plot(intensidad[idx_z, y_ini:y_fin, x_ini:x_fin], cmap_i, vmin_i, vmax_i, 'comparacion_intensidad_5_original_zoom.png')
    save_single_plot(intenBorrosaRuido2[idx_z, y_ini:y_fin, x_ini:x_fin], cmap_i, vmin_i, vmax_i, 'comparacion_intensidad_5_borrosa_zoom.png')
    
    save_single_plot(inten_rl[idx_z, y_ini:y_fin, x_ini:x_fin], cmap_i, vmin_i, vmax_i, 'comparacion_intensidad_6_deconvolucion_rl_zoom.png')
    save_single_plot(inten_fourier[idx_z, y_ini:y_fin, x_ini:x_fin], cmap_i, vmin_i, vmax_i, 'comparacion_intensidad_6_deconvolucion_fourier_zoom.png')
    save_single_plot(inten_wiener[idx_z, y_ini:y_fin, x_ini:x_fin], cmap_i, vmin_i, vmax_i, 'comparacion_intensidad_6_deconvolucion_wiener_zoom.png')

    print("--- Experimento Stokes V ---")
    intensidad0 = data[0, 0, :, :]
    compV0 = data[0, 1, :, :]
    intensidad_3d = np.expand_dims(intensidad0, axis=0)
    compV_3d = np.expand_dims(compV0, axis=0)
    intensidad_borrosa_3d = decon.convolucion3D(intensidad_3d, psf)
    compV_borrosa_3d = decon.convolucion3D(compV_3d, psf)
    _, compV_borrosa_ruido_3d = simular_ruido_telescopio_porcentaje(intensidad_borrosa_3d, compV_borrosa_3d, porcentaje_ruido=2.0)
    compV_borrosa_ruido = compV_borrosa_ruido_3d[0]
    compV_deco_wiener = decon.deconvolucion3D(compV_borrosa_ruido_3d, psf, metodo='wiener')[0]
    
    cmap_v = sns.diverging_palette(145, 300, s=85, l=50, center="light", as_cmap=True)
    v_max = np.percentile(np.abs(compV0), 99.0)
    save_single_plot(compV0, cmap_v, -v_max, v_max, 'experimento_stokesV_2_visualizacion_original.png')
    save_single_plot(compV_borrosa_ruido, cmap_v, -v_max, v_max, 'experimento_stokesV_2_visualizacion_borroso.png')
    save_single_plot(compV_deco_wiener, cmap_v, -v_max, v_max, 'experimento_stokesV_2_visualizacion_wiener.png')
    
    print("--- Experimento Datos Reales ---")
    psf_bruta = np.load('data/PSF_517_1600_x_1600_px.npy')
    cy, cx = psf_bruta.shape[0] // 2, psf_bruta.shape[1] // 2
    radio = 30
    psf_recortada = psf_bruta[cy-radio:cy+radio+1, cx-radio:cx+radio+1]
    psf_real = psf_recortada / np.sum(psf_recortada)
    
    with fits.open('data/01_QSUN_TM_00_Mg1_10_10072024T131604_LV_1.0_v0.4.fits') as hdul:
        data_real = hdul[0].data
        intensidad_real = data_real[:, 0, :, :]
        compV_real = data_real[:, 3, :, :]
    
    lambdas_real = 5170.0 + (np.arange(10) - 4.5) * 0.05
    I_centro = intensidad_real[5, :, :]
    I_centro_3d = np.expand_dims(I_centro, axis=0)
    I_centro_rl = decon.deconvolucion3D(I_centro_3d, psf_real, metodo='rl', pasos=15)[0]
    
    cmap_real_i = sns.color_palette("magma", as_cmap=True)
    vmin_r, vmax_r_i = np.percentile(I_centro, [1, 99.5])
    save_single_plot(I_centro, cmap_real_i, vmin_r, vmax_r_i, 'exp4_1_intensidad_original.png')
    save_single_plot(I_centro_rl, cmap_real_i, vmin_r, vmax_r_i, 'exp4_1_intensidad_rl.png')
    
    save_single_plot(psf_real, sns.color_palette("mako", as_cmap=True), 1e-5, psf_real.max(), 'exp4_4_psf.png', is_log=True)
    
    B_completo_crudo, _ = mag.calcularCampoMagnetico(intensidad_real, compV_real, lambdas_real, g=1.5)
    best_y, best_x = 800, 800
    size = 200
    y_min, y_max = max(0, best_y-size), min(1600, best_y+size)
    x_min, x_max = max(0, best_x-size), min(1600, best_x+size)
    intensidad_roi = intensidad_real[:, y_min:y_max, x_min:x_max]
    compV_roi = compV_real[:, y_min:y_max, x_min:x_max]
    
    B_directo_roi, _ = mag.calcularCampoMagnetico(intensidad_roi, compV_roi, lambdas_real, g=1.5)
    k_max_actual_real = mag.calcular_k_max(intensidad_roi, lambdas_real, g=1.5)
    lambda_dinamico_real = 0.5 * (k_max_actual_real**2)
    B_noor_roi, _ = mag.algoritmoDeNoor(
        intensidad_roi, compV_roi, lambdas_real, psf_real, g=1.5, 
        lambdaReg=lambda_dinamico_real, relLim=1e-6, pasosFor=100, cg_auto_close=True
    )
    
    b_max_real = np.percentile(np.abs(B_completo_crudo), 99.5)
    save_single_plot(B_directo_roi, cmap_b, -b_max_real, b_max_real, 'exp4_3_campoB_zoom_directo.png')
    save_single_plot(B_noor_roi, cmap_b, -b_max_real, b_max_real, 'exp4_3_campoB_zoom_noor.png')
    
    diferencia_B_abs = np.abs(B_noor_roi - B_directo_roi)
    dif_max = np.percentile(diferencia_B_abs, 99.5)
    save_single_plot(diferencia_B_abs, sns.color_palette("viridis", as_cmap=True), 0, dif_max, 'exp4_5_diferencia.png')

if __name__ == '__main__':
    run_all()
