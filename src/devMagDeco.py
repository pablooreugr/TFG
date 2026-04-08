from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import deconvolucion as decon


g_eff = 1.75 # Linea del magnesio I
constanteFormula = 4.67e-13 

def decWienerAxis0(imagen, psf, k=1e-3):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionWiener(imagen[i], psf, k)

    return imagen

def decWienerAxis0Multi(imagen, psf, k=1e-3):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionWienerMulti(imagen[i], psf, k)

    return imagen

def decRLAxis0(imagen, psf, iteraciones=500, k=1e-3, epsilon=1):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionRL(imagen[i], psf, iteraciones, k, epsilon)

    return imagen

def decRLAxis0Multi(imagen, psf, iteraciones=500, k=1e-3, epsilon=1):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionRLMulti(imagen[i], psf, iteraciones, k, epsilon)

    return imagen

def decFourierAxis0(imagen, psf, k=1e-3):
    for i in range(imagen.shape[0]):
        imagen[i] = decon.deconvolucionFourier(imagen[i], psf, k)

    return imagen

def decFourierAxis0Multi(imagen, psf, k=1e-3):
    for i in range(imagen.shape[0]):
        print(f"Deconvolución Fourier (Multi) - Lambda {i}...")
        imagen[i] = decon.deconvolucionFourierMulti(imagen[i], psf, k)

    return imagen


def magnetismoDirectamente(imagen, psf, lambdas, metDecon='wiener', iteraciones=1000, k=1e-3, epsilon=1):
    imagenIntensidad = imagen[:, 0, :, :]
    imagenV = imagen[:, 3, :, :]

    if metDecon == 'wiener':
        imagenIntensidad = decWienerAxis0(imagenIntensidad, psf, k)
        imagenV = decWienerAxis0(imagenV, psf, k)
    elif metDecon == 'fourier':
        imagenIntensidad = decFourierAxis0(imagenIntensidad, psf, k)
        imagenV = decFourierAxis0(imagenV, psf, k)
    elif metDecon == 'rl':
        imagenIntensidad = decRLAxis0(imagenIntensidad, psf, iteraciones, k, epsilon)
        imagenV = decRLAxis0(imagenV, psf, iteraciones, k, epsilon)

    lambdas3D = lambdas[:, np.newaxis, np.newaxis]

    gradIntensidad = np.gradient(imagenIntensidad, lambdas, axis=0) * (lambdas3D**2)

    # A partir de ahora calculamos la regresionLineal
    m = np.sum(gradIntensidad*imagenV, axis=0)/np.sum(gradIntensidad**2, axis=0)

    v_predicho = gradIntensidad * m

    mediaDatosV = np.mean(imagenV, axis=0)
    
    # Numerador: Suma de los residuos al cuadrado
    numerador = np.sum((imagenV - v_predicho)**2, axis=0)
    
    # Denominador: Suma de la varianza total
    denominador = np.sum((imagenV - mediaDatosV)**2, axis=0)
    
    # Calculamos R^2
    mapa_r_cuadrado = 1 - (numerador / denominador)

    campoMagnetico = -m*(1/(g_eff*constanteFormula))

    return campoMagnetico, mapa_r_cuadrado


def magnetismoDirectamenteMulti(imagen, psf, lambdas, metDecon='wiener', iteraciones=1000, k=1e-3, epsilon=1): # En este calculamos la deconvolucion de V y de I directamente, luego sacamos por regresion lineal el magnetismo
    imagenIntensidad = imagen[:, 0, :, :]
    imagenV = imagen[:, 3, :, :]

    if metDecon == 'wiener':
        imagenIntensidad = decWienerAxis0Multi(imagenIntensidad, psf, k)
        imagenV = decWienerAxis0Multi(imagenV, psf, k)
    elif metDecon == 'fourier':
        imagenIntensidad = decFourierAxis0Multi(imagenIntensidad, psf, k)
        imagenV = decFourierAxis0Multi(imagenV, psf, k)
    elif metDecon == 'rl':
        imagenIntensidad = decRLAxis0Multi(imagenIntensidad, psf, iteraciones, k, epsilon)
        imagenV = decRLAxis0Multi(imagenV, psf, iteraciones, k, epsilon)

    lambdas3D = lambdas[:, np.newaxis, np.newaxis]

    gradIntensidad = np.gradient(imagenIntensidad, lambdas, axis=0) * (lambdas3D**2)

    # A partir de ahora calculamos la regresionLineal
    m = np.sum(gradIntensidad*imagenV, axis=0)/np.sum(gradIntensidad**2, axis=0)

    v_predicho = gradIntensidad * m

    mediaDatosV = np.mean(imagenV, axis=0)
    
    # Numerador: Suma de los residuos al cuadrado
    numerador = np.sum((imagenV - v_predicho)**2, axis=0)
    
    # Denominador: Suma de la varianza total
    denominador = np.sum((imagenV - mediaDatosV)**2, axis=0)
    
    # Calculamos R^2 (NumPy resta el 1 a cada píxel automáticamente)
    mapa_r_cuadrado = 1 - (numerador / denominador)

    campoMagnetico = -m*(1/(g_eff*constanteFormula))

    return campoMagnetico, mapa_r_cuadrado



def dibujarMagYR(campoMagnetico, mapa_r_cuadrado):
    # Representacion
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- PRIMER RECUADRO (ax1): El Magnetograma ---
    im1 = ax1.imshow(campoMagnetico, cmap='RdBu_r') 
    fig.colorbar(im1, ax=ax1, label='Valor del campo magnético paralelo G (Gauss)') 

    # --- SEGUNDO RECUADRO (ax2): El mapa de R^2 ---
    im2 = ax2.imshow(mapa_r_cuadrado, vmin=0, vmax=1, cmap='viridis')
    ax2.set_title('Mapa de Fiabilidad (R^2)')
    fig.colorbar(im2, ax=ax2, label='R^2')


    plt.tight_layout()  #ajusta los márgenes automáticamente para que los títulos y las barras de color no se superpongan entre sí.

    plt.show()


def generar_comprobaciones_multi(imagen_datos, lambdas, metDecon='wiener', 
                                 tipo_psf='airy', param_psf=3, 
                                 iteraciones=1000, k=1e-3, epsilon=1, 
                                 output_dir='output/comprobacion'):
    """
    Calcula la PSF internamente, realiza la deconvolución paralela, calcula 
    el campo magnético junto con su mapa R², y guarda un plot 2x4 para cada longitud de onda.
    """
    # 1. Extraemos las imágenes originales
    I_orig = np.copy(imagen_datos[:, 0, :, :])
    V_orig = np.copy(imagen_datos[:, 3, :, :])

    # 2. Calculamos la PSF internamente
    print(f"Generando PSF de tipo '{tipo_psf}' con parámetro {param_psf}...")
    tipo_psf = tipo_psf.lower()
    if tipo_psf == 'gaussiana':
        psf = decon.psfGaussiana(imagen_datos, sigma=param_psf)
    elif tipo_psf == 'airy':
        psf = decon.psfAiry(I_orig[0], escala=1.32/param_psf)
    elif tipo_psf == 'niandrejo':
        psf = decon.psfNiandrejo(I_orig[0], escala=param_psf)
    else:
        raise ValueError("⚠️ Tipo de PSF no reconocido. Usa 'gaussiana', 'airy' o 'niandrejo'.")

    # 3. Deconvolucionamos
    print(f"Iniciando deconvolución de I y V con método '{metDecon}' (Multi)...")
    if metDecon == 'wiener':
        I_decon = decWienerAxis0Multi(np.copy(I_orig), psf, k)
        V_decon = decWienerAxis0Multi(np.copy(V_orig), psf, k)
    elif metDecon == 'fourier':
        I_decon = decFourierAxis0Multi(np.copy(I_orig), psf, k)
        V_decon = decFourierAxis0Multi(np.copy(V_orig), psf, k)
    elif metDecon == 'rl':
        print("Deconvolución RL - Intensidad...")
        I_decon = decRLAxis0Multi(np.copy(I_orig), psf, iteraciones, k, epsilon)

        V_pos = np.where(V_orig > 0, V_orig, 0)

        V_neg = np.where(V_orig < 0, abs(V_orig), 0)

        print("Deconvolución RL - V (positivo)...")

        V_pos = decRLAxis0Multi(np.copy(V_pos), psf, iteraciones, k, epsilon)

        print("Deconvolución RL - V (negativo)...")

        V_neg = decRLAxis0Multi(np.copy(V_neg), psf, iteraciones, k, epsilon)

        V_decon = V_pos - V_neg

    else:
        raise ValueError("Método no reconocido. Usa 'wiener', 'fourier' o 'rl'.")

    # 4. Calculamos el campo magnético y el mapa de R²
    print("Calculando el campo magnético y el mapa de fiabilidad R²...")
    lambdas3D = lambdas[:, np.newaxis, np.newaxis]
    gradIntensidad = np.gradient(I_decon, lambdas, axis=0) * (lambdas3D**2)
    
    m = np.sum(gradIntensidad * V_decon, axis=0) / np.sum(gradIntensidad**2, axis=0)
    
    g_eff = 1.75 
    constanteFormula = 4.67e-13 
    Mag = -m * (1 / (g_eff * constanteFormula))

    # --- NUEVO CÁLCULO DE R² ---
    v_predicho = gradIntensidad * m
    mediaDatosV = np.mean(V_decon, axis=0)
    
    numerador = np.sum((V_decon - v_predicho)**2, axis=0)
    denominador = np.sum((V_decon - mediaDatosV)**2, axis=0)
    
    # Ignoramos advertencias matemáticas (división por cero en pixeles sin datos)
    with np.errstate(divide='ignore', invalid='ignore'):
        mapa_r_cuadrado = 1 - (numerador / denominador)
        mapa_r_cuadrado = np.nan_to_num(mapa_r_cuadrado, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. Generamos y guardamos los gráficos
    num_lambdas = I_orig.shape[0]
    print(f"Generando y guardando {num_lambdas} gráficos. Por favor, espera...")
    
    for i in range(num_lambdas):
        # Ampliamos a 4 columnas y ajustamos el tamaño
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(24, 10))
        fig.suptitle(f'Comprobación - Lambda: {i} ({lambdas[i]:.4f} Å) | {metDecon.upper()} | PSF: {tipo_psf.capitalize()}', fontsize=16)

        # --- FILA SUPERIOR ---
        im0 = axs[0, 0].imshow(I_orig[i], cmap='hot', origin='lower')
        axs[0, 0].set_title('I Original')
        axs[0, 0].set_xlim(600, 800)
        axs[0, 0].set_ylim(500, 700)
        fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

        im1 = axs[0, 1].imshow(V_orig[i], cmap='Purples_r', origin='lower')
        axs[0, 1].set_title('V Original')
        axs[0, 1].set_xlim(600, 800)
        axs[0, 1].set_ylim(500, 700)
        fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

        im2 = axs[0, 2].imshow(psf, cmap='hot', origin='lower')
        axs[0, 2].set_title(f'PSF ({tipo_psf})')
        ny, nx = psf.shape
        cy, cx = ny // 2, nx // 2
        axs[0, 2].set_xlim(cx - 100, cx + 100)
        axs[0, 2].set_ylim(cy - 100, cy + 100)
        fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

        # Apagamos el panel que sobra arriba a la derecha
        axs[0, 3].axis('off')

        # --- FILA INFERIOR ---
        im3 = axs[1, 0].imshow(I_decon[i], cmap='hot', origin='lower')
        axs[1, 0].set_title('DI (I Deconvolucionada)')
        axs[1, 0].set_xlim(600, 800)
        axs[1, 0].set_ylim(500, 700)
        fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

        im4 = axs[1, 1].imshow(V_decon[i], cmap='Purples_r', origin='lower')
        axs[1, 1].set_title('DV (V Deconvolucionada)')
        axs[1, 1].set_xlim(600, 800)
        axs[1, 1].set_ylim(500, 700)
        fig.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

        im5 = axs[1, 2].imshow(Mag, cmap='RdBu_r', origin='lower')
        axs[1, 2].set_title('Mag (Campo Magnético G)')
        axs[1, 2].set_xlim(600, 800)
        axs[1, 2].set_ylim(500, 700)
        fig.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)

        # --- NUEVO PANEL R² ---
        im6 = axs[1, 3].imshow(mapa_r_cuadrado, cmap='viridis', vmin=0, vmax=1, origin='lower')
        axs[1, 3].set_title('Mapa de Fiabilidad (R²)')
        axs[1, 3].set_xlim(600, 800)
        axs[1, 3].set_ylim(500, 700)
        fig.colorbar(im6, ax=axs[1, 3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        
        # Guardamos usando la ruta que especificaste
        nombre_archivo = f"output/pruebaV2/mag_{metDecon}_{tipo_psf}_p{param_psf}_k{k:.1e}_eps{epsilon}_it{iteraciones}_lam{i:03d}.png"

        plt.savefig(nombre_archivo, dpi=150)
        plt.close(fig) 
        
    print(f"✅ ¡Completado!")


if __name__ == "__main__":
    import os
    print("Iniciando el programa de comprobación masiva...")
    
    # 1. Cargamos los datos
    ruta_archivo = 'data/prueba.fits'
    with fits.open(ruta_archivo) as hdul:
        datos = hdul[0].data
        cabecera = hdul[0].header

    # Extraemos el eje lambda de la cabecera
    eje_lambda = np.array([cabecera[f'L_{i}'] for i in range(datos.shape[0])])

    # 2. Definimos las listas de parámetros para el barrido
    #metodos = ['fourier', 'wiener', 'rl']
    metodos = ['rl']
    tipos_psf = ['airy', 'gaussiana']  # Asumiendo 'airy' y 'gaussiana' como las 2 PSFs
    params_psf = [1.0, 3.0, 5.0]
    valores_k = [1e-4, 1e-3, 1e-2]
    iteraciones_rl = [15, 30, 50]
    
    # Parámetro constante basado en tu ejemplo
    epsilon_base = 1.0e-12 

    # Asegurarnos de que el directorio de salida (hardcodeado en tu función) existe
    os.makedirs('output/pruebaV2', exist_ok=True)

    # 3. Bucle de barrido
    for metodo in metodos:
        for psf in tipos_psf:
            for p_psf in params_psf:
                for k_val in valores_k:
                    
                    # Si el método es RL, iteramos también sobre la lista de iteraciones
                    if metodo == 'rl':
                        for iters in iteraciones_rl:
                            print(f"\n=== Ejecutando: {metodo.upper()} | PSF: {psf} (p={p_psf}) | k={k_val:.1e} | Iters={iters} ===")
                            generar_comprobaciones_multi(
                                imagen_datos=datos,
                                lambdas=eje_lambda,
                                metDecon=metodo,
                                tipo_psf=psf,
                                param_psf=p_psf,
                                iteraciones=iters,
                                k=k_val,
                                epsilon=epsilon_base,
                                output_dir='output/comprobacion'
                            )
                    
                    # Si es Fourier o Wiener, ejecutamos solo una vez (sin importar iteraciones)
                    else:
                        print(f"\n=== Ejecutando: {metodo.upper()} | PSF: {psf} (p={p_psf}) | k={k_val:.1e} ===")
                        generar_comprobaciones_multi(
                            imagen_datos=datos,
                            lambdas=eje_lambda,
                            metDecon=metodo,
                            tipo_psf=psf,
                            param_psf=p_psf,
                            iteraciones=0, # Valor dummy, Fourier y Wiener no lo usan
                            k=k_val,
                            epsilon=epsilon_base,
                            output_dir='output/comprobacion'
                        )
    
    print("\n✅ ¡Barrido de parámetros completado con éxito!")


