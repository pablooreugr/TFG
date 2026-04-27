# Guía de Uso: Deconvolución 3D y Múltiple

Esta guía explica detalladamente cómo utilizar las funciones `deconvolucionMulti` y `aplicar_deconvolucion_3d` que se encuentran en el archivo `src/deconvolucion.py`. Estas funciones están diseñadas para aplicar diferentes algoritmos de deconvolución de forma optimizada, con soporte multinúcleo y tratamiento adecuado para imágenes con valores negativos (como perfiles de Stokes).

## 1. Función `deconvolucionMulti`

Esta función realiza la deconvolución en una imagen 2D. Lo que la hace especial es que separa la imagen en su parte positiva y negativa, aplica la deconvolución a ambas por separado y luego las resta. Esto es crítico en datos espectropolarimétricos (como Stokes V) donde existen valores de intensidad negativos.

### Parámetros principales

*   `imagen` (ndarray 2D): La matriz 2D de la imagen u observación borrosa.
*   `psf` (ndarray 2D): La Point Spread Function (Función de Dispersión de Punto). Requerida para todos los métodos excepto para `w_fran`.
*   `metodo` (str): El algoritmo a utilizar. Opciones válidas:
    *   `'rl'` o `'richardson-lucy'`: Richardson-Lucy.
    *   `'w'` o `'wiener'`: Filtro de Wiener propio.
    *   `'f'` o `'fourier'`: División directa en Fourier.
    *   `'w_fran'`: Método Wiener de Fran (requiere coeficientes de Zernike).
    *   `'w_skimage'`: Wiener de la librería scikit-image.
*   `k` (float): Constante de ruido utilizada por la mayoría de algoritmos para evitar la división por cero o amplificar ruido. (Por defecto `1e-3`).
*   `iteraciones` (int): Número de pasos para el algoritmo Richardson-Lucy (`'rl'`). (Por defecto `30`).
*   `epsilon` (float): Criterio de parada para RL por si el error es suficientemente pequeño. (Por defecto `1`).
*   `zernikes` (array): Vector de coeficientes Zernike. Requerido exclusivamente por el método `'w_fran'`.
*   `workers` (int): Número de hilos/núcleos a usar (`-1` para todos los disponibles).

### Ejemplo de uso (2D)

```python
from src.deconvolucion import deconvolucionMulti, psfAiry
import numpy as np

# Suponiendo que tienes una imagen de 100x100
imagen_2d = np.random.rand(100, 100) 
mi_psf = psfAiry(imagen_2d, escala=0.5)

# 1. Usar Wiener (Multinúcleo)
resultado_w = deconvolucionMulti(imagen_2d, psf=mi_psf, metodo='w', k=1e-3, workers=-1)

# 2. Usar Richardson-Lucy
resultado_rl = deconvolucionMulti(imagen_2d, psf=mi_psf, metodo='rl', iteraciones=50, workers=-1)
```

---

## 2. Función `aplicar_deconvolucion_3d`

Esta función extiende `deconvolucionMulti` para operar sobre un "cubo" de datos 3D. El cubo tiene típicamente la forma `[índice_espectral, y, x]`. La función itera a lo largo del eje 0 (espectral o de tiempo) aplicando el método elegido a cada marco espacial 2D.

### Parámetros principales

*   `imagen_3d` (ndarray 3D): El cubo de datos, donde `imagen_3d.shape[0]` es el número de imágenes a procesar.
*   `psf` (ndarray 2D): La PSF 2D que se aplicará uniformemente a todos los "cortes" o fotogramas del cubo.
*   `metodo` (str): Algoritmo de deconvolución (mismas opciones que en `deconvolucionMulti`). Por defecto es `'w'`.
*   Resto de parámetros (`iteraciones`, `epsilon`, `zernikes`, `k`, `workers`): Tienen el mismo significado que en `deconvolucionMulti` y se transmiten internamente a cada imagen 2D.

> [!NOTE]
> **Sobre la paralelización en 3D:**
> - Para métodos tradicionales (`'w'`, `'rl'`, `'f'`), la paralelización `workers=-1` se aplica en el nivel de las operaciones matemáticas 2D interiores (Fourier, NumExpr) de cada fotograma secuencialmente.
> - Para el método de Fran (`'w_fran'`), la función emplea `ProcessPoolExecutor` para distribuir los *diferentes fotogramas 2D* a través de varios procesos simultáneamente.

### Ejemplo de uso (3D)

```python
from src.deconvolucion import aplicar_deconvolucion_3d, psfGaussiana
from astropy.io import fits
import numpy as np

# 1. Cargar el cubo de datos (por ejemplo, [lambda, y, x])
# Asegúrate de extraer solo la parte de intensidad si el cubo es 4D [Stokes, lambda, y, x]
with fits.open('data/observacion.fits') as hdul:
    # Seleccionamos Stokes I (índice 0), y todos los lambdas, x, y
    cubo_3d = hdul[0].data[0, :, :, :] 

# 2. Generar PSF en base a las dimensiones espaciales de un frame
psf_2d = psfGaussiana(cubo_3d[0], sigma=2.5)

# 3. Aplicar deconvolución 3D (Richardson-Lucy con 20 pasos)
# Modificará cada imagen del cubo y devolverá el cubo completo procesado.
cubo_procesado_rl = aplicar_deconvolucion_3d(
    imagen_3d = cubo_3d.copy(), # .copy() para no sobreescribir el original
    psf = psf_2d,
    metodo = 'rl',
    iteraciones = 20,
    k = 1e-4,
    workers = -1  # Todos los procesadores
)

# 4. (Opcional) Aplicar usando el método de Fran con Zernikes
z_coefs = np.zeros(18)
z_coefs[3] = 0.5  # Ejemplo ficticio

cubo_procesado_fran = aplicar_deconvolucion_3d(
    imagen_3d = cubo_3d.copy(),
    psf = None, 
    metodo = 'w_fran',
    zernikes = z_coefs,
    workers = 4  # Usar 4 núcleos en el Pool
)
```

## Resumen de Mejores Prácticas

1. **Valores Negativos:** Gracias a la lógica de `deconvolucionMulti`, puedes pasar libremente matrices de Stokes V, Q o U (que contienen valores por debajo de cero). El algoritmo separará lo positivo de lo negativo, las deconvolucionará evitando errores y las restará al final.
2. **Workers:** Deja siempre `workers=-1` a menos que quieras restringir la carga de CPU para otras tareas.
3. **Parámetro K:** Ajusta bien el balance de `k` (por defecto `1e-3`); valores más altos reducen el ruido pero emborronan un poco la imagen reconstruida, valores muy pequeños realzan el contraste pero disparan el granulado de ruido en el fondo.
4. **Conservar memoria:** `aplicar_deconvolucion_3d` sobreescribe el array que se le pasa si no realizas una copia previa (debido a cómo funciona internamente la manipulación de arrays o el paso por referencia). Pasa `imagen_3d.copy()` si quieres mantener la versión original borrosa en memoria RAM.
