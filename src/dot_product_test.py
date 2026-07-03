"""
dot_product_test.py
===================
Validación numérica del operador adjunto (Dot Product Test).

Comprueba si el operador J^T implementado en algoritmoDeNoor es la transpuesta
exacta del operador directo J, verificando la identidad:

    <J(x), y>  ==  <x, J^T(y)>

Un error relativo < 1e-10 indica que los operadores son adjuntos exactos.
Un error mayor señala una asimetría (normalmente en el padding o en K_scaled).

Uso:
    python dot_product_test.py

Las dimensiones pueden ajustarse con las constantes al inicio del script.
"""

import numpy as np
import deconvolucion as deco

# ---------------------------------------------------------------------------
# Parámetros de prueba — ajustar a las dimensiones reales del problema
# ---------------------------------------------------------------------------
N_LAMBDA = 6      # número de longitudes de onda
NY       = 32     # filas espaciales
NX       = 32     # columnas espaciales
PSF_SIZE = 11     # tamaño de la PSF (debe ser impar)
PSF_RADIUS = 3.0  # radio en píxeles para la PSF de Airy sintética
SEED     = 42     # semilla para reproducibilidad
VERBOSE  = True   # mostrar tabla de resultados

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Construir una PSF sintética y un K_scaled sintético
# ---------------------------------------------------------------------------
psf = deco.generar_psf_airy(PSF_SIZE, PSF_RADIUS)

# K_scaled sintético: simula -(cte_formula * g * dI/dlambda * lambda^2) / k_max
# Aquí lo generamos directamente como tensor aleatorio normalizado para el test.
K_scaled = np.random.randn(N_LAMBDA, NY, NX)
k_norm = np.abs(K_scaled).max()
if k_norm > 0:
    K_scaled /= k_norm

# ---------------------------------------------------------------------------
# 2. Definir los mismos operadores que usa algoritmoDeNoor
#    (con padding de CEROS — Tarea 3)
# ---------------------------------------------------------------------------

def opK(campoB):
    """Multiplica por K_scaled (operación lambda-por-lambda)."""
    return K_scaled * campoB[np.newaxis, :, :]

def opKt(campoV):
    """Transpuesta de opK: suma a lo largo del eje lambda."""
    return np.sum(K_scaled * campoV, axis=0)

def opP(campoV):
    """Zero-padding 3D simétrico."""
    pad_size = psf.shape[0] // 2
    res = np.pad(
        campoV,
        ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
        mode='constant', constant_values=0
    )
    return res, pad_size

def opPt(imagen_extendida):
    """
    P^T: Transpuesta del zero-padding.
    La transpuesta exacta del zero-padding es un simple crop.
    """
    pad_size = psf.shape[0] // 2
    p = pad_size
    return imagen_extendida[:, p:-p, p:-p]

def opC(campoV):
    """Convolución 3D (sin padding interno — el padding ya lo aplica opP)."""
    return deco.convolucion3D(campoV, psf, usar_padding=False)

def opCt(campoV):
    """Convolución con PSF girada (correlación)."""
    psf_girada = deco.girarPSF(psf)
    return deco.convolucion3D(campoV, psf_girada, usar_padding=False)

def opR(campoV, pad_size):
    """Recorta los bordes tras la convolución."""
    return campoV[:, pad_size:-pad_size, pad_size:-pad_size]

def opRt(imagen):
    """Transpuesta del recorte: zero-padding."""
    pad_size = psf.shape[0] // 2
    return np.pad(
        imagen,
        pad_width=((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
        mode='constant', constant_values=0
    )

# ---------------------------------------------------------------------------
# 3. Operador directo J y su adjunto J^T
# ---------------------------------------------------------------------------

def J(dB_2D):
    """J: R^(NY×NX) → R^(N_LAMBDA×NY×NX)"""
    compV = opK(dB_2D)
    padding, pad_size = opP(compV)
    convolucion = opC(padding)
    recorte = opR(convolucion, pad_size)
    return recorte

def JT(res_3D):
    """J^T: R^(N_LAMBDA×NY×NX) → R^(NY×NX)"""
    recorte = opRt(res_3D)
    convolucion = opCt(recorte)
    paddinTras = opPt(convolucion)
    res = opKt(paddinTras)
    return res

# ---------------------------------------------------------------------------
# 4. Dot Product Test: <J(x), y> vs <x, J^T(y)>
# ---------------------------------------------------------------------------

def dot_product_test(n_trials: int = 5) -> bool:
    """
    Ejecuta n_trials ensayos con vectores aleatorios y verifica la identidad.

    Retorna True si todos los ensayos pasan (error relativo < 1e-10).
    """
    print("=" * 60)
    print("  DOT PRODUCT TEST  —  J adjunto de J^T")
    print("=" * 60)
    print(f"  Dimensiones: x en R^({NY}x{NX}), y en R^({N_LAMBDA}x{NY}x{NX})")
    print(f"  PSF: {PSF_SIZE}x{PSF_SIZE}, radio Airy = {PSF_RADIUS} px")
    print(f"  Padding: mode='constant' (ceros)")
    print("-" * 60)

    all_passed = True
    threshold  = 1e-10

    for trial in range(1, n_trials + 1):
        x = np.random.randn(NY, NX)           # vector de entrada de J
        y = np.random.randn(N_LAMBDA, NY, NX) # vector de entrada de J^T

        lhs = np.vdot(J(x), y)    # <J(x), y>
        rhs = np.vdot(x, JT(y))   # <x, J^T(y)>

        denom     = max(abs(lhs), abs(rhs), 1e-300)
        rel_error = abs(lhs - rhs) / denom
        passed    = rel_error < threshold

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        if VERBOSE:
            print(f"  Trial {trial:2d}: LHS={lhs:+.8e}  RHS={rhs:+.8e}"
                  f"  err_rel={rel_error:.3e}  [{status}]")

    print("-" * 60)
    verdict = "TODOS LOS ENSAYOS PASARON" if all_passed else "FALLO EN ALGUN ENSAYO"
    print(f"  Veredicto final: [{verdict}]")
    print("=" * 60)

    if not all_passed:
        print("\n  AVISO: Si el test falla, revisa:")
        print("      - opPt: debe ser un simple crop (para zero-padding)")
        print("      - opCt: debe usar la PSF girada 180")
        print("      - opRt: debe usar zero-padding (no reflect)")
        print("      - opKt: debe sumar sobre el eje lambda con K_scaled")

    return all_passed


# ---------------------------------------------------------------------------
# 5. Test adicional: verificar sub-operadores individualmente
# ---------------------------------------------------------------------------

def test_sub_operators():
    """Verifica cada sub-operador por separado para localizar fallos."""
    print("\n" + "=" * 60)
    print("  TEST DE SUB-OPERADORES INDIVIDUALES")
    print("=" * 60)

    pad_size = psf.shape[0] // 2
    threshold = 1e-10

    # --- P y P^T ---
    a3d = np.random.randn(N_LAMBDA, NY, NX)
    b_padded, _ = opP(a3d)
    b3d = np.random.randn(*b_padded.shape)

    lhs_P = np.vdot(b_padded, b3d)         # <P(a), b>
    rhs_P = np.vdot(a3d, opPt(b3d))        # <a, P^T(b)>
    err_P = abs(lhs_P - rhs_P) / max(abs(lhs_P), abs(rhs_P), 1e-300)
    print(f"  P / P^T   err_rel = {err_P:.3e}  [{'PASS' if err_P < threshold else 'FAIL'}]")

    # --- C y C^T ---
    ny_pad, nx_pad = NY + 2 * pad_size, NX + 2 * pad_size
    c1 = np.random.randn(N_LAMBDA, ny_pad, nx_pad)
    c2 = np.random.randn(N_LAMBDA, ny_pad, nx_pad)
    lhs_C = np.vdot(opC(c1), c2)
    rhs_C = np.vdot(c1, opCt(c2))
    err_C = abs(lhs_C - rhs_C) / max(abs(lhs_C), abs(rhs_C), 1e-300)
    print(f"  C / C^T   err_rel = {err_C:.3e}  [{'PASS' if err_C < threshold else 'FAIL'}]")

    # --- R y R^T ---
    r_ext = np.random.randn(N_LAMBDA, ny_pad, nx_pad)
    r_crop = opR(r_ext, pad_size)
    r2 = np.random.randn(*r_crop.shape)
    lhs_R = np.vdot(r_crop, r2)
    rhs_R = np.vdot(r_ext, opRt(r2))
    err_R = abs(lhs_R - rhs_R) / max(abs(lhs_R), abs(rhs_R), 1e-300)
    print(f"  R / R^T   err_rel = {err_R:.3e}  [{'PASS' if err_R < threshold else 'FAIL'}]")

    # --- K y K^T ---
    k_in = np.random.randn(NY, NX)
    k_out = opK(k_in)
    k2 = np.random.randn(*k_out.shape)
    lhs_K = np.vdot(k_out, k2)
    rhs_K = np.vdot(k_in, opKt(k2))
    err_K = abs(lhs_K - rhs_K) / max(abs(lhs_K), abs(rhs_K), 1e-300)
    print(f"  K / K^T   err_rel = {err_K:.3e}  [{'PASS' if err_K < threshold else 'FAIL'}]")

    print("=" * 60)


# ---------------------------------------------------------------------------
# 6. Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_ok = dot_product_test(n_trials=10)
    test_sub_operators()
