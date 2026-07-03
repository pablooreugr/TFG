import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ajustar path para importar desde src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import magnetismo
import deconvolucion as deco

# 1. Ground Truth Generation
nx, ny = 64, 64
B_real = np.zeros((ny, nx))
y, x = np.ogrid[-ny//2:ny//2, -nx//2:nx//2]

# Zonas positivas y negativas
r1 = np.sqrt((x-10)**2 + (y-10)**2)
r2 = np.sqrt((x+15)**2 + (y+15)**2)
B_real += 1000 * np.exp(-r1**2 / (8**2))
B_real -= 800 * np.exp(-r2**2 / (6**2))

# 2. Profile Synthesis
lambdas = np.linspace(6300, 6303, 60)
lambda0 = 6301.5
# Perfil de absorción Gaussiano
I0 = 1.0 - 0.5 * np.exp(-((lambdas - lambda0)/0.2)**2)

# Añadimos dimensiones para broadcasting (n_lambda, ny, nx)
I_3D = np.tile(I0[:, np.newaxis, np.newaxis], (1, ny, nx))

# Derivada de I respecto a lambda
dI0_dl = np.gradient(I0, lambdas)
dI_3D = np.tile(dI0_dl[:, np.newaxis, np.newaxis], (1, ny, nx))

lambdas3D = lambdas[:, np.newaxis, np.newaxis]
constanteFormula = 4.67e-13
g = 3

# V = - C * g * lambda^2 * B_paralelo * dI/dlambda
V_3D = -constanteFormula * g * (lambdas3D**2) * B_real[np.newaxis, :, :] * dI_3D

# 3. Degradation
psf = deco.generar_psf_airy(15, 3.0)

# Convolución
print("Convolucionando...")
I_conv = deco.convolucion3D(I_3D, psf, usar_padding=True)
V_conv = deco.convolucion3D(V_3D, psf, usar_padding=True)

# Añadir ruido
noise_std_I = 1e-3
# Stokes V suele tener un ruido menor absoluto pero mayor relativo. Ponemos algo realista
noise_std_V = 1e-4 
np.random.seed(42)
I_deg = I_conv + np.random.normal(0, noise_std_I, size=I_conv.shape)
V_deg = V_conv + np.random.normal(0, noise_std_V, size=V_conv.shape)

# 4. Reconstruction
print("Reconstruyendo con RL...")
B_rec, _ = magnetismo.calcularCampoMagneticoRL(I_deg, V_deg, lambdas, psf, g=g, pasos=30)

# 5. Cálculo de métricas
rmse = np.sqrt(np.mean((B_real - B_rec)**2))
mae = np.mean(np.abs(B_real - B_rec))

max_real = B_real.max()
min_real = B_real.min()
max_rec = B_rec.max()
min_rec = B_rec.min()

print(f"\n--- RESULTADOS METRICAS ---")
print(f"RMSE: {rmse:.4f} G")
print(f"MAE: {mae:.4f} G")
print(f"Amplitud Max Real: {max_real:.4f} G | Min Real: {min_real:.4f} G")
print(f"Amplitud Max Rec:  {max_rec:.4f} G | Min Rec:  {min_rec:.4f} G")
print("---------------------------\n")

# 6. Reconstrucción interna para plot 1D
v_min = V_deg.min()
shift = -v_min if v_min < 0 else 0.0
V_shifted = V_deg + shift
V_shifted_deco = deco.deconvolucion3D(V_shifted, psf, metodo='rl', pasos=30)
V_deco = V_shifted_deco - shift

# Elegir un píxel con campo fuerte positivo (centro del blob r1)
py, px = ny//2 + 10, nx//2 + 10

# 7. Visualización
plt.figure(figsize=(15, 10))

# Panel 1: Mapas
plt.subplot(2, 3, 1)
plt.title("B Real")
plt.imshow(B_real, cmap='coolwarm', origin='lower', vmin=-1000, vmax=1000)
plt.colorbar(label="Gauss")

plt.subplot(2, 3, 2)
plt.title("B Reconstruido (RL + Shift)")
plt.imshow(B_rec, cmap='coolwarm', origin='lower', vmin=-1000, vmax=1000)
plt.colorbar(label="Gauss")

plt.subplot(2, 3, 3)
plt.title("Residuos |Real - Rec|")
plt.imshow(np.abs(B_real - B_rec), cmap='viridis', origin='lower')
plt.colorbar(label="Gauss")

# Panel 2: Perfil 1D
plt.subplot(2, 1, 2)
plt.title(f"Perfil Stokes V en píxel ({py}, {px}) (Campo Positivo)")
plt.plot(lambdas, V_3D[:, py, px], label="Original (Limpio)", linestyle='--', linewidth=2)
plt.plot(lambdas, V_deg[:, py, px], label="Degradado (Blur + Ruido)", alpha=0.7)
plt.plot(lambdas, V_deco[:, py, px], label="Reconstruido internamente (RL + Shift)", linewidth=2)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlabel(r"$\lambda$ ($\AA$)")
plt.ylabel("Stokes V")

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "resultado_experimento.png")
plt.savefig(output_path)
print(f"Gráfica guardada en {output_path}")
