import numpy as np

def opPt_3D(imagen_extendida, psf_shape):
    pad_size = psf_shape // 2
    y = imagen_extendida.copy()
    p = pad_size

    # Eje Y (axis 1)
    y[:, p+1 : 2*p+1, :] += y[:, 0:p, :][:, ::-1, :]
    y[:, -2*p-1 : -p-1, :] += y[:, -p:, :][:, ::-1, :]

    # Eje X (axis 2)
    y[:, :, p+1 : 2*p+1] += y[:, :, 0:p][:, :, ::-1]
    y[:, :, -2*p-1 : -p-1] += y[:, :, -p:][:, :, ::-1]

    return y[:, p:-p, p:-p]

x = np.random.randn(3, 10, 10)
p = 2
x_pad = np.pad(x, ((0,0), (p,p), (p,p)), mode='reflect')

# Test adjoint property: <Px, y> = <x, P^T y>
y = np.random.randn(*x_pad.shape)
res1 = np.sum(x_pad * y)
res2 = np.sum(x * opPt_3D(y, 4))
print(f"Adjoint error: {abs(res1 - res2)}")
