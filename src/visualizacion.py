import matplotlib
matplotlib.use('QtAgg') # Obliga a usar la interfaz de Qt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def mostrar_array(arr, title=None, cmap='viridis', vmin=None, vmax=None,
				   figsize=(6, 4), save_path=None, colorbar=True,
				   interpolation='nearest', origin='upper', log_scale=False):
	"""Muestra un array 2D (o imagen RGB) usando `imshow`.

	Parámetros:
	- arr: array-like 2D (grayscale) o 3D (RGB/RGBA).
	- title: título opcional.
	- cmap: mapa de colores para arrays 2D.
	- vmin, vmax: límites de color.
	- figsize: tamaño de la figura.
	- save_path: ruta para guardar la imagen (opcional).
	- colorbar: mostrar barra de color para arrays 2D.
	- interpolation: método de interpolación para `imshow`.
	- origin: origen de la imagen ('upper' o 'lower').
	- log_scale: si es True, se utiliza una escala logarítmica para los colores.
	"""
	arr = np.asarray(arr)
	if arr.size == 0:
		raise ValueError('El array está vacío')

	# Manejar arrays con formato canales-primero (C, H, W)
	if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
		arr = np.transpose(arr, (1, 2, 0))

	# Si es (H, W, 1) convertir a (H, W)
	if arr.ndim == 3 and arr.shape[2] == 1:
		arr = arr[:, :, 0]

	if arr.ndim not in (2, 3):
		raise ValueError('Array con dimensiones no soportadas: ' + str(arr.shape))

	fig, ax = plt.subplots(figsize=figsize)

	# Si es imagen RGB/RGBA, no usar cmap
	if arr.ndim == 3 and arr.shape[2] in (3, 4):
		im = ax.imshow(arr, interpolation=interpolation, origin=origin)
	else:
		norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
		# Cuando se usa norm, no se pueden pasar vmin y vmax a imshow directamente
		vmin_arg = None if log_scale else vmin
		vmax_arg = None if log_scale else vmax
		im = ax.imshow(arr, cmap=cmap, vmin=vmin_arg, vmax=vmax_arg, norm=norm,
					   interpolation=interpolation, origin=origin)

	if title:
		ax.set_title(title)

	ax.set_axis_off()

	if colorbar and (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1)):
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, bbox_inches='tight')
	plt.show()


def mostrar_dos_arrays(arr1, arr2, title1=None, title2=None, cmap='viridis', 
                       vmin1=None, vmax1=None, vmin2=None, vmax2=None,
                       figsize=(12, 4), save_path=None, colorbar=True,
                       interpolation='nearest', origin='upper', log_scale=False,
                       share_colors=False):
	"""Muestra dos arrays 2D (o imágenes RGB) uno al lado del otro usando `imshow`.

	Parámetros:
	- arr1, arr2: arrays-like 2D (grayscale) o 3D (RGB/RGBA).
	- title1, title2: títulos opcionales para cada subplot.
	- cmap: mapa de colores para arrays 2D (aplicado a ambos, o tupla/lista para cada uno).
	- vmin1, vmax1, vmin2, vmax2: límites de color para cada array.
	- figsize: tamaño de la figura (por defecto más ancha para dos gráficos).
	- save_path: ruta para guardar la imagen (opcional).
	- colorbar: mostrar barra de color para arrays 2D.
	- interpolation: método de interpolación para `imshow`.
	- origin: origen de la imagen ('upper' o 'lower').
	- log_scale: si es True, usa escala logarítmica (puede ser un bool o una tupla de dos bools).
	- share_colors: si es True, ambas imágenes compartirán la misma escala de colores (mismos vmin y vmax).
	"""
	def procesar_array(arr):
		arr = np.asarray(arr)
		if arr.size == 0:
			raise ValueError('El array está vacío')
		if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
			arr = np.transpose(arr, (1, 2, 0))
		if arr.ndim == 3 and arr.shape[2] == 1:
			arr = arr[:, :, 0]
		if arr.ndim not in (2, 3):
			raise ValueError('Array con dimensiones no soportadas: ' + str(arr.shape))
		return arr

	arr1 = procesar_array(arr1)
	arr2 = procesar_array(arr2)

	if share_colors:
		global_vmin = min(
			vmin1 if vmin1 is not None else np.nanmin(arr1),
			vmin2 if vmin2 is not None else np.nanmin(arr2)
		)
		global_vmax = max(
			vmax1 if vmax1 is not None else np.nanmax(arr1),
			vmax2 if vmax2 is not None else np.nanmax(arr2)
		)
		vmin1 = vmin2 = global_vmin
		vmax1 = vmax2 = global_vmax

	if isinstance(cmap, (list, tuple)) and len(cmap) == 2:
		cmap1, cmap2 = cmap
	else:
		cmap1 = cmap2 = cmap

	if isinstance(log_scale, (list, tuple)) and len(log_scale) == 2:
		log_scale1, log_scale2 = log_scale
	else:
		log_scale1 = log_scale2 = log_scale

	fig, axes = plt.subplots(1, 2, figsize=figsize)

	for ax, arr, title, cmp, vmin, vmax, log_s in zip(
		axes, 
		[arr1, arr2], 
		[title1, title2], 
		[cmap1, cmap2], 
		[vmin1, vmin2], 
		[vmax1, vmax2],
		[log_scale1, log_scale2]
	):
		if arr.ndim == 3 and arr.shape[2] in (3, 4):
			im = ax.imshow(arr, interpolation=interpolation, origin=origin)
		else:
			norm = LogNorm(vmin=vmin, vmax=vmax) if log_s else None
			vmin_arg = None if log_s else vmin
			vmax_arg = None if log_s else vmax
			im = ax.imshow(arr, cmap=cmp, vmin=vmin_arg, vmax=vmax_arg, norm=norm,
						   interpolation=interpolation, origin=origin)
		
		if title:
			ax.set_title(title)
			
		ax.set_axis_off()
		
		if colorbar and (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1)):
			fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, bbox_inches='tight')
	plt.show()


def mostrar_n_arrays(arrays, titles=None, cmap='viridis', 
                       vmins=None, vmaxs=None,
                       figsize=None, save_path=None, colorbar=True,
                       interpolation='nearest', origin='upper', log_scale=False,
                       share_colors=False, cols=None):
	"""Muestra n arrays 2D (o imágenes RGB) en una cuadrícula usando `imshow`.

	Parámetros:
	- arrays: lista de arrays-like 2D (grayscale) o 3D (RGB/RGBA).
	- titles: lista de títulos opcionales para cada subplot.
	- cmap: mapa de colores para arrays 2D (aplicado a todos, o lista para cada uno).
	- vmins, vmaxs: listas de límites de color para cada array. Si es None, se autoajusta.
	- figsize: tamaño de la figura. Si es None, se calcula automáticamente.
	- save_path: ruta para guardar la imagen (opcional).
	- colorbar: mostrar barra de color para arrays 2D.
	- interpolation: método de interpolación para `imshow`.
	- origin: origen de la imagen ('upper' o 'lower').
	- log_scale: si es True, usa escala logarítmica (puede ser un bool o lista de bools).
	- share_colors: si es True, todas las imágenes compartirán la misma escala de colores (mismos vmin y vmax).
	- cols: número de columnas en la cuadrícula. Si es None, se ponen todas en una fila.
	"""
	def procesar_array(arr):
		arr = np.asarray(arr)
		if arr.size == 0:
			raise ValueError('Un array está vacío')
		if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
			arr = np.transpose(arr, (1, 2, 0))
		if arr.ndim == 3 and arr.shape[2] == 1:
			arr = arr[:, :, 0]
		if arr.ndim not in (2, 3):
			raise ValueError('Array con dimensiones no soportadas: ' + str(arr.shape))
		return arr

	n = len(arrays)
	if n == 0:
		return

	arrays = [procesar_array(arr) for arr in arrays]

	if titles is None:
		titles = [None] * n
	elif len(titles) < n:
		titles = list(titles) + [None] * (n - len(titles))

	if share_colors:
		global_vmin = min(
			vmin if vmin is not None else np.nanmin(arr)
			for arr, vmin in zip(arrays, vmins if vmins else [None]*n)
		)
		global_vmax = max(
			vmax if vmax is not None else np.nanmax(arr)
			for arr, vmax in zip(arrays, vmaxs if vmaxs else [None]*n)
		)
		vmins = [global_vmin] * n
		vmaxs = [global_vmax] * n
	else:
		if vmins is None:
			vmins = [None] * n
		if vmaxs is None:
			vmaxs = [None] * n

	if isinstance(cmap, (list, tuple)):
		cmaps = list(cmap) + [cmap[-1]] * max(0, n - len(cmap))
	else:
		cmaps = [cmap] * n

	if isinstance(log_scale, (list, tuple)):
		log_scales = list(log_scale) + [log_scale[-1]] * max(0, n - len(log_scale))
	else:
		log_scales = [log_scale] * n

	if cols is None:
		cols = n
	rows = int(np.ceil(n / cols))

	if figsize is None:
		figsize = (5 * cols, 4 * rows)

	fig, axes = plt.subplots(rows, cols, figsize=figsize)
	
	if n == 1 and rows == 1 and cols == 1:
		axes = [axes]
	elif isinstance(axes, np.ndarray):
		axes = axes.flatten()
	else:
		axes = [axes]

	for i in range(n):
		ax = axes[i]
		arr = arrays[i]
		title = titles[i]
		cmp = cmaps[i]
		vmin = vmins[i]
		vmax = vmaxs[i]
		log_s = log_scales[i]

		if arr.ndim == 3 and arr.shape[2] in (3, 4):
			im = ax.imshow(arr, interpolation=interpolation, origin=origin)
		else:
			norm = LogNorm(vmin=vmin, vmax=vmax) if log_s else None
			vmin_arg = None if log_s else vmin
			vmax_arg = None if log_s else vmax
			im = ax.imshow(arr, cmap=cmp, vmin=vmin_arg, vmax=vmax_arg, norm=norm,
						   interpolation=interpolation, origin=origin)
		
		if title:
			ax.set_title(title)
			
		ax.set_axis_off()
		
		if colorbar and (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1)):
			fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

	for i in range(n, len(axes)):
		axes[i].set_axis_off()

	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, bbox_inches='tight')
	plt.show()
class GraficaConvergencia:
    def __init__(self, titulo="Convergencia del Algoritmo de Noor", auto_close=False):
        self.auto_close = auto_close
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_title(titulo)
        self.ax.set_xlabel("Iteración")
        self.ax.set_ylabel("Norma del Residuo")
        self.ax.set_yscale("log")
        self.line, = self.ax.plot([], [], 'b.-', label="Residuo")
        self.ax.legend()
        self.ax.grid(True, which="both", ls="--", alpha=0.5)
        self.iteraciones = []
        self.normas = []
        self.iter_actual = 0
        
    def actualizar(self, norma):
        self.iter_actual += 1
        self.iteraciones.append(self.iter_actual)
        self.normas.append(norma)
        
        self.line.set_data(self.iteraciones, self.normas)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finalizar(self):
        plt.ioff()
        if self.auto_close:
            plt.close(self.fig)
        else:
            plt.show()

if __name__ == "__main__":
    arr = np.random.rand(200,200)        # o (H,W,3) o (3,H,W)
    mostrar_array(arr, title='Mi imagen', cmap='gray')