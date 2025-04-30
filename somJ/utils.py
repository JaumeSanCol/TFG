import numpy as np
import matplotlib.pyplot as plt

def check_input_data(data):
    if data is None:
        raise ValueError("Los datos de entrada no pueden ser None.")
    if not isinstance(data, np.ndarray):
        raise TypeError("Los datos deben ser un numpy array.")
    if data.ndim != 2:
        raise ValueError("Los datos deben ser un array bidimensional.")
    if data.shape[1] < 2:
        raise ValueError("Los datos deben tener al menos dos características para PCA.")

def validate_batch_size(batch_size):
    if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
        raise ValueError("El tamaño del batch debe ser un entero positivo.")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot_map(som_map, title="Self-Organizing Map", save_path=None):
    m, n, dim = som_map.shape

    if dim == 3:
        # Visualizar como imagen RGB
        img = np.clip(som_map, 0, 1)
        plt.figure(figsize=(8, 6))
        plt.imshow(img, interpolation='nearest')
        plt.title(title)
        plt.axis('off')
    else:
        # Reducir dimensiones con PCA para visualizar en 2D
        som_2d = som_map.reshape(-1, dim)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(som_2d)
        reduced = reduced.reshape(m, n, 2)

        plt.figure(figsize=(8, 6))
        plt.quiver(
            np.arange(n), np.arange(m)[:, None],
            reduced[:, :, 0], reduced[:, :, 1],
            angles='xy', scale_units='xy', scale=1
        )
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.axis('equal')
        plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()
