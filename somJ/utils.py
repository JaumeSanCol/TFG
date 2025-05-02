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

