import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from somJ.som import SoM
from functions import *
import somJ.config as config
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------------------------------------------------------------------------
#   Experimento para observar el tiempo de jecución nuestra implementación de SOM respecto de MiniSOM en relación al
#   número de muestras del dataset
# ------------------------------------------------------------------------------------------------------------------------------
#
#   Los resultados son almacenados en una grafica en la carpeta g_time_comp
#
# ------------------------------------------------------------------------------------------------------------------------------

# Después de evaluar la accuracy respecto del numero de epochs, podemos observar que minisom no requiere de tantas muestras para
# obtener información por lo que con una epoch es suficiente. Poner "ajustado" verdadero, significa que para el experimento se
# usara una sola epoch de Minisom frente a las 50 de SOM. Si es falso, se usarán 50 epochs en cada algoritmo.

AJUSTADO=True

sample_sizes = [1000, 5000, 10000, 20000,40000] # Máximo 60 000
fixed_dimension = 784 # Máximo de MNIST
epochs = config.EPOCHS
som_shape = int(np.sqrt(config.TOTAL_NODES))

# Cargar MNIST y normalizar
X_all,_=load_dataset("MNIST")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all)

# Almacenar tiempos
somJ_times = []
minisom_times = []

for n_samples in sample_sizes:
    print(f"Muestras: {n_samples}")
    X_sample = np.array(X_all[:n_samples])

    som = SoM(
        method=config.INIT_METHOD,
        data=X_sample,
        total_nodes=config.TOTAL_NODES
    )
    start_time = time.time()
    som.train(
        train_data=X_sample,
        learn_rate=config.LEARNING_RATE,
        radius_sq=config.RADIUS_SQ,
        lr_decay=config.LR_DECAY,
        radius_decay=config.RADIUS_DECAY,
        epochs=epochs,
        update=config.UPDATE_METHOD,
        batch_size=config.BATCH_SIZE,
        step=config.STEP,
        save=config.SAVE_HISTORY,
        prog_bar=False
    )
    somJ_times.append(time.time() - start_time)

    # MiniSom
    minisom = MiniSom(som_shape, som_shape, fixed_dimension, sigma=config.RADIUS_SQ, learning_rate=config.LEARNING_RATE)
    start_time = time.time()
    if AJUSTADO:
        minisom.train_batch(X_sample, len(X_sample))
    else:
        for i in range(0,epochs):
            minisom.train_batch(X_sample, len(X_sample))
    minisom_times.append(time.time() - start_time)


plt.figure(figsize=(8, 6))
plt.plot(sample_sizes, somJ_times, label='SoM (somJ)', marker='o')
plt.plot(sample_sizes, minisom_times, label='MiniSom', marker='o')
plt.xlabel('Número de muestras')
plt.ylabel('Tiempo de entrenamiento (s)')
plt.title(f'Tiempo vs muestras')
plt.legend()
plt.grid(True)
plt.tight_layout()
if AJUSTADO:plt.savefig(f'experimentos/g_time_comp/tiempos_vs_muestras_batchmap_ajustado.png', dpi=300)
else:plt.savefig(f'experimentos/g_time_comp/tiempos_vs_muestras_bathcmap.png', dpi=300)
plt.show()