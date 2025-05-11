import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from somJ.som import SoM
import somJ.config as config
from minisom import MiniSom

# ------------------------------------------------------------------------------------------------------------------------------
#   Experimento para observar el tiempo de jecución nuestra implementación de SOM respecto de MiniSOM en relación al
#   número de muestras del dataset
# ------------------------------------------------------------------------------------------------------------------------------
#
#   Los resultados son almacenados en una grafica en la carpeta g_time_comp
#
# ------------------------------------------------------------------------------------------------------------------------------


sample_sizes = [1000, 5000, 10000, 20000] # Máximo 60 000
fixed_dimension = 784 # Máximo de MNIST
epochs = config.EPOCHS
som_shape = int(np.sqrt(config.TOTAL_NODES))

# Cargar MNIST y normalizar
mnist = fetch_openml('mnist_784', version=1)
X_all = mnist.data.astype('float32') / 255.0 

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
    minisom.train_batch(X_sample, epochs*len(X_sample))
    minisom_times.append(time.time() - start_time)

plt.figure(figsize=(8, 6))
plt.plot(sample_sizes, somJ_times, label='SoM (somJ)', marker='o')
plt.plot(sample_sizes, minisom_times, label='MiniSom', marker='o')
plt.xlabel('Número de muestras')
plt.ylabel('Tiempo de entrenamiento (s)')
plt.title(f'Tiempo vs muestras (dimensión fija: {fixed_dimension})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'experimentos/g_time_comp/tiempos_vs_muestras_{fixed_dimension}_batchmap_fiar.png', dpi=300)
plt.show()
