import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from functions import *
from sklearn.decomposition import PCA
from somJ.som import SoM
import somJ.config as config
from minisom import MiniSom

# ------------------------------------------------------------------------------------------------------------------------------
#   Experimento para observar el tiempo de jecución nuestra implementación de SOM respecto de MiniSOM en relación al
#   número de dimensiones en el dataset
# ------------------------------------------------------------------------------------------------------------------------------
#
#   Los resultados son almacenados en una grafica en la carpeta g_time_comp
#
# ------------------------------------------------------------------------------------------------------------------------------
# Después de evaluar la accuracy respecto del numero de epochs, podemos observar que minisom no requiere de tantas muestras para
# obtener información por lo que con una epoch es suficiente. Poner "ajustado" verdadero, significa que para el experimento se
# usara una sola epoch de Minisom frente a las 50 de SOM. Si es falso, se usarán 50 epochs en cada algoritmo.

AJUSTADO=False

# Parámetros
n_samples = 4000
dimensions = [25, 100, 225, 400, 625, 784] # 5x5 10x10 15x15 20x20 25x25 28x28
epochs = config.EPOCHS
som_shape = int(np.sqrt(config.TOTAL_NODES))

# Cargar MNIST y normalizar
X_all,_=load_dataset("MNIST")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all)
X_sample = X_scaled[:n_samples]

# Almacenar tiempos
somJ_times = []
minisom_times = []

# Evaluar para cada dimensión
for d in dimensions:
    print(f"Dimensión: {d}")
    
    # Reducir dimensiones si es necesario
    if d < 784:
        pca = PCA(n_components=d)
        X_scaled = pca.fit_transform(X_sample)
    else:
        X_scaled = X_sample

    X_scaled = np.array(X_scaled)  
    # SoM (somJ)
    som = SoM(
        method=config.INIT_METHOD,
        data=X_scaled,
        total_nodes=config.TOTAL_NODES
    )
    start_time = time.time()
    som.train(
        train_data=X_scaled,
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
    minisom = MiniSom(som_shape, som_shape, d, sigma=config.RADIUS_SQ, learning_rate=config.LEARNING_RATE)
    start_time = time.time()
    if AJUSTADO:
        minisom.train_batch(X_scaled, len(X_scaled))
    else:
        for i in range(0,epochs):
            minisom.train_batch(X_scaled, len(X_scaled))
    minisom_times.append(time.time() - start_time)

# Graficar resultados en una sola figura
plt.figure(figsize=(8, 6))
plt.plot(dimensions, somJ_times, label='SoM (somJ)', marker='o')
plt.plot(dimensions, minisom_times, label='MiniSom', marker='o')
plt.xlabel('Dimensión de entrada')
plt.ylabel('Tiempo de entrenamiento (s)')
plt.title(f'Tiempo vs # de dimensiones de entrada')
plt.legend()
plt.grid(True)
plt.tight_layout()
if AJUSTADO:plt.savefig(f'experimentos/g_time_comp/tiempos_vs_dimensiones_ajustado.png', dpi=300)
else:plt.savefig(f'experimentos/g_time_comp/tiempos_vs_dimensiones.png', dpi=300)
plt.show()
