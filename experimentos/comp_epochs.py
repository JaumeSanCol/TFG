import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from somJ.som import SoM
import somJ.config as config
from minisom import MiniSom
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# ------------------------------------------------------------------------------------------------------------------------------
#   Experimento para observar la relación de accuracy respecto del numero de epochs en el entrenamiento de nuesrto SOM respecto 
#   de MiniSOM 
# ------------------------------------------------------------------------------------------------------------------------------
#
#   Los resultados son almacenados en una grafica en la carpeta g_time_comp
#
# ------------------------------------------------------------------------------------------------------------------------------
def evaluate_classification_embedded(X_train_emb, y_train, X_test_emb, y_test):
    # Construir mapa de etiquetas por neurona
    label_map = {}
    for coord, label in zip(X_train_emb, y_train):
        coord = tuple(coord)
        label_map.setdefault(coord, []).append(label)
    neuron_labels = {coord: max(labels, key=labels.count)
                     for coord, labels in label_map.items()}

    # Predecir para test
    y_pred = []
    for coord in X_test_emb:
        coord = tuple(coord)
        if coord in neuron_labels:
            y_pred.append(neuron_labels[coord])
        else:
            # buscar neurona etiquetada más cercana
            dists = [ (coord[0]-c[0])**2 + (coord[1]-c[1])**2
                      for c in neuron_labels.keys() ]
            nearest = list(neuron_labels.keys())[np.argmin(dists)]
            y_pred.append(neuron_labels[nearest])
    y_pred = np.array(y_pred)

    # Métricas
    acc  = accuracy_score(y_test, y_pred)
    return acc

num_epochs = [1,5,10,25,50,75,100]
fixed_dimension = 784 # Máximo de MNIST
som_shape = int(np.sqrt(config.TOTAL_NODES))

# Cargar MNIST y normalizar
mnist = fetch_openml('mnist_784', version=1, cache=True)
X, y = mnist.data, mnist.target.astype(int)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Almacenar tiempos
somJ_times = []
minisom_times = []

for epochs in num_epochs:
    print(f"Epochs: {epochs}")

    som = SoM(
        method=config.INIT_METHOD,
        data=X_train,
        total_nodes=config.TOTAL_NODES
    )
    som.train(
        train_data=X_train,
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
    X_som_train = np.array([som.find_winner(x) for x in X_train])
    X_som_test  = np.array([som.find_winner(x) for x in X_test])
    somJ_times.append(evaluate_classification_embedded(X_som_train, y_train, X_som_test, y_test))


    # MiniSom
    minisom = MiniSom(som_shape, som_shape, fixed_dimension, sigma=config.RADIUS_SQ, learning_rate=config.LEARNING_RATE)
    for i in range(0,epochs):
        minisom.train_batch(X_train, len(X_train))
    X_min_train = np.array([minisom.winner(x) for x in X_train])
    X_min_test  = np.array([minisom.winner(x) for x in X_test])
    minisom_times.append(evaluate_classification_embedded(X_min_train, y_train, X_min_test, y_test))


plt.figure(figsize=(8, 6))
plt.plot(num_epochs, somJ_times, label='SoM (somJ)', marker='o')
plt.plot(num_epochs, minisom_times, label='MiniSom', marker='o')
plt.xlabel('Número de epochs')
plt.ylabel('Accuracy')
plt.title(f'Accuracy vs # epochs durante el entrenamiento')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'experimentos/g_time_comp/acc_vs_epochs.png', dpi=300)
plt.show()