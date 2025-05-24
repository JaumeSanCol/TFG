import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from somJ.som import SoM
from functions import load_dataset
import somJ.config as config

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Parámetros generales
samples=[1,5,10,25,50,75,100,150,200]
datasets = ['Iris', 'Digits', 'MNIST', 'Fashion MNIST']
total_nodes = config.TOTAL_NODES

# Aquí guardaremos los resultados
results = {}

for ds in datasets:
    print(f"\n=== Procesando {ds} ===")
    # 1) Carga y normalización
    X, y = load_dataset(ds)

    # 2) Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # 3) Entrenamos el SOM (SoM-J)
    som = SoM(
        method="pca",
        data=X_train,
        total_nodes=total_nodes
    )
    som.train(
        train_data=X_train,
        learn_rate=config.LEARNING_RATE,
        epochs=1,               # o el nº de epochs que necesites
        update="online",
        prog_bar=True
    )
    X_som_train = np.array(som.find_all_winner(X_train))
    results[ds] = []
    for s in samples:
        print(f"  Etiquetando solo {s} muestras…", end=" ")
        # tomamos solo la fracción per de X_train/y_train
        cutoff = s
        X_lab = X_som_train[:cutoff]
        y_lab = y_train[:cutoff]

        y_pred=som.predict(X_lab,y_lab,X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Acc={acc:.3f}")
        results[ds].append(acc)

# 5) Graficamos todas las curvas en una sola figura
plt.figure(figsize=(10,6))
for ds in datasets:
    plt.plot(samples, results[ds],
             marker='o', linestyle='-',
             label=ds)

plt.xlabel('Numero de muestras usado en la etiquetación')
plt.ylabel('Accuracy')
plt.title('Accuracy vs # de muestras para el etiquetado distintos datasets')
plt.grid(True)
plt.legend()

#Guardar
out_dir = 'g_label'
os.makedirs(out_dir, exist_ok=True)
fp = os.path.join(out_dir, 'accuracy_vs_samples_all_datasets.png')
plt.savefig(fp, dpi=150, bbox_inches='tight')
print(f"\nGráfica guardada en {fp}")

plt.show()
