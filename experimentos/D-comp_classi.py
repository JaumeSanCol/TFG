import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import tracemalloc
from minisom import MiniSom
import somJ.config as config
from functions import *
from somJ.som import SoM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gc
import warnings
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Ignorar FutureWarnings y UserWarnings específicos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------------------------------------------------------------------
#   Experimento para observar la capacidad de clasificación de nuestra implementación de SOM respecto de MiniSOM
# ------------------------------------------------------------------------------------------------------------------------------
#   
#   Métricas:
#       -Acuraccy       -Precision         -Recall          -F1         -Tiempo de ejecución (s)       -Uso de memoria (Mb)
#
#   Los resultados son almacenados en un csv llamado resultados_Class.csv y resultados_Class_memory.csv dependiendo de si se ha 
#   medido medido el uso de memoria (puede suponer un incremento en el timepo de ejecución). Las graficas se guardan en g_class/

# ------------------------------------------------------------------------------------------------------------------------------

# Después de evaluar la accuracy respecto del numero de epochs, podemos observar que minisom no requiere de tantas muestras para
# obtener información por lo que con una epoch es suficiente. Poner "ajustado" verdadero, significa que para el experimento se
# usara una sola epoch de Minisom frente a las 50 de SOM. Si es falso, se usarán 50 epochs en cada algoritmo.

AJUSTADO=False


# Aplica MiniSom y devuelve el objeto entrenado

def apply_som(X_sample):
    epochs = config.EPOCHS
    som_shape = int(np.sqrt(config.TOTAL_NODES))
    minisom = MiniSom(som_shape, som_shape, X_sample.shape[1], 
                      sigma=1, learning_rate=config.LEARNING_RATE,
                      random_seed=42)
    if AJUSTADO:
        minisom.train_batch(X_sample, len(X_sample))
    else:
        for i in range(3):
            minisom.train_batch(X_sample, len(X_sample))
    return minisom

# Aplica tu implementación SoM y devuelve el objeto entrenado

def apply_som_j(X_scaled):
    som = SoM(method="pca", data=X_scaled, total_nodes=config.TOTAL_NODES)
    som.train(
        train_data=X_scaled,
        learn_rate=config.LEARNING_RATE,
        sigma=1,
        epochs=3,
        update="online",
        save=config.SAVE_HISTORY,
        prog_bar=False
    )
    return som

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
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted')
    f1   = f1_score(y_test, y_pred, average='weighted')
    return acc, prec, rec, f1

### --- FUNCIÓN GENERAL --- ###

def evaluate_all(datasets, memory, random_state=42, n_splits=5):
    results = []
    for name in datasets:
        print(f"\n Dataset: {name}")
        X, y = load_dataset(name)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            timer = {}
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X_train_scaled, X_test_scaled = X_scaled[train_idx], X_scaled[test_idx]

            # --- Embedding SOM custom ---
            if memory: tracemalloc.start()
            start = time.perf_counter()
            somJ = apply_som_j(X_train_scaled)
            X_som_train = np.array([somJ.find_winner(x) for x in X_train_scaled])
            X_som_test  = np.array([somJ.find_winner(x) for x in X_test_scaled])
            timer["SOM    "] = time.perf_counter() - start
            if memory:
                _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
                mem_som = peak / 1024 / 1024
            else:
                mem_som = 0

            # --- Embedding MiniSOM ---
            if memory: tracemalloc.start()
            start = time.perf_counter()
            minisom = apply_som(X_train_scaled)
            X_min_train = np.array([minisom.winner(x) for x in X_train_scaled])
            X_min_test  = np.array([minisom.winner(x) for x in X_test_scaled])
            timer["MiniSOM"] = time.perf_counter() - start
            if memory:
                _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
                mem_min = peak / 1024 / 1024
            else:
                mem_min = 0

            embeddings = {
                "SOM    ": (X_som_train, X_som_test, mem_som),
                "MiniSOM": (X_min_train, X_min_test, mem_min)
            }
            for method, (X_tr_emb, X_te_emb, used_mem) in embeddings.items():
                acc, prec, rec, f1 = evaluate_classification_embedded(
                    X_tr_emb, y_train, X_te_emb, y_test)
                results.append({
                    "Dataset": name,
                    "Método": method,
                    "Accuracy": round(acc, 3),
                    "Precision": round(prec, 3),
                    "Recall": round(rec, 3),
                    "F1": round(f1, 3),
                    "Time (s)": timer[method],
                    "Memory (Mb)": used_mem
                })
                print(f"{method}  Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, "
                      f"Time={timer[method]:.3f}s, Mem={used_mem:.3f}Mb")

            gc.collect()

    df = pd.DataFrame(results)
    return df

if __name__ == '__main__':
    datasets = ['Iris', 'Digits', 'MNIST', 'Fashion MNIST']
    df_results = evaluate_all(datasets, memory=True)
    df_mean = df_results.groupby(["Dataset","Método"]).mean(numeric_only=True).reset_index()
    print("\n Resultados promediados:")
    print(df_mean)
    if AJUSTADO:df_mean.to_csv("experimentos/resultados_Class_memory_ajustado.csv", index=False)
    else:df_mean.to_csv("experimentos/resultados_Class_memory_mejorado.csv", index=False)

    sns.set(style="whitegrid", font_scale=1.2)
    order = ["Iris","Digits","MNIST","Fashion MNIST"]
    metrics = ["Accuracy","Precision","Recall","F1","Time (s)","Memory (Mb)"]
    for metric in metrics:
        plt.figure(figsize=(10,6))
        ax = sns.barplot(x="Dataset", y=metric, hue="Método", data=df_mean, order=order)
        plt.title(f"{metric} promedio por Dataset y Método")
        plt.ylim(0, df_mean[metric].max()*1.1)
        plt.legend(title="Método", bbox_to_anchor=(1.02,1), loc='upper left')
        plt.tight_layout(rect=[0,0,0.85,1])
        if AJUSTADO:plt.savefig(f"experimentos/g_class/mejorado/grafico_{metric.lower()}_ajustado.png")
        else:plt.savefig(f"experimentos/g_class/mejorado/grafico_{metric.lower()}.png")
        plt.show()
