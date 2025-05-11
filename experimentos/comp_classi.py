import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import tracemalloc
from minisom import MiniSom
import somJ.config as config
from functions import *
from somJ.som import SoM
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
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

def apply_som(X_sample):
    epochs = config.EPOCHS
    som_shape = int(np.sqrt(config.TOTAL_NODES))
    minisom = MiniSom(som_shape, som_shape, len(X_sample[0]), sigma=config.RADIUS_SQ, learning_rate=config.LEARNING_RATE)
    minisom.train_batch(X_sample,epochs*len(X_sample))
    return minisom
def apply_som_j(X_scaled,):
    som = SoM(
            method=config.INIT_METHOD,
            data=X_scaled,
            total_nodes=config.TOTAL_NODES
        )
    som.train(
            train_data=X_scaled,
            learn_rate=config.LEARNING_RATE,
            radius_sq=config.RADIUS_SQ,
            lr_decay=config.LR_DECAY,
            radius_decay=config.RADIUS_DECAY,
            epochs=config.EPOCHS,
            update=config.UPDATE_METHOD,
            batch_size=config.BATCH_SIZE,
            step=config.STEP,
            save=config.SAVE_HISTORY,
            prog_bar=False
        )
    return som

def compute_continuity(X, X_embedded, n_neighbors=5):
    n_samples = X.shape[0]

    nn_orig = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    neigh_orig = nn_orig.kneighbors(X, return_distance=False)[:, 1:]

    nn_embed = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_embedded)
    neigh_embed = nn_embed.kneighbors(X_embedded, return_distance=False)[:, 1:]

    ranks = np.zeros(n_samples)
    for i in range(n_samples):
        missing = set(neigh_orig[i]) - set(neigh_embed[i])
        ranks[i] = len(missing)

    continuity = 1 - (np.mean(ranks) / n_neighbors)
    return continuity

def evaluate_classification(X_embedded, y, random_state=42):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_embedded, y, test_size=0.3, random_state=random_state, stratify=y
    )

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return acc, prec, rec, f1

### --- FUNCIÓN GENERAL --- ###

def evaluate_all(datasets,memory, random_state=42, n_splits=5):
    results = []

    for name in datasets:
        print(f"\n Dataset: {name}")
        X, y = load_dataset(name)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            timer={}
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X_train_scaled, X_test_scaled = X_scaled[train_idx], X_scaled[test_idx]

            # --- ENTRENAR EMBEDDING SOLO EN TRAIN ---
            if memory: tracemalloc.start() 
            start = time.perf_counter()
            somJ= apply_som_j(X_train_scaled)
            X_som_j = np.array([somJ.find_winner(x) for x in X_test_scaled])
            end = time.perf_counter()
            timer["SOM    "]=end-start
            if memory:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                used_memory_som=peak / 1024 / 1024
            else:used_memory_mini=0

            if memory: tracemalloc.start() 
            start = time.perf_counter()
            minisom= apply_som(X_train_scaled)
            X_minisom = np.array([minisom.winner(x) for x in X_test_scaled])
            end = time.perf_counter()
            timer["MiniSOM"]=end-start
            if memory:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                used_memory_mini=peak / 1024 / 1024
            else:
                used_memory_mini=0
            # --- EVALUACIONES ---
            embeddings = {
                "SOM    ": (X_som_j,used_memory_som),
                "MiniSOM": (X_minisom,used_memory_mini)
            }
            for method, (X_full_embedded,used_memory) in embeddings.items():

                acc, prec, rec, f1 = evaluate_classification(X_full_embedded, y_test)

                results.append({
                    "Dataset": name,
                    "Método": method,
                    "Accuracy": round(acc, 3),
                    "Precision": round(prec, 3),
                    "Recall": round(rec, 3),
                    "F1": round(f1, 3),
                    "Time (s)": timer[method],
                    "Memory (Mb)":used_memory
                })
                print(f"{method}    Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, Time: {timer[method]:3f}, Memory: {used_memory:3f}")

            gc.collect()

    df = pd.DataFrame(results)
    return df


datasets = [
            'Iris', 
            'Digits', 
            'MNIST', 'Fashion MNIST']

memory=True
df_results = evaluate_all(datasets,memory)

# Agrupar por Dataset y Método, y calcular la media de las métricas
df_mean = df_results.groupby(["Dataset", "Método"]).mean(numeric_only=True).reset_index()

print("\n✅ Resultados promediados por Dataset y Método:")
print(df_mean)

# Guardar también los promedios
if memory:df_mean.to_csv("experimentos/resultados_Class_memory.csv", index=False)
else:df_mean.to_csv("experimentos/resultados_Class.csv", index=False)

print("\n Resultados guardados como CSV.")


orden_datasets = ["Iris", "Digits", "MNIST", "Fashion MNIST"]


# Configuración de estilo
sns.set(style="whitegrid", font_scale=1.2)
metrics = ["Accuracy", "Precision", "Recall", "F1","Time (s)","Memory (Mb)"]

# Gráfico de barras para cada métrica
for metric in metrics:
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x="Dataset", y=metric, hue="Método",
        data=df_mean, palette="muted",
        order=orden_datasets
    )
    max_val = df_mean[metric].max()
    plt.title(f"{metric} promedio por Dataset y Método")
    plt.ylim(0, max_val * 1.1)

    plt.legend(title="Método", loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.ylabel(metric)
    plt.xlabel("Dataset")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"experimentos/g_class/grafico_{metric.lower()}.png")
    plt.show()
