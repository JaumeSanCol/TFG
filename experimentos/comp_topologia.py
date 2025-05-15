import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE, trustworthiness
import umap
import time
import somJ.config as config
from functions import *
from somJ.som import SoM
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import gc
import warnings
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Ignorar FutureWarnings y UserWarnings específicos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------------------------------------------------------------------
#   Experimento para observar la capacidad preservar la topologia de nuestra implementación de SOM respecto de UMAP y TSNE
# ------------------------------------------------------------------------------------------------------------------------------
#   
#   Métricas:
#       -Trustworthiness       -Continuity
#
#   Los resultados son almacenados en un csv llamado resultados_topologia. Las graficas se guardan en g_top/

# ------------------------------------------------------------------------------------------------------------------------------


def apply_umap(X):
    reducer = umap.UMAP(n_components=2, random_state=42)
    return reducer.fit_transform(X) 

def apply_tsne(X):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    return tsne.fit_transform(X)

# Devuleve las muestras como las coordenadas del nodo ganador
def apply_som_j(X_scaled):
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
    pos=np.array([som.find_winner(x) for x in X_scaled])
    (m, n)=som.grid_size
    return pos,m,n

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

def evaluate_embedding(X, X_embedded, n_neighbors=5):
    trust = trustworthiness(X, X_embedded, n_neighbors=n_neighbors)
    cont = compute_continuity(X, X_embedded, n_neighbors=n_neighbors)
    return trust, cont


def compute_som_histogram(positions, labels):
    #Convierte el SOM winner positions into a histogram (dict) per neuron position.
    som_grid = {}
    for pos, label in zip(positions, labels):
        pos_tuple = tuple(pos)
        if pos_tuple not in som_grid:
            som_grid[pos_tuple] = {}
        som_grid[pos_tuple][label] = som_grid[pos_tuple].get(label, 0) + 1
    return som_grid

# Representar mapas en 2D 
def plot_embedding(X, y, method_name, dataset_name,m_neurons =0,n_neurons=0):
    if method_name == "SOM":
        if not isinstance(X, dict):
            raise TypeError(f"Expected dict for SOM, got {type(X)}")

        label_names = np.unique(y)
        fig = plt.figure(figsize=(10, 10))
        the_grid = gridspec.GridSpec(n_neurons, m_neurons, fig)

        for position in X:
            label_fracs = [X[position].get(label, 0) for label in label_names]
            ax = plt.subplot(the_grid[n_neurons - 1 - position[1], position[0]], aspect=1)
            patches, _ = ax.pie(label_fracs, colors=plt.cm.tab10.colors, radius=1)

        plt.tight_layout()
        plt.savefig(f'experimentos/g_top/{method_name}_{dataset_name}_pie.png', dpi=300)

    else:
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(y)
        markers = ['o', 's', '^', 'P', '*', 'X', 'D', 'v', '<', '>']
        colors = sns.color_palette('tab10', len(unique_labels))

        for idx, label in enumerate(unique_labels):
            indices = np.where(y == label)
            plt.scatter(X[indices, 0], X[indices, 1], label=f'Clase {label}',
                        marker=markers[idx % len(markers)], color=colors[idx],
                        s=50, alpha=0.7, edgecolors='w')

        plt.legend(title='Clases', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'{method_name} - {dataset_name}')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'experimentos/g_top/{method_name}_{dataset_name}.png', dpi=300)


### --- FUNCIÓN GENERAL --- ###

def evaluate_all(datasets, eval_subset_size=5000, n_neighbors=5, random_state=42, n_splits=5):
    results = []

    for name in datasets:
        print(f"\n Dataset: {name}")
        X, y = load_dataset(name)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        timer={}

        # Obtener los embedings de
        start = time.perf_counter()
        X_umap= apply_umap(X)
        end = time.perf_counter()
        timer["UMAP"]=end-start

        start = time.perf_counter()
        X_tsne= apply_tsne(X)
        end = time.perf_counter()
        timer["t-SNE"]=end-start

        start = time.perf_counter()
        X_som_j,m,n= apply_som_j(X)
        end = time.perf_counter()
        timer["SOM"]=end-start

        # Para evaluar, necesitaremos comprobar el tamaño del dataset, y para dataset grandes utilizaremos un número de muestras reducidos

        n_samples = len(X)
        if n_samples > eval_subset_size:
            np.random.seed(random_state)
            idx = np.random.choice(n_samples, eval_subset_size, replace=False)
        else:
            idx = np.arange(n_samples)

        # Hacemos subsets con las mismas muestras

        X_subset = X[idx]
        y_subset = y[idx]
        X_umap_subset = X_umap[idx]
        X_tsne_subset = X_tsne[idx]
        X_som_j_subset = X_som_j[idx]

        # --- EVALUACIONES ---
        embeddings = {
            "UMAP": X_umap_subset,
            "t-SNE": X_tsne_subset,
            "SOM": X_som_j_subset
        }
        for method, X_embedded_subset in embeddings.items():
            print(f" - Método: {method}")

            trust, cont = evaluate_embedding(X_subset, X_embedded_subset, n_neighbors=n_neighbors)

            results.append({
                "Dataset": name,
                "Método": method,
                "Trustworthiness": round(trust, 3),
                "Continuity": round(cont, 3),
                "Time": timer[method]
            })

            print(f"    Trustworthiness={trust:.3f}, Continuity={cont:.3f},Time: {timer[method]:3f}")
            if method == "SOM":
                som_histogram = compute_som_histogram(X_embedded_subset, y_subset)
                plot_embedding(som_histogram, y_subset, method, name,m,n)
            else:
                plot_embedding(X_embedded_subset, y_subset, method, name)
            gc.collect()

    df = pd.DataFrame(results)
    return df


datasets = [
            'Iris', 
            'Digits', 
            'MNIST', 'Fashion MNIST']

df_results = evaluate_all(datasets)

print("\n Resultados por Dataset y Método:")
print(df_results)

# Guardar también los promedios
df_results.to_csv("experimentos/resultados_topologia.csv", index=False)

print("\n Resultadosguardados como CSV")


# Graficos -------------------------------------------------------------------------

sns.set_context("talk", font_scale=1.2)
sns.set_style("whitegrid")

# Ejecutar la evaluación previamente definida

# Trustworthiness
plt.figure(figsize=(12, 8))
sns.barplot(data=df_results, x="Dataset", y="Trustworthiness", hue="Método")
plt.title('Comparación de Trustworthiness por método y dataset')
plt.ylim(0, 1)
plt.ylabel('Trustworthiness')
plt.xlabel('Dataset')
plt.legend(title='Método', loc='upper right')
plt.tight_layout()
plt.savefig('experimentos/g_top/trustworthiness_comparacion.png', dpi=300)

# Continuity
plt.figure(figsize=(12, 8))
sns.barplot(data=df_results, x="Dataset", y="Continuity", hue="Método")
plt.title('Comparación de Continuity por método y dataset')
plt.ylim(0, 1)
plt.ylabel('Continuity')
plt.xlabel('Dataset')
plt.legend(title='Método', loc='upper right')
plt.tight_layout()
plt.savefig('experimentos/g_top/continuity_comparacion.png', dpi=300)

# Tiempo de ejecución
plt.figure(figsize=(12, 8))
sns.barplot(data=df_results, x="Dataset", y="Time", hue="Método")
plt.title('Tiempo de ejecución por método y dataset')
plt.ylabel('Tiempo (segundos)')
plt.xlabel('Dataset')
plt.legend(title='Método', loc='upper left')
plt.tight_layout()
plt.savefig('experimentos/g_top/tiempo_ejecucion_comparacion.png', dpi=300)
