
import somJ.config as config
from somJ.functions import *
from somJ.som import SoM

import gc
import numpy as np
import pandas as pd
import time
import tracemalloc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import trustworthiness
import umap

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



def evaluate_all(datasets, eval_subset_size=30000,n_neighbors=5, random_state=42, n_splits=5):
    all_results = []

    for name in datasets:
        print(f"\nDataset: {name}")
        X, y = load_dataset(name)
        print(f"tamaño dataset: {len(X)}")
      
        # X, y = load_dataset(name)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        method_names = ['PCA', 'LDA', 'UMAP', 't-SNE', 'SOM']
        results_by_method = {k: [] for k in method_names}
        tracemalloc.stop()
        for fold, (train_idx, _) in enumerate(skf.split(X, y), 1):
            X_train, y_train = X[train_idx], y[train_idx]
            print(f" Fold {fold}/{n_splits} Tamaño fold:{len(X_train)}")

            # PCA
            gc.collect()
            tracemalloc.start()
            t0 = time.perf_counter()
            pca_emb = PCA(n_components=2, random_state=random_state).fit_transform(X_train)
            t = time.perf_counter() - t0
            current, peak = tracemalloc.get_traced_memory()
            mem = peak / 1e6  # Convertir a MB
            tracemalloc.stop()
            gc.collect()
            results_by_method['PCA'].append({"dataset": name, "method": 'PCA', "fold": fold, "tiempo (s)": t, "memoria (Mb)": mem})

            # LDA
            gc.collect()
            tracemalloc.start()
            t0 = time.perf_counter()
            lda_emb = LinearDiscriminantAnalysis(n_components=2).fit_transform(X_train, y_train)
            t = time.perf_counter() - t0
            current, peak = tracemalloc.get_traced_memory()
            mem = peak / 1e6  # Convertir a MB
            tracemalloc.stop()
            gc.collect()
            results_by_method['LDA'].append({"dataset": name, "method": 'LDA', "fold": fold, "tiempo (s)": t, "memoria (Mb)": mem})

            # UMAP
            gc.collect()
            tracemalloc.start()
            t0 = time.perf_counter()
            umap_emb = umap.UMAP(n_components=2, random_state=random_state,low_memory=True).fit_transform(X_train)
            t = time.perf_counter() - t0
            current, peak = tracemalloc.get_traced_memory()
            mem = peak / 1e6  # Convertir a MB
            tracemalloc.stop()
            gc.collect()
            results_by_method['UMAP'].append({"dataset": name, "method": 'UMAP', "fold": fold, "tiempo (s)": t, "memoria (Mb)": mem})

            # t-SNE
            gc.collect()
            tracemalloc.start()
            t0 = time.perf_counter()
            tsne_emb = TSNE(n_components=2, random_state=random_state, perplexity=30).fit_transform(X_train)
            t = time.perf_counter() - t0
            current, peak = tracemalloc.get_traced_memory()
            mem = peak / 1e6  # Convertir a MB
            tracemalloc.stop()
            gc.collect()
            results_by_method['t-SNE'].append({"dataset": name, "method": 't-SNE', "fold": fold, "tiempo (s)": t, "memoria (Mb)": mem})

            # SOM
            gc.collect()
            tracemalloc.start()
            t0 = time.perf_counter()
            som = SoM(method=config.INIT_METHOD, data=X_train[:1000], total_nodes=config.TOTAL_NODES)
            som.train(train_data=X_train)
            som_emb = som.find_all_winner(X_train)
            t = time.perf_counter() - t0
            current, peak = tracemalloc.get_traced_memory()
            mem = peak / 1e6  # Convertir a MB
            tracemalloc.stop()
            gc.collect()
            results_by_method['SOM'].append({"dataset": name, "method": 'SOM', "fold": fold, "tiempo (s)": t, "memoria (Mb)": mem,})

        # Agregamos media y desviación por método
        for method_name, records in results_by_method.items():
            df_method = pd.DataFrame(records)
            stats = df_method[["tiempo (s)", "memoria (Mb)"]].agg(['mean', 'std']).T
            stats.columns = ['mean', 'std']
            stats["method"] = method_name
            stats["dataset"] = name
            all_results.append(stats.reset_index())

    df_results = pd.concat(all_results, ignore_index=True)
    return df_results


if __name__ == '__main__':
    datasets = [
                'Iris', 
                'Digits', 
                'MNIST', 'Fashion MNIST']

    df_results = evaluate_all(datasets)
    # Guardar también los promedios
    df_results.to_csv("experimentos/exps_recursos/csv/resultados_memory.csv", index=False)