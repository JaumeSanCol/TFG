
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from somJ.som import SoM
import somJ.config as config
import os
import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import umap
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



def dim_exp(dimensions,n_splits,n_samples):
    # --- Preparar 5-fold ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # --- Diccionarios para acumular tiempos ---
    times_som  = {d: [] for d in dimensions}
    times_pca  = {d: [] for d in dimensions}
    times_lda  = {d: [] for d in dimensions}
    times_tsne = {d: [] for d in dimensions}
    times_umap = {d: [] for d in dimensions}

    for d in dimensions:
        print(f"\n=== Dimensión objetivo: {d} ===")
        rand = np.random.RandomState(0)
        train_data = rand.randint(0, 255, (n_samples, d))
        label_data = rand.randint(0, 10, (n_samples, 1))
        scaler     = MinMaxScaler()
        X_scaled   = scaler.fit_transform(train_data)

        for fold_idx, (train_idx, _) in enumerate(kf.split(X_scaled), start=1):
            X_train = X_scaled[train_idx]
            y_train = label_data[train_idx]
            print(f" Fold {fold_idx}/{n_splits}", end='… ')
            
            # SoM 
            som = SoM(
                method=config.INIT_METHOD,
                data=X_train,
                total_nodes=config.TOTAL_NODES
            )
            t0 = time.time()
            som.train(
                train_data=X_train,
                learn_rate=config.LEARNING_RATE,
                epochs=1,
                sigma=config.SIGMA,
                decay_rate=config.DECAY_RATE,
                prog_bar=False
            )
            times_som[d].append(time.time() - t0)
            # ---- PCA ----
            t0 = time.time()
            pca = PCA(n_components=2, random_state=42)
            _ = pca.fit_transform(X_train)
            times_pca[d].append(time.time() - t0)
            
            # ---- LDA ----
            t0 = time.time()
            lda = LinearDiscriminantAnalysis(n_components=2)
            _ = lda.fit_transform(X_train, y_train)
            times_lda[d].append(time.time() - t0)
            
            # ---- t-SNE ----
            t0 = time.time()
            tsne = TSNE(n_components=2, init='random', random_state=42, verbose=0)
            _ = tsne.fit_transform(X_train)
            times_tsne[d].append(time.time() - t0)
            
            # ---- UMAP ----
            t0 = time.time()
            umap_model = umap.UMAP(n_components=2, random_state=42)
            _ = umap_model.fit_transform(X_train)
            times_umap[d].append(time.time() - t0)
            
            print("OK")

    results = []
    for d in dimensions:
        row = {"dims": d}
        for name, times_dict in [("SOM", times_som),
                                    ("PCA", times_pca),
                                ("LDA", times_lda),
                                ("t-SNE", times_tsne),
                                ("UMAP", times_umap)]:
            arr = np.array(times_dict[d], dtype=float)
            row[f"{name}_mean"] = np.nanmean(arr)
            row[f"{name}_std"]  = np.nanstd(arr, ddof=1)
        results.append(row)
    return results


if __name__ == '__main__':
    # --- Parámetros ---
    n_samples   = 40000
    dimensions  = [10, 50, 100,250,500,750 ,1000]
    n_splits    = 5

    results=dim_exp(dimensions,n_splits,n_samples)
    df_results = pd.DataFrame(results)
    print("\nResumen de tiempos (media ± std) por técnica y dimensión:")
    df_results.to_csv("experimentos/exps_recursos/csv/resultados_dimensiones_xl.csv",index=False)