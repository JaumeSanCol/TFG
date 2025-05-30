import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import pandas as pd

from somJ.som import SoM
from minisom import MiniSom
import somJ.config as config
from somJ.functions import load_dataset

from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



sample_sizes = [1000, 5000, 10000, 20000, 40000]
n_splits     = 5

# Carga y normalización
X_all, y_all = load_dataset("MNIST")
scaler   = MinMaxScaler()
X_all    = scaler.fit_transform(X_all)

# Para guardar resultados de cada fold
times_som  = {s: [] for s in sample_sizes}
times_pca  = {s: [] for s in sample_sizes}
times_lda  = {s: [] for s in sample_sizes}
times_tsne = {s: [] for s in sample_sizes}
times_umap = {s: [] for s in sample_sizes}

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

for n in sample_sizes:
    print(f"\n=== Muestras: {n} ===")
    X_sample = X_all[:(n*5)]
    y_sample = y_all[:(n*5)]
    # 5-fold splits
    for fold, (train_idx, _) in enumerate(kf.split(X_sample), 1):
        X_train = X_sample[train_idx]
        y_train = y_sample[train_idx]
        print(f" Fold {fold}/{n_splits}", end='… ')
        
        # --- SoM (somJ) ---
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
            prog_bar=False
        )
        times_som[n].append(time.time() - t0)
        
        # ---- PCA ----
        t0 = time.time()
        pca = PCA(n_components=2, random_state=42)
        _ = pca.fit_transform(X_train)
        times_pca[n].append(time.time() - t0)
        
        # ---- LDA ----
        t0 = time.time()
        lda = LinearDiscriminantAnalysis(n_components=2)
        _ = lda.fit_transform(X_train, y_train)
        times_lda[n].append(time.time() - t0)
        
        # ---- t-SNE ----
        t0 = time.time()
        tsne = TSNE(n_components=2, init='random', random_state=42, verbose=0)
        _ = tsne.fit_transform(X_train)
        times_tsne[n].append(time.time() - t0)
        
        # ---- UMAP ----
        t0 = time.time()
        umap_model = umap.UMAP(n_components=2, random_state=42)
        _ = umap_model.fit_transform(X_train)
        times_umap[n].append(time.time() - t0)
        
        
        print("OK")

results = []
for sample in sample_sizes:
    row = {"size": sample}
    for name, times_dict in [("SOM", times_som),
                                ("PCA", times_pca),
                             ("LDA", times_lda),
                             ("t-SNE", times_tsne),
                             ("UMAP", times_umap)]:
        arr = np.array(times_dict[sample], dtype=float)
        row[f"{name}_mean"] = np.nanmean(arr)
        row[f"{name}_std"]  = np.nanstd(arr, ddof=1)
    results.append(row)
df_results = pd.DataFrame(results)
df_results.to_csv("experimentos/exps_recursos/csv/resultados_muestras.csv",index=False)