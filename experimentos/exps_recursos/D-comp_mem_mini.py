import tracemalloc
import numpy as np
import pandas as pd
import time
import tracemalloc
from somJ.som import SoM
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import somJ.config as config
from somJ.functions import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_experiment_kfold(dataset_name, n_splits=5):
    X, y = load_dataset(dataset_name)
    X_scaled = MinMaxScaler().fit_transform(X)
    grid_size = int(np.sqrt(config.TOTAL_NODES))
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

   
    metrics = {
        "SOM":   {"memoria (MB)": [], "tiempo (s)": []},
        "MiniSOM": {"memoria (MB)": [], "tiempo (s)": []},
    }

    for fold, (train_idx, _) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}/{n_splits}")
        X_train = X_scaled[train_idx]

        # --- SoM (somJ) ---
        tracemalloc.start()
        t0 = time.time()
        som = SoM(method="pca",
                  data=X_train,
                  total_nodes=config.TOTAL_NODES)
        som.train(train_data=X_train,
                  learn_rate=config.LEARNING_RATE,
                  sigma=config.SIGMA,
                  decay_func_name="asymptotic_decay",
                  update="online")
        t1 = time.time()
        _, som_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics["SOM"]["memoria (MB)"].append(som_peak / (1024*1024))  # MB
        metrics["SOM"]["tiempo (s)"].append(t1 - t0)               # seconds

        # --- MiniSom ---
        tracemalloc.start()
        t0 = time.time()
        minisom = MiniSom(grid_size, grid_size, X_train.shape[1],
                          sigma=config.SIGMA,
                          learning_rate=config.LEARNING_RATE)
        minisom.random_weights_init(X_train)
        minisom.train_batch(X_train, len(X_train))  # 1 epoch
        t1 = time.time()
        _, minisom_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics["MiniSOM"]["memoria (MB)"].append(minisom_peak / (1024*1024))  # MB
        metrics["MiniSOM"]["tiempo (s)"].append(t1 - t0)                   # seconds

    return metrics

if __name__ == '__main__':
    datasets = ['Iris', 'Digits', 'MNIST', 'Fashion MNIST']
    all_results = []

    for ds in datasets:
        print(f"Dataset: {ds}")
        mets = run_experiment_kfold(ds, n_splits=5)
        for method, vals in mets.items():
            all_results.append({
                "Dataset": ds,
                "Method": method,
                "Memory peak (mean MB)": np.mean(vals["memoria (MB)"]),
                "Memory peak (std MB)":  np.std(vals["memoria (MB)"]),
                "Train time (mean s)": np.mean(vals["tiempo (s)"]),
                "Train time (std s)":  np.std(vals["tiempo (s)"]),
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("experimentos/exps_recursos/csv/resultados_memoria_minisom.csv", index=False)
