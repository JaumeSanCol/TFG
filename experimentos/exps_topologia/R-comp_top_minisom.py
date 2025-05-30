import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from somJ.som import SoM
from minisom import MiniSom
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import somJ.config as config
from somJ.functions import *

import warnings
# Ignorar FutureWarnings y UserWarnings específicos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_experiment_kfold(dataset_name, n_splits=5):
    X, y = load_dataset(dataset_name)
    X_scaled = MinMaxScaler().fit_transform(X)
    grid_size = int(np.sqrt(config.TOTAL_NODES))
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    som_scores = []
    mini_scores = []

    max_samples=30000
    max_samples_c=20000
    metrics = {
        "SOM":   {"trust": [], "cont": [], "C": []},
        "MiniSOM":    {"trust": [], "cont": [], "C": []},
    }
    for fold, (train_idx,_) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}/{n_splits}")
        X_train= X_scaled[train_idx]

        # --- SoM (somJ) ---
        som = SoM(method="pca",
                  data=X_train,
                  total_nodes=config.TOTAL_NODES)
        som.train(train_data=X_train,
                  learn_rate=config.LEARNING_RATE,
                  sigma=config.SIGMA,
                  decay_func_name="asymptotic_decay",
                  update="online")
        m, n = som.grid_size
        M = som.som_map.reshape(-1, som.input_dim)

        # Límite del hardware
        if len(X)>max_samples:
            X_train=X_train[:max_samples]
        X_embed = np.array(som.find_all_winner(X_train))


        trust = trustworthiness(X_train, X_embed, n_neighbors=5)
        cont  = compute_continuity(X_train, X_embed, n_neighbors=5)

        # Límite del hardware
        if len(X)>max_samples_c:
            X_train=X_train[:max_samples_c]
        X_embed = np.array(som.find_all_winner(X_train))

        C_val = compute_som_continuity_quantization(X_train, M, m, n)
        
        metrics["SOM"]["trust"].append(trust)
        metrics["SOM"]["cont"].append(cont)
        metrics["SOM"]["C"].append(C_val)

        # --- MiniSom ---
        X_train = X_scaled[train_idx] # Recargamos todo el train
        minisom = MiniSom(grid_size, grid_size, X_train.shape[1],
                          sigma=config.SIGMA,
                          learning_rate=config.LEARNING_RATE)
        minisom.random_weights_init(X_train)
        minisom.train_batch(X_train, len(X_train))  # 1 epoch
        m, n = 10,10
        M =  minisom.get_weights().reshape(-1, X_train.shape[1])

        # Límite del hardware
        if len(X)>max_samples:
            X_train=X_train[:max_samples]
        X_embed = np.array([minisom.winner(x) for x in X_train])


        trust = trustworthiness(X_train, X_embed, n_neighbors=5)
        cont  = compute_continuity(X_train, X_embed, n_neighbors=5)

        # Límite del hardware
        if len(X)>max_samples_c:
            X_train=X_train[:max_samples_c]
        X_embed = np.array(som.find_all_winner(X_train))

        C_val = compute_som_continuity_quantization(X_train, M, m, n)
        
        metrics["MiniSOM"]["trust"].append(trust)
        metrics["MiniSOM"]["cont"].append(cont)
        metrics["MiniSOM"]["C"].append(C_val)
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
                "Trust (mean)": np.mean(vals["trust"]),
                "Trust (std)":  np.std(vals["trust"]),
                "Continuity (mean)": np.mean(vals["cont"]),
                "Continuity (std)":  np.std(vals["cont"]),
                "C (mean)": np.mean(vals["C"]),
                "C (std)":  np.std(vals["C"]),
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("experimentos/exps_topologia/csv/resultados_topologica_minisom.csv", index=False)