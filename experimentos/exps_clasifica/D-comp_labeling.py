import os
import numpy as np
import matplotlib.pyplot as plt
from somJ.som import SoM
from somJ.functions import *
import somJ.config as config
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Parámetros generales
samples = [1, 5, 10, 25, 50, 75, 100, 150, 200]
datasets = ['Iris', 'Digits', 'MNIST', 'Fashion MNIST']
total_nodes = config.TOTAL_NODES
n_splits = 5

# Directorio de salida
out_dir = 'experimentos/exps_clasifica/csv'
os.makedirs(out_dir, exist_ok=True)

import pandas as pd

# Almacenar resultados en una lista de diccionarios
results_list = []

for ds in datasets:
    print(f"\n=== Procesando {ds} con 5-fold ===")
    X, y = load_dataset(ds)
    
    acc_folds = {s: [] for s in samples}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold_idx}/{n_splits}")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]
        
        som = SoM(
            method="pca",
            data=X_train,
            total_nodes=total_nodes
        )
        som.train(
            train_data=X_train,
            learn_rate=config.LEARNING_RATE,
            epochs=1,            
            update="online",
            prog_bar=False
        )
        
        for s in samples:
            cutoff = s
            X_lab = X_train[:cutoff]
            y_lab = y_train[:cutoff]
            y_pred = som.predict(X_lab, y_lab, X_test)
            acc = accuracy_score(y_test, y_pred)
            acc_folds[s].append(acc)
    
    for s in samples:
        arr = np.array(acc_folds[s])
        row = {
            'dataset': ds,
            'samples': s,
            'mean': arr.mean(),
            'std': arr.std(ddof=1)
        }
        results_list.append(row)
        print(f"  Muestras={s:3d} → mean={arr.mean():.3f}, std={arr.std(ddof=1):.3f}")

# Creamos el DataFrame final
results_df = pd.DataFrame(results_list)
print(results_df)

# Si quieres guardar a CSV:
results_df.to_csv(os.path.join(out_dir, "results_etiquetado.csv"), index=False)
