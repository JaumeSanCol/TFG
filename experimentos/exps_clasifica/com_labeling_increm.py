import pandas as pd
import numpy as np
import time
import somJ.config as config
from somJ.functions import *
from somJ.som import SoM
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def fill_labels(neuron_labels, shape, missing_value=-1, max_iter=10):
    matrix = np.full(shape, missing_value)
    for (i, j), value in neuron_labels.items():
        matrix[i, j] = value

    def fill_with_neighbor_mode(matrix, missing_value=-1):
        filled = matrix.copy()
        rows, cols = filled.shape

        for i in range(rows):
            for j in range(cols):
                if filled[i, j] == missing_value:
                    neighbors = []
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < rows and 0 <= nj < cols and filled[ni, nj] != missing_value:
                                neighbors.append(filled[ni, nj])
                    if neighbors:
                        counts = np.bincount(np.array(neighbors, dtype=int))
                        most_common = np.argmax(counts)
                        filled[i, j] = most_common
        return filled

    filled = matrix.copy()
    for _ in range(max_iter):
        prev = filled.copy()
        filled = fill_with_neighbor_mode(filled, missing_value=missing_value)
        if np.array_equal(prev, filled):
            break

    # Convertir matriz resultante a diccionario con todas las coordenadas
    new_labels = {(i, j): int(filled[i, j]) for i in range(shape[0]) for j in range(shape[1])}
    return new_labels


if __name__ == '__main__':
    datasets = ["Iris","Digits", "MNIST", "Fashion MNIST"]
    n_splits = 5
    n_max = 500  

    all_results = []

    for dataset_name in datasets:
        print(f"\nProcesando dataset: {dataset_name}")
        X, y = load_dataset(dataset_name)
        acc_all_folds = []
        acc_rf_all_folds = []

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            som = SoM(
                method="pca",
                data=X_train,
                total_nodes=config.TOTAL_NODES
            )
            som.train(
                train_data=X_train,
                learn_rate=config.LEARNING_RATE,
                sigma=1,
                epochs=1,
            )
            X_train_scaled = som.scaler.transform(X_train)
            X_test_scaled = som.scaler.transform(X_test)
            X_train_emb = np.array([som.find_winner(x) for x in X_train_scaled])
            X_test_emb  = np.array([som.find_winner(x) for x in X_test_scaled])

            acc = []
            for n in range(1, n_max):
                label_map = {}
                for coord, label in zip(X_train_emb[:n], y_train[:n]):
                    coord = tuple(coord)
                    label_map.setdefault(coord, []).append(label)
                neuron_labels = {coord: max(labels, key=labels.count)
                                 for coord, labels in label_map.items()}
                neuron_labels = fill_labels(neuron_labels, som.grid_size)
                y_pred = []
                for coord in X_test_emb:
                    coord = tuple(coord)
                    if coord in neuron_labels:
                        y_pred.append(neuron_labels[coord])
                    else:
                        dists = [ (coord[0]-c[0])**2 + (coord[1]-c[1])**2
                                  for c in neuron_labels.keys() ]
                        nearest = list(neuron_labels.keys())[np.argmin(dists)]
                        y_pred.append(neuron_labels[nearest])
                y_pred = np.array(y_pred)
                acc.append(accuracy_score(y_test, y_pred))

            acc_all_folds.append(acc)

            acc_rf = []
            for n in range(1, n_max):
                rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    random_state=42
                )
                rf.fit(X_train[:n], y_train[:n])
                rf_pred = rf.predict(X_test)
                acc_rf.append(accuracy_score(y_test, rf_pred))
            acc_rf_all_folds.append(acc_rf)

        acc_all_folds = np.array(acc_all_folds)
        acc_rf_all_folds = np.array(acc_rf_all_folds)
        acc_mean = acc_all_folds.mean(axis=0)
        acc_std = acc_all_folds.std(axis=0)
        acc_rf_mean = acc_rf_all_folds.mean(axis=0)
        acc_rf_std = acc_rf_all_folds.std(axis=0)

        results_df = pd.DataFrame({
            'n': np.arange(1, n_max),
            'SOM_mean': acc_mean,
            'SOM_std': acc_std,
            'RF_mean': acc_rf_mean,
            'RF_std': acc_rf_std,
            'dataset': dataset_name
        })
        all_results.append(results_df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("experimentos/exps_clasifica/csv/results_etiquetado_incremental.csv", index=False)
    