import pandas as pd
import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier
import somJ.config as config
from somJ.functions import *
from somJ.som import SoM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gc
import time
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import warnings
# Ignorar FutureWarnings y UserWarnings específicos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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


def evaluate_all(datasets,limitado=False, random_state=42, n_splits=5):
    """
    Para cada nombre en `datasets`:
      - carga X, y
      - aplica StratifiedKFold
      - escala con MinMaxScaler
      - entrena SOM (1 época) y RF
      - en cada fold calcula accuracy, precision, recall, f1 y tiempo de entrenamiento
    Devuelve un DataFrame con columnas:
      ['dataset',
       'method',
       'accuracy_mean','accuracy_std',
       'precision_mean','precision_std',
       'recall_mean','recall_std',
       'f1_mean','f1_std',
       'train_time_mean','train_time_std']
    """
    all_results = []

    for name in datasets:
        print(f"\nEvaluando dataset: {name}")
        X, y = load_dataset(name)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # contenedores de métricas
        metrics = {
            "SOM":   {"accuracy": [], "precision": [], "recall": [], "f1": [], "train_time": []},
            "Random Forest":    {"accuracy": [], "precision": [], "recall": [], "f1": [], "train_time": []},
        }

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"Fold {fold}/{n_splits}")
            # split
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # --- Entrena y predice con SOM ---
            som = SoM(
                method="pca",
                data=X_train,
                total_nodes=config.TOTAL_NODES
            )
            t0 = time.time()
            som.train(
                train_data=X_train,
                learn_rate=config.LEARNING_RATE,
                sigma=1,
                epochs=1,
            )
            if limitado:som_pred=som.predict(X_train[:200],y_train[:200],X_test)
            else:som_pred=som.predict(X_train,y_train,X_test)
            t1 = time.time()
            # mide métricas SOM
            metrics["SOM"]["train_time"].append(t1 - t0)
            metrics["SOM"]["accuracy"].append(accuracy_score(y_test, som_pred))
            metrics["SOM"]["precision"].append(
                precision_score(y_test, som_pred, average="macro", zero_division=0)
            )
            metrics["SOM"]["recall"].append(
                recall_score(y_test, som_pred, average="macro", zero_division=0)
            )
            metrics["SOM"]["f1"].append(
                f1_score(y_test, som_pred, average="macro", zero_division=0)
            )
            # --- Entrena y predice con Random Forest ---
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=random_state
            )
            t0 = time.time()
            if limitado:rf.fit(X_train[:200], y_train[:200])
            else:rf.fit(X_train, y_train)
            
            rf_pred = rf.predict(X_test)
            t1 = time.time()
            # mide métricas RF
            metrics["Random Forest"]["train_time"].append(t1 - t0)
            metrics["Random Forest"]["accuracy"].append(accuracy_score(y_test, rf_pred))
            metrics["Random Forest"]["precision"].append(
                precision_score(y_test, rf_pred, average="macro", zero_division=0)
            )
            metrics["Random Forest"]["recall"].append(
                recall_score(y_test, rf_pred, average="macro", zero_division=0)
            )
            metrics["Random Forest"]["f1"].append(
                f1_score(y_test, rf_pred, average="macro", zero_division=0)
            )

            gc.collect()

        # Agrega al resultado general
        for method, vals in metrics.items():
            all_results.append({
                "dataset":          name,
                "method":           method,
                "accuracy_mean":    np.mean(vals["accuracy"]),
                "accuracy_std":     np.std(vals["accuracy"], ddof=1),
                "precision_mean":   np.mean(vals["precision"]),
                "precision_std":    np.std(vals["precision"], ddof=1),
                "recall_mean":      np.mean(vals["recall"]),
                "recall_std":       np.std(vals["recall"], ddof=1),
                "f1_mean":          np.mean(vals["f1"]),
                "f1_std":           np.std(vals["f1"], ddof=1),
                "train_time_mean":  np.mean(vals["train_time"]),
                "train_time_std":   np.std(vals["train_time"], ddof=1),
            })

    # Devuelve un DataFrame ordenado
    cols = [
        'dataset','method',
        'accuracy_mean','accuracy_std',
        'precision_mean','precision_std',
        'recall_mean','recall_std',
        'f1_mean','f1_std',
        'train_time_mean','train_time_std'
    ]
    return pd.DataFrame(all_results, columns=cols)


if __name__ == '__main__':
    datasets = ['Iris', 'Digits', 'MNIST', 'Fashion MNIST']
    limitado=True
    df_results = evaluate_all(datasets, limitado=limitado)

    # Guardar DataFrame en CSV
    if limitado:name = 'results_supervisado_limitado.csv'
    else:name = 'results_supervisado.csv'
    df_results.to_csv(f"experimentos/exps_clasifica/csv/{name}", index=False)
