import pandas as pd
import numpy as np
import time
import somJ.config as config
from somJ.functions import *
from somJ.som import SoM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gc
import warnings

# Ignorar FutureWarnings y UserWarnings específicos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def make_subset(X_train, y_train, porc=0.9, clase_objetivo=0, total=200, random_state=42):
    np.random.seed(random_state)
    # Calcula el número de muestras para la clase objetivo
    n_objetivo = int(total * porc)
    # Calcula el número de muestras restantes y cómo se reparten
    n_resto = total - n_objetivo
    clases = np.unique(y_train)
    clases_resto = [c for c in clases if c != clase_objetivo]
    n_clases_resto = len(clases_resto)
    muestras_por_clase_resto = n_resto // n_clases_resto
    resto_extra = n_resto % n_clases_resto  # Por si no es divisible exacto

    idxs = []
    # Selecciona índices para la clase objetivo
    idx_obj = np.where(y_train == clase_objetivo)[0]
    idx_obj_sel = np.random.choice(idx_obj, n_objetivo, replace=False)
    idxs.extend(idx_obj_sel)

    # Selecciona índices para el resto de clases
    for i, c in enumerate(clases_resto):
        n = muestras_por_clase_resto + (1 if i < resto_extra else 0)
        idx = np.where(y_train == c)[0]
        idx_sel = np.random.choice(idx, n, replace=False)
        idxs.extend(idx_sel)

    # Barajamos el resultado final
    np.random.shuffle(idxs)
    X_subset = X_train[idxs]
    y_subset = y_train[idxs]
    return X_subset, y_subset

# Carga el dataset y configuración
X, y = load_dataset("Digits")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_results = []

for porc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9]:
    # Inicializa (vacía) métricas para cada porcentaje
    metrics = {
        "SOM":   {"accuracy": [], "precision": [], "recall": [], "f1": [], "train_time": []},
        "Random Forest": {"accuracy": [], "precision": [], "recall": [], "f1": [], "train_time": []},
    }
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_subset, y_subset = make_subset(X_train, y_train, porc=porc, clase_objetivo=0, total=150)
        unique, counts = np.unique(y_subset, return_counts=True)
        print(f"Porc: {porc} - Fold {fold}/{n_splits} - Distribución de clases en el subset: {dict(zip(unique, counts))}")

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
        som_pred = som.predict(X_subset, y_subset, X_test)
        t1 = time.time()

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
            random_state=42
        )
        t0 = time.time()
        rf.fit(X_subset, y_subset)
        rf_pred = rf.predict(X_test)
        t1 = time.time()

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

    # Almacena resultados globales para cada método
    for method, vals in metrics.items():
        all_results.append({
            "porc": porc,
            "method": method,
            "accuracy_mean": np.mean(vals["accuracy"]),
            "accuracy_std": np.std(vals["accuracy"], ddof=1),
            "precision_mean": np.mean(vals["precision"]),
            "precision_std": np.std(vals["precision"], ddof=1),
            "recall_mean": np.mean(vals["recall"]),
            "recall_std": np.std(vals["recall"], ddof=1),
            "f1_mean": np.mean(vals["f1"]),
            "f1_std": np.std(vals["f1"], ddof=1),
            "train_time_mean": np.mean(vals["train_time"]),
            "train_time_std": np.std(vals["train_time"], ddof=1),
        })

# Guarda en CSV
df_results = pd.DataFrame(all_results)
df_results.to_csv("experimentos/exps_clasifica/csv/results_mnist_desbalan.csv", index=False)
print("Resultados guardados en resultados_mnist_subset.csv")
