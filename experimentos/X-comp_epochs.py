import numpy as np
import matplotlib.pyplot as plt
from somJ.som import SoM
from functions import load_dataset
import somJ.config as config
from minisom import MiniSom
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import matplotlib.colors as mcolors
# ------------------------------------------------------------------------------------------------------------------------------
#   Experimento para observar la accuracy respecto del numero de epochs usadas durante el entrenamiento
# ------------------------------------------------------------------------------------------------------------------------------
#
#   Los resultados son almacenados en una grafica en la carpeta g_time_comp
#
# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
#   Funciones auxiliares
# ------------------------------------------------------------------------------------------------------------------------------

def evaluate_classification_embedded(X_train_emb, y_train, X_test_emb, y_test):
    """
    Etiqueta cada neurona por mayoría en train y clasifica test usando neurona más cercana.
    """
    label_map = {}
    for coord, label in zip(X_train_emb, y_train):
        coord = tuple(coord)
        label_map.setdefault(coord, []).append(label)
    neuron_labels = {coord: max(labels, key=labels.count)
                     for coord, labels in label_map.items()}

    y_pred = []
    for coord in X_test_emb:
        coord = tuple(coord)
        if coord in neuron_labels:
            y_pred.append(neuron_labels[coord])
        else:
            dists = [(coord[0]-c[0])**2 + (coord[1]-c[1])**2
                     for c in neuron_labels.keys()]
            nearest = list(neuron_labels.keys())[np.argmin(dists)]
            y_pred.append(neuron_labels[nearest])
    return accuracy_score(y_test, np.array(y_pred))


def adjust_color(color, factor):
    """
    Aclara (factor>1) u oscurece (factor<1) una tupla RGB.
    """
    c = np.array(mcolors.to_rgb(color))
    c = np.clip(c * factor, 0, 1)
    return tuple(c)


# ------------------------------------------------------------------------------------------------------------------------------
#   Experimento con K-Fold Cross-Validation
# ------------------------------------------------------------------------------------------------------------------------------

def run_experiment_kfold(dataset_name, num_epochs, total_nodes, n_splits=5):
    print(f"Procesando {dataset_name} con {n_splits}-Fold CV…")
    X, y = load_dataset(dataset_name)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    fixed_dim = X_scaled.shape[1]
    grid_size = int(np.sqrt(total_nodes))

    # Preparar K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    somJ_means, somJ_stds = [], []
    minisom_means, minisom_stds = [], []

    for epoch in num_epochs:
        somJ_scores, minisom_scores = [], []

        for train_idx, val_idx in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # ----- SoM (somJ) -----
            som = SoM(method="pca", data=X_train, total_nodes=total_nodes)
            som.train(
                train_data=X_train,
                learn_rate=config.LEARNING_RATE,
                epochs=epoch,
                decay_func_name="exp",
                update="online",
                batch_size=config.BATCH_SIZE,
                save=config.SAVE_HISTORY,
                prog_bar=True
            )
            X_som_train = np.array([som.find_winner(x) for x in X_train])
            X_som_val   = np.array([som.find_winner(x) for x in X_val])
            somJ_scores.append(
                evaluate_classification_embedded(X_som_train, y_train, X_som_val, y_val)
            )

            # ----- MiniSom -----
            minisom = MiniSom(grid_size, grid_size, fixed_dim,
                              sigma=config.RADIUS,
                              learning_rate=config.LEARNING_RATE)
            minisom.random_weights_init(X_train)
            for _ in range(epoch):
                minisom.train_batch(X_train, len(X_train))
            X_min_val = np.array([minisom.winner(x) for x in X_val])
            minisom_scores.append(
                evaluate_classification_embedded(
                    np.array([minisom.winner(x) for x in X_train]),
                    y_train,
                    X_min_val,
                    y_val
                )
            )

        # Media y desviación estándar sobre folds
        somJ_means.append(np.mean(somJ_scores))
        somJ_stds.append(np.std(somJ_scores))
        minisom_means.append(np.mean(minisom_scores))
        minisom_stds.append(np.std(minisom_scores))

    return somJ_means, somJ_stds, minisom_means, minisom_stds


# ------------------------------------------------------------------------------------------------------------------------------
#   Script principal
# ------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    num_epochs  = [1, 5, 10, 15, 25, 30, 50]
    total_nodes = config.TOTAL_NODES
    datasets    = ["MNIST"]

    # Ejecutar experimentos
    results = {ds: run_experiment_kfold(ds, num_epochs, total_nodes)
               for ds in datasets}

    # Plot profesional con errores
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    plt.figure(figsize=(10, 7))

    for i, ds in enumerate(datasets):
        somJ_mean, somJ_std, mini_mean, mini_std = results[ds]
        color_light = adjust_color(base_colors[i], 1.3)
        color_dark  = adjust_color(base_colors[i], 0.7)

        plt.errorbar(
            num_epochs, somJ_mean, yerr=somJ_std,
            label=f"SoM (somJ) – {ds}",
            color=color_light, linestyle='-', marker='o', capsize=5
        )
        plt.errorbar(
            num_epochs, mini_mean, yerr=mini_std,
            label=f"MiniSom – {ds}",
            color=color_dark, linestyle='--', marker='s', capsize=5
        )

    plt.xlabel("Número de epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy medio con desviación estándar (5-Fold CV)", fontsize=14)
    plt.legend(loc="best", fontsize="small")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("experimentos/g_time_comp/acc_vs_epochs_MNIST_5fold.png", dpi=300)
    plt.show()