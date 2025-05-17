import numpy as np
import matplotlib.pyplot as plt
from somJ.som import SoM
from functions import load_dataset
import somJ.config as config
from minisom import MiniSom
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.colors as mcolors

# ------------------------------------------------------------------------------------------------------------------------------
#   Experimento para observar la accuracy respecto del numero de epochs usadas durante el entrenamiento
# ------------------------------------------------------------------------------------------------------------------------------
#
#   Los resultados son almacenados en una grafica en la carpeta g_time_comp
#
# ------------------------------------------------------------------------------------------------------------------------------

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
            dists = [(coord[0]-c[0])**2 + (coord[1]-c[1])**2
                     for c in neuron_labels.keys()]
            nearest = list(neuron_labels.keys())[np.argmin(dists)]
            y_pred.append(neuron_labels[nearest])
    return accuracy_score(y_test, np.array(y_pred))

def run_experiment(dataset_name, num_epochs, total_nodes):
    # Carga y normaliza
    print(f"Procesando {dataset_name}…")
    X, y = load_dataset(dataset_name)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,test_size=0.3, random_state=42)

    fixed_dim = X_train.shape[1]                # Dinámico para cada dataset
    grid_size = int(np.sqrt(total_nodes))

    somJ_accs, minisom_accs = [], []
  
    
    for epoch in num_epochs:
        print(epoch)
          # Entrena SoM (somJ)
        som = SoM(
            method=config.INIT_METHOD,
            data=X_train,
            total_nodes=total_nodes
        )

        som.train(
            train_data=X_train,
            learn_rate=config.LEARNING_RATE,
            radius_sq=config.RADIUS_SQ,
            lr_decay=config.LR_DECAY,
            radius_decay=config.RADIUS_DECAY,
            epochs=epoch,
            update=config.UPDATE_METHOD,
            batch_size=config.BATCH_SIZE,
            step=config.STEP,
            save=config.SAVE_HISTORY,
            prog_bar=False
        )
        X_som_train = np.array([som.find_winner(x) for x in X_train])
        X_som_test  = np.array([som.find_winner(x) for x in X_test])
        somJ_accs.append(evaluate_classification_embedded(
            X_som_train, y_train, X_som_test, y_test))
        
        # Entrena MiniSom
        minisom = MiniSom(grid_size, grid_size, fixed_dim,
                            sigma=config.RADIUS_SQ,
                            learning_rate=config.LEARNING_RATE)
        for e in range(0,epoch):
            minisom.train_batch(X_train, len(X_train))
        X_min_train = np.array([minisom.winner(x) for x in X_train])
        X_min_test  = np.array([minisom.winner(x) for x in X_test])
        minisom_accs.append(evaluate_classification_embedded(
            X_min_train, y_train, X_min_test, y_test))

    return somJ_accs, minisom_accs


def adjust_color(color, factor):
    """
    Multiplica cada canal RGB por 'factor' y lo recorta a [0,1].
    factor > 1 aclara; factor < 1 oscurece.
    """
    c = np.array(mcolors.to_rgb(color))
    c = np.clip(c * factor, 0, 1)
    return tuple(c)

# ... (tus funciones evaluate_classification_embedded y run_experiment quedan igual)

if __name__ == "__main__":
    num_epochs    = [1,10,20,30,40,50]
    total_nodes   = config.TOTAL_NODES
    datasets      = ["MNIST"]
    # datasets      = ["Iris", "Digits", "MNIST", "Fashion MNIST"]

    # Ejecuta experimentos
    results = {ds: run_experiment(ds, num_epochs, total_nodes)
               for ds in datasets}

    # 1) Paleta base para cada dataset
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    plt.figure(figsize=(10, 7))
    for i, ds in enumerate(datasets):
        somJ_accs, minisom_accs = results[ds]

        # 2) Genera versión clara y versión oscura
        color_light = adjust_color(base_colors[i], factor=1.3)  # linea continua
        color_dark  = adjust_color(base_colors[i], factor=0.7)  # linea discontinua

        # 3) Dibuja
        plt.plot(num_epochs, somJ_accs,
                 label=f"SoM (somJ) – {ds}",
                 color=color_light,
                 linestyle='-')
        plt.plot(num_epochs, minisom_accs,
                 label=f"MiniSom – {ds}",
                 color=color_dark,
                 linestyle='--')

    plt.xlabel("Número de epochs")
    plt.ylabel("Accuracy")
    plt.title("Comparación de accuracy vs epochs para distintos datasets")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experimentos/g_time_comp/acc_vs_epochs_MNIST_05.png", dpi=300)
    plt.show()