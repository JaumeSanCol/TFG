# Archivo: main.py

import numpy as np
from som.som import SoM
from som.utils import plot_map
import som.config as config

def main():
    # 1. Generar datos de ejemplo
    data = np.random.rand(500, 3)  # 500 puntos en 3 dimensiones (puedes cambiarlo)

    # 2. Crear instancia de SOM
    som = SoM(
        method='pca', 
        input_dim=data.shape[1], 
        data=data, 
        total_nodes=config.TOTAL_NODES
    )

    # 3. Entrenar el SOM
    som.train(
        train_data=data,
        learn_rate=config.LEARNING_RATE,
        radius_sq=config.RADIUS_SQ,
        lr_decay=config.LR_DECAY,
        radius_decay=config.RADIUS_DECAY,
        epochs=config.EPOCHS,
        update=config.UPDATE_METHOD,
        batch_size=config.BATCH_SIZE,
        step=config.STEP,
        save=config.SAVE_HISTORY
    )

    # 4. Visualizar el mapa final
    som.plot(title="Mapa final del Self-Organizing Map")
    # También puedes usar directamente la función utilitaria
    # plot_map(som.som_map, title="Mapa final usando plot_map")

if __name__ == "__main__":
    main()
