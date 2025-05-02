from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)
import numpy as np

from somJ.som import *

def cross_val_som_metrics(X, y, k, method, update, config,random_state=42, scale=True):

    if scale:
        scaler = MinMaxScaler()
        X_proc = scaler.fit_transform(X)
    else:
        X_proc = X

    # 2. Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=k, shuffle=True, random_state=random_state
    )

    # listas para acumular cada métrica
    accs, precs, recs, f1s = [], [], [], []

    # 3. Bucle de pliegues
    for train_idx, test_idx in skf.split(X_proc, y):
        X_train, X_test = X_proc[train_idx], X_proc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 4. Inicializa y entrena un SOM nuevo
        som = SoM(
            method=method,
            input_dim=X_train.shape[1],
            data=X_train,
            total_nodes=config.TOTAL_NODES
        )
        som.train(
            train_data=X_train,
            learn_rate=config.LEARNING_RATE,
            radius_sq=config.RADIUS_SQ,
            lr_decay=config.LR_DECAY,
            radius_decay=config.RADIUS_DECAY,
            epochs=config.EPOCHS,
            update=update,
            batch_size=config.BATCH_SIZE,
            step=config.STEP,
            save=config.SAVE_HISTORY
        )

        # 5. Predicciones y métricas
        y_pred = som.predict(X_test, y_test)
        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, average='macro'))
        recs.append(recall_score(y_test, y_pred, average='macro'))
        f1s.append(f1_score(y_test, y_pred, average='macro'))

    # 6. Construye el diccionario de medias
    return {
        'accuracy':  float(np.mean(accs)),
        'precision': float(np.mean(precs)),
        'recall':    float(np.mean(recs)),
        'f1':        float(np.mean(f1s))
    }
