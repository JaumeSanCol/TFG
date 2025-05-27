import os
import gc
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import KFold
from sklearn.manifold import trustworthiness
from somJ.som import SoM
from functions import load_dataset
import somJ.config as config
from functions import *

def evaluate_embedding(X, X_embedded, n_neighbors=5):
    trust = trustworthiness(X, X_embedded, n_neighbors=n_neighbors)
    cont = compute_continuity(X, X_embedded, n_neighbors=n_neighbors)
    return trust, cont
datasets = [
            'Iris', 
            'Digits', 
            'MNIST', 'Fashion MNIST']
inits = ["random", "pca"]
updates = ["online", "batchmap", "minibatch"]
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
results = []
for ds in datasets:
    print(f"\n=== Procesando {ds} ===")
    X, y = load_dataset(ds)

    for ini, update in product(inits, updates):
        trust_scores, cont_scores, C_scores = [], [], []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]

            som = SoM(method=ini,
                      data=X_train,
                      total_nodes=config.TOTAL_NODES)
            som.train(train_data=X_train,
                      learn_rate=config.LEARNING_RATE,
                      update=update,
                      batch_size=int(len(X_train)/20))

            # 1) extraer dimensiones de la rejilla y codebook
            m, n = som.grid_size
            M = som.som_map.reshape(-1, som.input_dim)

            # 2) embedding para trust/continuity
            X_embed = np.array(som.find_all_winner(X_test))

            # 3) métricas
            trust = trustworthiness(X_test, X_embed, n_neighbors=5)
            cont  = compute_continuity(X_test, X_embed, n_neighbors=5)
            C_val = compute_som_continuity_quantization(X_test, M, m, n)

            trust_scores.append(trust)
            cont_scores.append(cont)
            C_scores.append(C_val)

            print(f"  Fold {fold}: trust={trust:.4f}, cont={cont:.4f}, C={C_val:.4f}")

            del som
            gc.collect()

        # medias
        mean_trust = np.mean(trust_scores)
        mean_cont  = np.mean(cont_scores)
        mean_C     = np.mean(C_scores)

        print(f">>> {ds} | {ini} | {update}"
              f" | mean_trust={mean_trust:.4f}"
              f" | mean_cont={mean_cont:.4f}"
              f" | mean_C={mean_C:.4f}")

        results.append({
            'dataset': ds,
            'init': ini,
            'update': update,
            'mean_trustworthiness': mean_trust,
            'mean_continuity': mean_cont,
            'mean_C': mean_C
        })

df = pd.DataFrame(results)
output_path = 'experimentos/resultados_update_goodness.csv'
df.to_csv(output_path, index=False, float_format='%.4f')
print(f"\nResultados guardados en: {output_path}")
