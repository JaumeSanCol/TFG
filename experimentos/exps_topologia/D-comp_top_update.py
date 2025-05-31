import os
import gc
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import KFold

from sklearn.manifold import trustworthiness
from somJ.som import SoM
from somJ.functions import *
from sklearn.preprocessing import MinMaxScaler
import somJ.config as config

def evaluate_embedding(X, X_embedded, n_neighbors=5):
    
    trust = trustworthiness(X, X_embedded, n_neighbors=n_neighbors)
    cont = compute_continuity(X, X_embedded, n_neighbors=n_neighbors)
    return trust, cont



if __name__ == '__main__':
    datasets = ['Iris', 'Digits', 'MNIST', 'Fashion MNIST']
    inits = ["random", "pca"]
    updates = ["online", "batchmap", "minibatch"]
    n_splits = 5

    # Límite del hardware
    max_samples=30000
    max_samples_c=20000

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []

    for ds in datasets:
        print(f"\n=== Procesando {ds} ===")
        X, y = load_dataset(ds)
        X= MinMaxScaler().fit_transform(X)
        for ini, update in product(inits, updates):
            trust_scores, cont_scores, C_scores = [], [], []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
                X_train ,X_test= X[train_idx],X[test_idx]
                som = SoM(method=ini,
                        data=X_train,
                        total_nodes=config.TOTAL_NODES)
                som.train(train_data=X_train,
                        learn_rate=config.LEARNING_RATE,
                        update=update,
                        batch_size=int(len(X_train)/20))
                
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
                

                trust_scores.append(trust)
                cont_scores.append(cont)
                C_scores.append(C_val)

                print(f"  Fold {fold}: trust={trust:.4f}, cont={cont:.4f}, C={C_val:.4f}")

                del som
                gc.collect()

            # medias y desviaciones
            mean_trust = np.mean(trust_scores)
            std_trust  = np.std(trust_scores, ddof=1)
            mean_cont  = np.mean(cont_scores)
            std_cont   = np.std(cont_scores, ddof=1)
            mean_C     = np.mean(C_scores)
            std_C      = np.std(C_scores, ddof=1)

            print(f">>> {ds} | {ini} | {update}"
                f" | mean_trust={mean_trust:.4f}±{std_trust:.4f}"
                f" | mean_cont={mean_cont:.4f}±{std_cont:.4f}"
                f" | mean_C={mean_C:.4f}±{std_C:.4f}")

            results.append({
                'dataset': ds,
                'init': ini,
                'update': update,
                'mean_trustworthiness': mean_trust,
                'std_trustworthiness': std_trust,
                'mean_continuity': mean_cont,
                'std_continuity': std_cont,
                'mean_C': mean_C,
                'std_C': std_C
            })

    df = pd.DataFrame(results)
    output_path = 'experimentos/exps_topologia/csv/resultados_topologia_update.csv'
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\nResultados guardados en: {output_path}")
