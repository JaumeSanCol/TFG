import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from somJ.som import SoM
from functions import load_dataset
import somJ.config as config
import gc
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

def compute_continuity(X, X_embedded, n_neighbors=5):
    n_samples = X.shape[0]

    nn_orig = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    neigh_orig = nn_orig.kneighbors(X, return_distance=False)[:, 1:]

    nn_embed = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_embedded)
    neigh_embed = nn_embed.kneighbors(X_embedded, return_distance=False)[:, 1:]

    ranks = np.zeros(n_samples)
    for i in range(n_samples):
        missing = set(neigh_orig[i]) - set(neigh_embed[i])
        ranks[i] = len(missing)

    continuity = 1 - (np.mean(ranks) / n_neighbors)
    return continuity

def evaluate_embedding(X, X_embedded, n_neighbors=5):
    trust = trustworthiness(X, X_embedded, n_neighbors=n_neighbors)
    cont = compute_continuity(X, X_embedded, n_neighbors=n_neighbors)
    return trust, cont

datasets = ['Iris', 'Digits'
             ]
total_nodes = config.TOTAL_NODES


inits = ["random", "pca"]
updates = ["online", "batchmap", "minibatch"]

for ds in datasets:
    print(f"\n=== Procesando {ds} ===")
    X, y = load_dataset(ds)

    for ini, update in product(inits, updates):
        som = SoM(
            method=ini,
            data=X,
            total_nodes=total_nodes
        )
        som.train(
            train_data=X,
            learn_rate=config.LEARNING_RATE,
            update=update,
            prog_bar=False
        )
        X_embed = np.array(som.find_all_winner(X))
        del som
        gc.collect()
        trust, cont = evaluate_embedding(X, X_embed, n_neighbors=5)
        print(f"Para {ds}: {ini} {update}; trustworthiness={trust:.4f}, continuity={cont:.4f}")
