from sklearn.datasets import load_iris, load_digits
from keras.datasets import mnist, fashion_mnist
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= -128 and c_max <= 127:
                    df[col] = df[col].astype('int8')
                elif c_min >= -32768 and c_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif c_min >= -2147483648 and c_max <= 2147483647:
                    df[col] = df[col].astype('int32')
                else:
                    df[col] = df[col].astype('int64')
            else:
                df[col] = df[col].astype('float32') 
    return df

def load_dataset(name):
    if name == 'Iris':
        data = load_iris()
        X, y = data.data, data.target

    elif name == 'Digits':
        data = load_digits()
        X, y = data.data, data.target

    elif name == 'MNIST':
        (X_train, y_train), _ = mnist.load_data()
        X = X_train.reshape((X_train.shape[0], -1))
        y = y_train

    elif name == 'Fashion MNIST':
        (X_train, y_train), _ = fashion_mnist.load_data()
        X = X_train.reshape((X_train.shape[0], -1))
        y = y_train

    else:
        raise ValueError(f"Dataset {name} no reconocido.")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = reduce_mem_usage(X)

    return X.values, y


def compute_continuity(X, X_embedded, n_neighbors=5):
    n_samples = X.shape[0]

    # Vecinos originales
    nn_orig = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    neigh_orig = nn_orig.kneighbors(X, return_distance=False)[:, 1:]

    # Vecinos en el espacio reducido
    nn_embed = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_embedded)
    neigh_embed = nn_embed.kneighbors(X_embedded, return_distance=False)[:, 1:]

    # Calculamos los rankings para todos los puntos
    nn_embed_all = NearestNeighbors(n_neighbors=n_samples).fit(X_embedded)
    all_embed_neighbors = nn_embed_all.kneighbors(X_embedded, return_distance=False)

    total = 0.0
    for i in range(n_samples):
        orig_neighbors = set(neigh_orig[i])
        emb_neighbors = set(neigh_embed[i])
        # Vecinos originales que NO están en los vecinos proyectados
        missing = orig_neighbors - emb_neighbors
        for j in missing:
            # Miramos cuál es la diferencia entre los rankings 
            rank = np.where(all_embed_neighbors[i] == j)[0][0] + 1
            total += (rank - n_neighbors)
    
    norm = n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1)
    continuity = 1 - (2 / norm) * total
    return continuity


def compute_som_continuity_quantization(X, M, n_rows, n_cols):
    # X: datos originales
    # M: prototipos (pesos) del SOM
    # n_rows, n_cols: dimensiones de la cuadrícula

    n_units = M.shape[0]

    # Creamos un grafo donde cada neurona se conecta con sus vecinas
    G = nx.Graph()
    for i in range(n_units):
        r, c = divmod(i, n_cols)
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            rr, cc = r+dr, c+dc
            if 0 <= rr < n_rows and 0 <= cc < n_cols:
                j = rr * n_cols + cc
                # El peso entre neuronas es la distancia euclidiana entre sus vectores de pesos
                G.add_edge(i, j, weight=np.linalg.norm(M[i] - M[j]))

    # Calculamos las distancias de cada muestra a cada neurona 
    dists = np.linalg.norm(X[:, None, :] - M[None, :, :], axis=2)
    # Para cada muestra, sacamos las dos neuronas más cercanas
    bmus = np.argsort(dists, axis=1)[:, :2]

    d_x = np.zeros(X.shape[0])
    for idx in range(X.shape[0]):
        bmu, second = bmus[idx]
        # Medimos el error de cuantización
        q_err = dists[idx, bmu]
        #Buscamos el camino más corto entre las dos neuronas ganadoras, aumulando distancias
        topo = nx.shortest_path_length(G, bmu, second, weight='weight')
        d_x[idx] = q_err + topo

    # Promediamos todo para tener el valor final de la continuidad cuantizada C
    return d_x.mean()
