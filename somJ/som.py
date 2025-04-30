# Archivo: som/som.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from somJ.utils import check_input_data, validate_batch_size

class SoM:

    def __init__(self, method='random', input_dim=3, data=None, total_nodes=100):
        check_input_data(data)
        self.rand = np.random.RandomState(0)
        self.input_dim = input_dim
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.pca = PCA(n_components=2)

        train_data_scaled = self.scaler.fit_transform(data)
        self.pca.fit(train_data_scaled)

        l1 = np.sqrt(self.pca.explained_variance_[0])
        l2 = np.sqrt(self.pca.explained_variance_[1])
        ratio = l1 / l2
        n = int(np.round(np.sqrt(total_nodes / ratio)))
        m = int(np.round(ratio * n))
        self.grid_size = (m, n)
        print(f"Dimensiones del SOM: m = {m}, n = {n}")

        if method == 'pca':
            self.init_pca(train_data_scaled)
        else:
            self.init_random()

        self.map_history = [np.copy(self.som_map)]

    def init_random(self):
        rows, cols = self.grid_size
        self.som_map = self.rand.rand(rows, cols, self.input_dim)
        print("SOM inicializado aleatoriamente.")

    def init_pca(self, data):
        rows, cols = self.grid_size
        grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
        grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        projected_grid = grid @ self.pca.components_[:2, :]
        projected_grid += np.mean(data, axis=0)
        self.som_map = projected_grid.reshape(rows, cols, self.input_dim)
        self.som_map = np.clip(self.som_map, 0, 1)
        print("SOM inicializado usando PCA.")

    def find_winner(self, x):
        if x.shape[0] != self.input_dim:
            raise ValueError(f"Dimensión de entrada inválida: {x.shape[0]} vs {self.input_dim}")
        distSq = (np.square(self.som_map - x)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq), distSq.shape)

    def update_weights(self, train_ex, learn_rate, radius_sq, BMU_coord, step=3, save=False):
        g, h = BMU_coord
        map_h, map_w, _ = self.som_map.shape
        i_min, i_max = max(0, g - step), min(map_h, g + step + 1)
        j_min, j_max = max(0, h - step), min(map_w, h + step + 1)

        ii, jj = np.meshgrid(np.arange(i_min, i_max), np.arange(j_min, j_max), indexing='ij')
        dist_sq = (ii - g)**2 + (jj - h)**2
        influence = np.exp(-dist_sq / (2 * radius_sq)) * learn_rate
        influence = influence[..., np.newaxis]

        submap = self.som_map[i_min:i_max, j_min:j_max, :]
        self.som_map[i_min:i_max, j_min:j_max, :] += influence * (train_ex - submap)
        if save:
            self.map_history.append(np.copy(self.som_map))

    def update_weights_batchmap(self, train_data_scaled, radius_sq, step=3, save=False):
        map_h, map_w, dim = self.som_map.shape
        new_weights = np.zeros_like(self.som_map)
        weight_sums = np.zeros((map_h, map_w))

        for x in train_data_scaled:
            g, h = self.find_winner(x)
            i_min, i_max = max(0, g - step), min(map_h, g + step + 1)
            j_min, j_max = max(0, h - step), min(map_w, h + step + 1)

            ii, jj = np.meshgrid(np.arange(i_min, i_max), np.arange(j_min, j_max), indexing='ij')
            dist_sq = (ii - g)**2 + (jj - h)**2
            hci = np.exp(-dist_sq / (2 * radius_sq))

            new_weights[i_min:i_max, j_min:j_max, :] += hci[..., np.newaxis] * x
            weight_sums[i_min:i_max, j_min:j_max] += hci

        mask = weight_sums > 0
        self.som_map[mask] = new_weights[mask] / weight_sums[mask, np.newaxis]
        self.som_map = np.clip(self.som_map, 0, 1)
        if save:
            self.map_history.append(np.copy(self.som_map))

    def train(self, train_data, learn_rate=0.1, radius_sq=1, lr_decay=0.1, radius_decay=0.1, epochs=10,
              update='online', batch_size=None, step=3, save=False):

        check_input_data(train_data)
        if update == 'minibatch':
            validate_batch_size(batch_size)

        train_data_scaled = self.scaler.fit_transform(train_data)
        learn_rate_0, radius_0 = learn_rate, radius_sq

        for epoch in range(epochs):
            if update == 'batchmap':
                self.update_weights_batchmap(train_data_scaled, radius_sq, step, save)
            else:
                self.rand.shuffle(train_data_scaled)
                batches = [train_data_scaled] if batch_size is None else [train_data_scaled[i:i+batch_size] for i in range(0, len(train_data_scaled), batch_size)]
                for batch in batches:
                    for x in batch:
                        g, h = self.find_winner(x)
                        self.update_weights(x, learn_rate, radius_sq, (g, h), step, save)

            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)
            self.som_map = np.clip(self.som_map, 0, 1)

        filas, columnas, n_features = self.som_map.shape
        som_map_2d = self.som_map.reshape(-1, n_features)
        som_map_inversed = self.scaler.inverse_transform(som_map_2d)
        self.som_map = som_map_inversed.reshape(filas, columnas, n_features)
        if save:
            np.save('som_map_history.npy', np.array(self.map_history))
        return self.som_map

