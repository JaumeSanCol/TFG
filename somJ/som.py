# Archivo: som/som.py

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from somJ.utils import check_input_data, validate_batch_size

map_history = []
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
        #print(f"Dimensiones del SOM: m = {m}, n = {n}")

        if method == 'pca':
            self.init_pca(train_data_scaled)
        else:
            self.init_random()

        self.map_history = [np.copy(self.som_map)]

    def init_random(self):
        rows, cols = self.grid_size
        self.som_map =  self.rand.rand(rows, cols, self.input_dim)
        #print("SOM inicializado aleatoriamente.")

    def init_pca(self, data):
        rows, cols = self.grid_size
        grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
        grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        projected_grid = grid @ self.pca.components_[:2, :]
        projected_grid += np.mean(data, axis=0)

        self.som_map = projected_grid.reshape(rows, cols, self.input_dim)
        self.som_map =np.clip(self.som_map, 0, 1)
        #print("SOM inicializado con PCA.")


    def find_winner(self,x):
        distSq = (np.square(self.som_map - x)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)
        
    def update_weights(self, train_ex, learn_rate, radius_sq, BMU_coord, step=3,save=False):
        g, h = BMU_coord
        map_h, map_w, dim = self.som_map.shape

        i_min = max(0, g - step)
        i_max = min(map_h, g + step + 1)
        j_min = max(0, h - step)
        j_max = min(map_w, h + step + 1)

        # Coordenadas de los nodos vecinos
        ii, jj = np.meshgrid(np.arange(i_min, i_max), np.arange(j_min, j_max), indexing='ij')
        dist_sq = (ii - g)**2 + (jj - h)**2

        # Función de vecindad (gaussiana)
        influence = np.exp(-dist_sq / (2 * radius_sq)) * learn_rate
        influence = influence[..., np.newaxis]  # para broadcasting

        # Extraer submapa (vecinos del BMU)
        submap = self.som_map[i_min:i_max, j_min:j_max, :]

        # Actualización vectorizada
        self.som_map[i_min:i_max, j_min:j_max, :] += influence * (train_ex - submap)
        if save:map_history.append(np.copy(self.som_map))
    def update_weights_batchmap(self, train_data_scaled, radius_sq, step=3, save=False):
        map_h, map_w, dim = self.som_map.shape
        new_weights = np.zeros_like(self.som_map)
        weight_sums = np.zeros((map_h, map_w))

        for x in train_data_scaled:
            # Encontrar BMU para la entrada x (según mapa actual)
            g, h = self.find_winner(x)

            # Definir los vecinos (zona afectada por el BMU)
            i_min = max(0, g - step)
            i_max = min(map_h, g + step + 1)
            j_min = max(0, h - step)
            j_max = min(map_w, h + step + 1)

            # Coordenadas de la vecindad
            ii, jj = np.meshgrid(np.arange(i_min, i_max), np.arange(j_min, j_max), indexing='ij')
            dist_sq = (ii - g)**2 + (jj - h)**2

            # Función gaussiana de vecindad
            hci = np.exp(-dist_sq / (2 * radius_sq))

            # Acumular las actualizaciones ponderadas
            new_weights[i_min:i_max, j_min:j_max, :] += hci[..., np.newaxis] * x
            weight_sums[i_min:i_max, j_min:j_max] += hci

        # Finalizar: dividir suma ponderada entre suma de influencias
        mask = weight_sums > 0
        self.som_map[mask] = new_weights[mask] / weight_sums[mask, np.newaxis]
        self.som_map = np.clip(self.som_map, 0, 1)

        if save:
            map_history.append(np.copy(self.som_map))

    def update_weights_minibatch(self, batch_data, radius_sq, step=3,save=False):
        map_h, map_w, dim = self.som_map.shape
        new_weights = np.zeros_like(self.som_map)
        weight_sums = np.zeros((map_h, map_w))

        for x in batch_data:
            # Encontrar BMU para la entrada
            g, h = self.find_winner(x)

            # Definir los límites de la vecindad usando "step"
            i_min = max(0, g - step)
            i_max = min(map_h, g + step + 1)
            j_min = max(0, h - step)
            j_max = min(map_w, h + step + 1)

            # Crear la malla de coordenadas para la vecindad
            ii, jj = np.meshgrid(np.arange(i_min, i_max), np.arange(j_min, j_max), indexing='ij')
            dist_sq = (ii - g)**2 + (jj - h)**2

            # Calcular la influencia gaussiana para toda la vecindad
            hci = np.exp(-dist_sq / (2 * radius_sq))

            # Se actualiza la submatriz correspondiente en new_weights, 
            # multiplicando cada vector x por la influencia (broadcasting en la dimensión de características)
            new_weights[i_min:i_max, j_min:j_max, :] += hci[..., np.newaxis] * x
            # Actualizamos weight_sums de forma similar
            weight_sums[i_min:i_max, j_min:j_max] += hci
        # Actualizar el mapa: dividir la suma ponderada de entradas por la suma de influencias, nodo a nodo
        mask = weight_sums > 0
        self.som_map[mask] = new_weights[mask] / weight_sums[mask, np.newaxis]
        self.som_map =np.clip(self.som_map, 0, 1)
        if save:map_history.append(np.copy(self.som_map))



    def train(self, train_data, learn_rate=0.1, radius_sq=1, lr_decay=0.1, radius_decay=0.1, epochs=10,
            update='online',     # Método de actualización
            batch_size=None,     # Tamaño del batch
            step=3,              # Radio de modificación (None para calcularlo en base a radius_sq y k)
            save=False,
            n_jobs=False):         # Guardar los valores para una animación
        
        # 1. Escalado de los datos de entrada al rango [0, 1]
        train_data_scaled = self.scaler.fit_transform(train_data)
        
        learn_rate_0 = learn_rate
        radius_0 = radius_sq
        if n_jobs:epoch_bar = tqdm(range(epochs), desc="Entrenando SOM", ncols=100)

        for epoch in range(0,epochs):
            # Usar los datos escalados
            
            if update == 'batchmap':
                self.update_weights_batchmap(train_data_scaled, radius_sq, step=step, save=save)
            else:
                self.rand.shuffle(train_data_scaled)
                if update == 'minibatch':
                    if batch_size is None:
                        batches = [train_data_scaled]
                    else:
                        batches = [train_data_scaled[i:i + batch_size] for i in range(0, len(train_data_scaled), batch_size)]

                    for batch in batches:
                        self.update_weights_minibatch(batch, radius_sq, step=step,save=save)

                else:
                    for train_ex in train_data_scaled:
                        g, h = self.find_winner(train_ex)
                        self.update_weights(train_ex, learn_rate, radius_sq, (g, h), step=step,save=save)

            # Decaimiento de la tasa de aprendizaje y radio
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)

            self.som_map =np.clip(self.som_map, 0, 1)

            if n_jobs:epoch_bar.set_postfix(lr=learn_rate, radius=np.sqrt(radius_sq))
        
        # 2. Inversa del escalado en el mapa de pesos
        # Se asume que self.som_map tiene forma (filas, columnas, n_features)
        filas, columnas, n_features = self.som_map.shape
        som_map_2d = self.som_map.reshape(-1, n_features)
        som_map_inversed = self.scaler.inverse_transform(som_map_2d)
        self.som_map = som_map_inversed.reshape(filas, columnas, n_features)
        map_history_array = np.array(map_history)
        if save:
            
            np.save('som_map_history.npy', map_history_array)
        return self.som_map


    def predict(self,X_test,y_test):
        labels_map = {}
        for xi, label in zip(X_test,y_test):
            w = self.find_winner(xi)             # índice (i,j) de la BMU
            labels_map.setdefault(w, []).append(label)

        # toma la etiqueta más frecuente por nodo
        labels_map = {pos: np.bincount(lbls).argmax() 
                    for pos, lbls in labels_map.items()}

        # 6. Predicción y precisión
        y_pred = np.array([labels_map[self.find_winner(xi)] for xi in X_test])

        return y_pred

