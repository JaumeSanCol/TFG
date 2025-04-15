import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

class SoM:

    def __init__(self, m, n, method='random', input_dim=3, data=None, min_val=0, max_val=1,total_nodes=100):
        self.rand = np.random.RandomState(0)
        self.grid_size = (m, n)
        self.input_dim = input_dim

        # Calculamos las dimensiones del mapa a partir del pca de las dos atributos más significativos
        if method == 'pca' and data is not None:
            pca = PCA(n_components=2)
            pca.fit(data)
            l1 = np.sqrt(pca.explained_variance_[0])
            l2 = np.sqrt(pca.explained_variance_[1])
            ratio = l1 / l2
            n = int(np.round(np.sqrt(total_nodes / ratio)))
            m = int(np.round(ratio * n))
            self.grid_size = (m, n)
            print(f"Dimensiones del SOM calculadas a partir del PCA: m = {m}, n = {n}")
        else:
            if m is None or n is None:
                raise ValueError("Se debe proporcionar m y n o datos para el método 'pca'.")
            self.grid_size = (m, n)
        
        if method == 'pca':
            self.init_pca(data)
        else:
            self.init_random(min_val, max_val)

    def init_random(self,min_val,max_val):
        rows, cols = self.grid_size
        self.som_map = min_val + (max_val - min_val) * self.rand.rand(rows, cols, self.input_dim)
        print("SOM inicializado aleatoriamente.")

    def init_pca(self, data):
        rows, cols = self.grid_size

        pca = PCA(n_components=2)
        pca.fit(data)
        grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
        grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        projected_grid = grid @ pca.components_[:2, :]
        projected_grid += np.mean(data, axis=0)

        self.som_map = projected_grid.reshape(rows, cols, self.input_dim)
        print("SOM inicializado con PCA.")


    def find_winner(self,x):
        distSq = (np.square(self.som_map - x)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)
        
    def update_weights(self, train_ex, learn_rate, radius_sq, BMU_coord, step=3):
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
 

    def update_weights_batch(self, batch_data, radius_sq, step=3):
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


    def train(self, train_data, learn_rate=0.1, radius_sq=1, lr_decay=0.1, radius_decay=0.1, epochs=10,
                   update='online',     # Método de actualización
                   batch_size=None,     # Tamaño del batch
                   step=3):             # Radio de modificación (None para calcularlo en base a radius_sq y k)
      
        learn_rate_0 = learn_rate
        radius_0 = radius_sq
        epoch_bar = tqdm(range(epochs), desc="Entrenando SOM", ncols=100)

        for epoch in epoch_bar:
            self.rand.shuffle(train_data)

            if update == 'batch':
                if batch_size is None:
                    batches = [train_data]
                else:
                    batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]

                for batch in batches:
                    self.update_weights_batch(batch, radius_sq, step=step)

            else:
                for train_ex in train_data:
                    g, h = self.find_winner(train_ex)
                    self.update_weights(train_ex, learn_rate, radius_sq, (g, h), step=step)

            # Decaimiento de tasa de aprendizaje y radio
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)

            # Actualizar texto en la barra (opcional)
            epoch_bar.set_postfix(lr=learn_rate, radius=np.sqrt(radius_sq))

        return self.som_map 