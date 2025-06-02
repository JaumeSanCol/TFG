import numpy as np
import time
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from somJ.utils import check_input_data, validate_batch_size


def decay_exp(valor_ini, step, total_steps,decay_rate):
    return valor_ini * np.exp(-step *decay_rate/ total_steps)

def decay_lin(valor_ini, step, total_steps):
    return valor_ini *(1-(step / total_steps))

def asymptotic_decay(valor_ini, t, max_iter):
    return valor_ini / (1 + t / (max_iter / 2))

class SoM:
    def __init__(self, method='pca', data=None, total_nodes=100):
        check_input_data(data)
        self.rand = np.random.RandomState(0)
        self.input_dim = data.shape[1]
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=2)
        self.map_history = []
        self.decay_func=None
        # Escalado y PCA
        train_scaled = self.scaler.fit_transform(data)
        self.pca.fit(train_scaled)

        # Dimensiones del mapa usando varianza de PCA
        l1 = np.sqrt(self.pca.explained_variance_[0])
        l2 = np.sqrt(self.pca.explained_variance_[1])
        ratio = l1 / l2
        n = int(np.round(np.sqrt(total_nodes / ratio)))
        m = int(np.round(ratio * n))
        self.grid_size = (m, n)

        # Definimos unas matrices del mismo tamaño que los ejes de coordenadas del mapa para guardar las distancias a las BMU durante los calculos
        rosi, cols = self.grid_size
        ii, jj = np.meshgrid(np.arange(rosi), np.arange(cols), indexing='ij')
        self.mat_dist_ii = ii
        self.mat_dist_jj = jj

        # Inicialización de pesos
        if method == 'pca':
            self._init_pca(train_scaled)
        else:
            self._init_random()

        # matrices para los sumatorios de batch
        self.new_weights = np.zeros(self.som_map.shape, dtype=float)
        self.suma_influencias = np.zeros(self.grid_size, dtype=float)

    def _init_random(self):
        rosi, cols = self.grid_size
        self.som_map = self.rand.rand(rosi, cols, self.input_dim)

    def _init_pca(self, data):
        rosi, cols = self.grid_size
        grid_x, grid_y = np.meshgrid(
            np.linspace(-1, 1, cols),
            np.linspace(-1, 1, rosi)
        )
        grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        projected = grid @ self.pca.components_[:2, :]
        projected += np.mean(data, axis=0)
        self.som_map = projected.reshape(rosi, cols, self.input_dim)

    def find_winner(self, x):
        dist_sq = np.sum((self.som_map - x) ** 2, axis=2) 
        return np.unravel_index(np.argmin(dist_sq, axis=None), dist_sq.shape)
    
    def find_all_winner(self, X):
        X_scaled=self.scaler.fit_transform(X)
        winner_list=[]
        for x in X_scaled:
            winner_list.append(self.find_winner(x))
        return winner_list

    def _decay(self,valor_ini, step, total_steps):
        if self.decay_func=="linear":
            return decay_lin(valor_ini, step, total_steps)
        elif self.decay_func=="asymptotic_decay":
            return asymptotic_decay(valor_ini, step, total_steps)
        elif self.decay_func=="exp":
            return decay_exp(valor_ini, step, total_steps,self.decay_rate)
        else:raise ValueError(f"Función de descomposición no reconocida:{self.decay_func}")
        
    def _update_weights(self, x, learn_rate, sigma_sq):
        # Encontrar la neurona ganadora
        g, h= self.find_winner(x)

        # Calcular la distancia al cuadrado entre la neurona ganadora y todas las neuronas del mapa (mat_dist)
        dist_sq = (self.mat_dist_ii - g) ** 2 + (self.mat_dist_jj - h) ** 2

        # Calcular la inflluencia (hci) mediante la vecindad gaussiana
        hci = np.exp(-dist_sq / (2 * sigma_sq))[..., np.newaxis] 

        # Sumar al mapa la la matriz de error multiplicado por la matriz de inflluecia multiplicada por el lr
        self.som_map += learn_rate* hci * (x - self.som_map)

    # BATCHMAP (TODOS al mismo tiempo)

    def _update_weights_batchmap(self, data_scaled, sigma_sq, learn_rate):
        # Matriz para guardar los valores de acumulados de las actualizaciones del batch
        nw = self.new_weights  # suma nuevos pesos: inflluecia * muestra
        si = self.suma_influencias  # suma de inflluecias: incluecias

        # Inicializamos a 0
        nw.fill(0.)
        si.fill(0.)

        for x in data_scaled:
            g, h = self.find_winner(x)
            # Calcular la distancia al cuadrado entre la neurona ganadora y todas las neuronas del mapa (mat_dist)
            dist_sq = (self.mat_dist_ii - g) ** 2 + (self.mat_dist_jj - h) ** 2
            # Calcular la inflluencia mediante la vecindad gaussiana  
            hci = np.exp(-dist_sq / (2 * sigma_sq))
            nw += hci[..., np.newaxis] * x
            si += hci
        # Elegimos solo lso valores que se ven modificados
        mask = si > 0
        new_map = nw[mask] / si[mask, None]
        self.som_map[mask] += learn_rate * (new_map - self.som_map[mask])

    # ------------------------------------------------------------------------------------------------------------------------------
    #  TRAIN 
    # ------------------------------------------------------------------------------------------------------------------------------
    def train(self, train_data,
            learn_rate=0.1,             # Learning rate inicial
            sigma=1,                    # Valor inicial de sigma para definir la influencia
            epochs=1,
            decay_func_name="exp",      # Función de descomposición: "linear" o "exp"
            decay_rate=1,               # Valor para escalar la descomposición exponencial, a mayor valor, mayor descomposición
            update='online',            # Método de actualización: "online", "bathmap", "minibatch"
            batch_size=None,            # Tamaño del minibatch
            save=False,                 # Guardar estado del mapa de pesos
            prog_bar=False):            # Mostrar barra de progreso

        # Escalar los datos de entrada
        data_scaled = self.scaler.fit_transform(train_data)

        # Guardar valores iniciales
        lr0, rad0 = learn_rate, sigma

        # Prepara decay por paso
        total_steps = epochs * len(data_scaled)
        step = 0

        # Asignamos la función de decay y el valor de tau
        self.decay_func=decay_func_name
        self.decay_rate=decay_rate

        # Barra de progreso
        if prog_bar:
            bar = tqdm(range(epochs), desc="Entrenando SOM", ncols=100)

        for epoch in range(epochs):
            if update == 'batchmap':
                # Actualizar los valores de learning rate y sigma²
                sigma_t_sq = (self._decay(rad0,step,total_steps))**2
                lr_t =self._decay(lr0,step,total_steps)

                self._update_weights_batchmap(data_scaled, sigma_t_sq, lr_t)

                if save:
                    self.map_history.append(self.re_scale())

                step += len(data_scaled)
            else:
                self.rand.shuffle(data_scaled)
                if update == 'minibatch':
                    batches = [data_scaled] if batch_size is None else [
                        data_scaled[i:i+batch_size]
                        for i in range(0, len(data_scaled), batch_size)
                    ]                                                                                           
                    for b in batches:
                        sigma_t_sq = (self._decay(rad0,step,total_steps))**2
                        lr_t = self._decay(lr0,step,total_steps)
                        self._update_weights_batchmap(b, sigma_t_sq, lr_t)
                        if save:
                            self.map_history.append(self.re_scale())
                        step += len(b)
                else:
                    for x in data_scaled:
                        sigma_t_sq = (self._decay(rad0,step,total_steps))**2
                        lr_t = self._decay(lr0,step,total_steps)
                        
                        self._update_weights(x, lr_t, sigma_t_sq)

                        if save:
                            self.map_history.append(self.re_scale())
                        step += 1

            if prog_bar:
                bar.set_postfix(lr=lr_t, sigma_sq=sigma_t_sq)
                bar.update(1)

        if prog_bar:
            bar.close()

    # ------------------------------------------------------------------------------------------------------------------------------
    #  Dado un grupo de muestras, devuleve un diccionario con las estiquetas de cada neurona
    # ------------------------------------------------------------------------------------------------------------------------------

    def neuron_labels(self, X, y):
        label_map = {}
        for coord, label in zip(X, y):
            coord = tuple(coord)
            label_map.setdefault(coord, []).append(label)
        neuron_labels = {coord: max(labels, key=labels.count)
                        for coord, labels in label_map.items()}
        return neuron_labels

    
    # ------------------------------------------------------------------------------------------------------------------------------
    # Devuelve el mapa del som con el rango de valores original
    # ------------------------------------------------------------------------------------------------------------------------------

    def re_scale(self):
        filas, columnas, n_features = self.som_map.shape
        som_map_2d = self.som_map.reshape(-1, n_features)
        som_map_inversed = self.scaler.inverse_transform(som_map_2d)
        map = som_map_inversed.reshape(filas, columnas, n_features)

        return map
    
    # ------------------------------------------------------------------------------------------------------------------------------
    # Rellena las etiquetas faltantes de las neuronas que no han sido activadas en la etiquetacion. Rellena mediante la moda de los vecinos
    # ------------------------------------------------------------------------------------------------------------------------------

    def fill_labels(self,neuron_labels, shape, missing_value=-1, max_iter=10):
        matrix = np.full(shape, missing_value)
        for (i, j), value in neuron_labels.items():
            matrix[i, j] = value

        def fill_with_neighbor_mode(matrix, missing_value=-1):
            filled = matrix.copy()
            rows, cols = filled.shape

            for i in range(rows):
                for j in range(cols):
                    if filled[i, j] == missing_value:
                        neighbors = []
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                ni, nj = i + dx, j + dy
                                if 0 <= ni < rows and 0 <= nj < cols and filled[ni, nj] != missing_value:
                                    neighbors.append(filled[ni, nj])
                        if neighbors:
                            counts = np.bincount(np.array(neighbors, dtype=int))
                            most_common = np.argmax(counts)
                            filled[i, j] = most_common
            return filled

        # Iterar hasta rellenar todos los huecos o max_iter
        filled = matrix.copy()
        for _ in range(max_iter):
            prev = filled.copy()
            filled = fill_with_neighbor_mode(filled, missing_value=missing_value)
            if np.array_equal(prev, filled):
                break

        # Convertir matriz resultante a diccionario con todas las coordenadas
        new_labels = {(i, j): int(filled[i, j]) for i in range(shape[0]) for j in range(shape[1])}
        return new_labels

    # ------------------------------------------------------------------------------------------------------------------------------
    # Devuelve el una lista de predicciones para X_predict. Utiliza X_label para asignar etiquetas al mapa
    # ------------------------------------------------------------------------------------------------------------------------------
    
    def predict(self,X_train,y_train,X_test):
        X_train_scaled=self.scaler.transform(X_train)
        X_test_scaled=self.scaler.transform(X_test)
        X_som_train = np.array([self.find_winner(x) for x in X_train_scaled])
        X_som_test  = np.array([self.find_winner(x) for x in X_test_scaled])
        neuron_labels=self.neuron_labels(X_som_train,y_train)
        # Rellenamos para asegurar que todas las neuronas tienen etiqueta
        neuron_labels_filled=self.fill_labels(neuron_labels, self.grid_size)
        # Predecir test
        y_pred = []
        for coord in X_som_test:
            coord = tuple(coord)
            if coord in neuron_labels_filled:
                y_pred.append(neuron_labels_filled[coord])
            else:
                # buscar neurona etiquetada más cercana
                dists = [ (coord[0]-c[0])**2 + (coord[1]-c[1])**2
                        for c in neuron_labels_filled.keys() ]
                nearest = list(neuron_labels_filled.keys())[np.argmin(dists)]
                y_pred.append(neuron_labels_filled[nearest])
        return np.array(y_pred)