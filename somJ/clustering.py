import numpy as np
from itertools import product, combinations

def som_clustering(som_map, num_clusters):
    m, n, d = som_map.shape
    nodes = [(i, j) for i in range(m) for j in range(n)]
    node_vectors = {(i, j): som_map[i, j] for i, j in nodes}

    # Paso 1: Calcular el centroide de cada nodo
    centroids = {(i, j): som_map[i, j] for i, j in nodes}

    # Paso 2: Asignar un grupo (inicialmente cada nodo es su propio grupo)
    group_map = {(i, j): idx for idx, (i, j) in enumerate(nodes)}
    groups = {idx: [(i, j)] for idx, (i, j) in enumerate(nodes)}
    group_centroids = {k: np.mean([node_vectors[node] for node in nodes], axis=0)
                       for k, nodes in groups.items()}

    def calculate_group_variance(group_nodes, centroid):
        return sum(np.linalg.norm(node_vectors[n] - centroid) ** 2 for n in group_nodes)

    # Paso 3: Calcular la varianza total
    def calculate_total_variance(groups, centroids):
        return sum(calculate_group_variance(groups[k], centroids[k]) for k in groups)

    V_total = calculate_total_variance(groups, group_centroids)

    # Paso 4: Fusionar grupos vecinos iterativamente
    def get_neighbors(i, j, radius=1):
        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                ni, nj = i + dx, j + dy
                if 0 <= ni < m and 0 <= nj < n and (dx != 0 or dy != 0):
                    neighbors.append((ni, nj))
        return neighbors

    while len(groups) > num_clusters:
        min_variance = float('inf')
        best_pair = None
        best_merged = None

        group_ids = list(groups.keys())
        checked_pairs = set()

        for p, q in combinations(group_ids, 2):
            if (p, q) in checked_pairs or (q, p) in checked_pairs:
                continue
            checked_pairs.add((p, q))

            nodes_p = groups[p]
            nodes_q = groups[q]

            # Determinar si p y q son vecinos (al menos un par de nodos debe ser vecino)
            neighbor_flag = any(n2 in get_neighbors(*n1) for n1 in nodes_p for n2 in nodes_q)
            if not neighbor_flag:
                continue

            merged_nodes = nodes_p + nodes_q
            centroid_merged = np.mean([node_vectors[n] for n in merged_nodes], axis=0)
            V_pq = calculate_group_variance(merged_nodes, centroid_merged)

            V_new_total = V_total + V_pq - calculate_group_variance(nodes_p, group_centroids[p]) - calculate_group_variance(nodes_q, group_centroids[q])

            if V_new_total < min_variance:
                min_variance = V_new_total
                best_pair = (p, q)
                best_merged = merged_nodes
                merged_centroid = centroid_merged

        if best_pair is None:
            break  # No se pueden fusionar más grupos vecinos

        # Fusionar grupos
        p, q = best_pair
        new_id = min(p, q)
        groups[new_id] = best_merged
        group_centroids[new_id] = merged_centroid
        del groups[max(p, q)]
        del group_centroids[max(p, q)]
        V_total = min_variance

    return groups, group_centroids
