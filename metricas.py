import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def correlation(X_original, X_reduced):
    D_orig = pairwise_distances(X_original)
    D_reduced = pairwise_distances(X_reduced)
    return np.corrcoef(D_orig.ravel(), D_reduced.ravel())[0, 1]

def evaluate(X_original, reduced_dict, n_clusters=3):
    """
    Evalúa varias técnicas de reducción de dimensionalidad.
    
    Parámetros:
    - X_original: np.ndarray con los datos originales.
    - reduced_dict: dict con nombre -> embedding reducido (ej: {"umap": X_umap, "tsne": X_tsne})
    - n_clusters: int, número de clusters a usar en KMeans (default=3)
    
    Retorna:
    - dict con métricas por técnica.
    """
    results = {}
    
    kmeans_original = KMeans(n_clusters=n_clusters, random_state=42).fit(X_original)
    
    for name, X_reduced in reduced_dict.items():
        trust = trustworthiness(X_original, X_reduced)
        dist_corr = correlation(X_original, X_reduced)
        
        kmeans_reduced = KMeans(n_clusters=n_clusters, random_state=42).fit(X_reduced)
        ari = adjusted_rand_score(kmeans_original.labels_, kmeans_reduced.labels_)
        
        results[name] = {
            "trustworthiness": trust,
            "distance_correlation": dist_corr,
            "adjusted_rand_index": ari
        }
    
    return results
