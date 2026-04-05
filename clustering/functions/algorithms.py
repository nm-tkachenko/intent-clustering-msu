'''
This file contains all the functions that perform clustering algorithms on a list of embeddings.
'''
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, Birch

def apply_DBSCAN(embeddings, eps, min_samples):
  clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
  return clustering.labels_

def apply_HDBSCAN(embeddings, min_cluster_size, cluster_selection_epsilon): # not used due to environment conflicts
  hdb = HDBSCAN(copy=True, min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon, allow_single_cluster=True)
  hdb.fit(embeddings)
  return hdb.labels_

def apply_OPTICS(embeddings, max_eps, min_samples):
  clustering = OPTICS(min_samples=min_samples, max_eps=max_eps).fit(embeddings)
  return clustering.labels_

def apply_BIRCH(embeddings, threshold, branching_factor):
  brc = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None)
  brc.fit(embeddings)
  clustering = brc.predict(embeddings)
  return clustering