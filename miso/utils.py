import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import scipy
import scanpy as sc

def protein_norm(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)


def preprocess(adata,modality):
  adata.var_names_make_unique()
  if modality=='rna':
    sc.pp.filter_genes(adata,min_cells=10)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata,zero_center=True,max_value=6)
    return adata

  elif modality=='protein':
    adata.X = np.apply_along_axis(protein_norm, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X)))
    return adata     


def calculate_affinity(X1, sig=30, sparse = 'False', neighbors = 100):
  if not sparse:
    dist1 = pairwise_distances(X1)
    a1 = np.exp(-1*(dist1**2)/(2*(sig**2)))
    return a1
  else:
    dist1 = kneighbors_graph(X1, n_neighbors = neighbors, mode='distance')
    dist1.data = np.exp(-1*(dist1.data**2)/(2*(sig**2)))
    dist1.eliminate_zeros()
    return dist1

