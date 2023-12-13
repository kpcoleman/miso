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


def calculate_affinity(X1, sig=30):
	dist1 = pairwise_distances(X1)
	a1 = np.exp(-1*(dist1**2)/(2*(sig**2)))
	return a1

def update_adj(A, iter=100, k=15, device = 'cpu'):
    A = [torch.Tensor(i).to(device) for i in A]
    n = len(A)
    for i in range(iter):
        print(i)
        D = [i.sum(1) for i in A]
        L = [torch.diag(D[i]**(-0.5))@A[i]@torch.diag(D[i]**(-0.5)) for i in range(len(A))]
        u = [torch.svd(i, compute_uv=True)[0][:,:k] for i in L]
        for i in range(len(A)):
            for j in range(len(A)):
                if i!=j:
                    A[i] = u[j]@u[j].T@A[i]
                    A[i] = (A[i]+A[i].T)/2
    A = [(i-i.min())/i.max() for i in A]
    return A


