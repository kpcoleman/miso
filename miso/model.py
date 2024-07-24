from . nets import *
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from . utils import calculate_affinity
import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.metrics.pairwise import euclidean_distances
from scanpy.external.tl import phenograph
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse import kron
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from PIL import Image
import scipy
from tqdm import tqdm

class Miso(nn.Module):
    def __init__(self, features, ind_views='all', combs='all', sparse=False, neighbors = None, device='cpu'):
        super(Miso, self).__init__()
        self.device = device
        self.num_views = len(features)
        self.features = [torch.Tensor(i).to(self.device) for i in features] 
        self.sparse = sparse
        features = [StandardScaler().fit_transform(i) for i in features]
        if neighbors is None and self.sparse:
            neighbors=100
            
        adj = [calculate_affinity(i, sparse = self.sparse, neighbors=neighbors) for i in features]
        self.adj1 = adj
        pcs = [PCA(128).fit_transform(i) if i.shape[1] > 128 else i for i in features]
        self.pcs = [torch.Tensor(i).to(self.device) for i in pcs] 
        if not self.sparse:
          self.adj = [torch.Tensor(i).to(self.device) for i in adj]
        else:
          adj = [coo_matrix(i) for i in adj]
          indices = [torch.LongTensor(np.vstack((i.row, i.col))) for i in adj]
          values = [torch.FloatTensor(i.data) for i in adj]
          shape = [torch.Size(i.shape) for i in adj]
          self.adj = [torch.sparse.FloatTensor(indices[i], values[i], shape[i]).to(self.device) for i in range(len(adj))]

        if ind_views=='all':
            self.ind_views = list(range(len(self.pcs)))
        else:
            self.ind_views = ind_views   
        if combs=='all':
            self.combinations = list(combinations(list(range(len(self.pcs))),2))
        else:
            self.combinations = combs        

    def train(self):
        self.mlps = [MLP(input_shape = self.features[i].shape[1], output_shape = 32).to(self.device) for i in range(len(self.features))]
        def sc_loss(A,Y):
            if not self.sparse:
              return (torch.triu(torch.cdist(Y,Y))*torch.triu(A)).mean()
            else:
              row = A.coalesce().indices()[0]
              col = A.coalesce().indices()[1]
              rows1 = Y[row]
              rows2 = Y[col]
              dist = torch.norm(rows1 - rows2, dim=1)
              return (dist*A.coalesce().values()).mean()

        def mse_loss(X,X_hat):
            nn.MSELoss()

        for i in range(self.num_views):
            print('Training network for modality ' + str(i+1))
            self.mlps[i].train()
            optimizer = optim.Adam(self.mlps[i].parameters(), lr=1e-3)          
            for epoch in tqdm(range(1000)):
                optimizer.zero_grad()
                x_hat = self.mlps[i](self.features[i])
                Y1 = self.mlps[i].get_embeddings(self.features[i])
                loss1 = nn.MSELoss()(self.features[i],x_hat)
                loss2 = sc_loss(self.adj[i], Y1)
                loss=loss1+loss2
                loss.backward()
                optimizer.step()

        [self.mlps[i].eval() for i in range(self.num_views)]
        Y = [self.mlps[i].get_embeddings(self.pcs[i]) for i in range(self.num_views)]
        if self.combinations is not None:
            interactions = [Y[i][:, :, None]*Y[j][:, None, :] for i,j in self.combinations]
            interactions = [i.reshape(i.shape[0],-1) for i in interactions]
            interactions = [torch.matmul(i,torch.pca_lowrank(i,q=32)[2]) for i in interactions]
        Y = [Y[i] for i in self.ind_views]
        Y = [StandardScaler().fit_transform(i.cpu().detach().numpy()) for i in Y]
        Y = np.concatenate(Y,1)
        if self.combinations is not None:
            interactions = [StandardScaler().fit_transform(i.cpu().detach().numpy()) for i in interactions]
            interactions = np.concatenate(interactions,1)
            emb = np.concatenate((Y,interactions),1)
        else:
            emb = Y
        self.emb = emb

    def cluster(self, n_clusters=10):
      clusters = KMeans(n_clusters, random_state = 100).fit_predict(self.emb)
      self.clusters = clusters
      return clusters
    
