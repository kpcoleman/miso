from . nets import *
import torch
from torch import nn, optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import combinations


class deep_spectral(nn.Module):
    def __init__(self, features, adj, ind_views='all', combs='all', device='cpu'):
        super(deep_spectral, self).__init__()
        self.device = device
        self.num_views = len(features)
        self.features = [torch.Tensor(i).to(self.device) for i in features] 
        self.adj = [torch.Tensor(i).to(self.device) for i in adj]
        if ind_views=='all':
            self.ind_views = list(range(len(self.features)))
        else:
            self.ind_views = ind_views
        if combs=='all':
            self.combinations = list(combinations(list(range(len(features))),2))
        else:
            self.combinations = combs

    def train(self):
        self.mlps = [MLP(input_shape = self.features[i].shape[1], output_shape = 32).to(self.device) for i in range(len(self.features))]
        def sc_loss(A,Y):
            return (torch.triu(torch.cdist(Y,Y))*torch.triu(A)).mean()
        
        def mse_loss(X,X_hat):
            nn.MSELoss()

        for i in range(self.num_views):
            self.mlps[i].train()
            optimizer = optim.Adam(self.mlps[i].parameters(), lr=1e-3)          
            for epoch in range(1000):
                print(epoch)
                optimizer.zero_grad()
                x_hat = self.mlps[i](self.features[i])
                Y1 = self.mlps[i].get_embeddings(self.features[i])
                loss1 = nn.MSELoss()(self.features[i],x_hat)
                loss2 = sc_loss(self.adj[i], Y1)
                loss=loss1+loss2
                loss.backward()
                optimizer.step()

        [self.mlps[i].eval() for i in range(self.num_views)]
        Y = [self.mlps[i].get_embeddings(self.features[i]) for i in range(self.num_views)]
        interactions = [Y[i][:, :, None]*Y[j][:, None, :] for i,j in self.combinations]
        interactions = [i.reshape(i.shape[0],-1) for i in interactions]
        interactions = [torch.matmul(i,torch.pca_lowrank(i,q=32)[2]) for i in interactions]
        Y = [Y[i] for i in self.ind_views]
        Y = [StandardScaler().fit_transform(i.cpu().detach().numpy()) for i in Y]
        interactions = [StandardScaler().fit_transform(i.cpu().detach().numpy()) for i in interactions]
        Y = np.concatenate(Y,1)
        interactions = np.concatenate(interactions,1)
        emb = np.concatenate((Y,interactions),1)
        self.emb = emb
