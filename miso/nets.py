import torch.nn as nn
import torch
from torch.nn.utils.parametrizations import orthogonal

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(in_features=kwargs["input_shape"], out_features=128)
        self.decoder = nn.Linear(in_features=128, out_features=kwargs["input_shape"])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        x = self.decoder(x)
        x = self.relu(x)
        return x

    def get_embeddings(self,x):
        x = self.encoder(x)
        x = self.relu(x)
        return x
        
class MLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer1 = nn.Linear(kwargs["input_shape"], 32)
        #self.layer2 = nn.Linear(64, kwargs["output_shape"])
        self.layer3 = orthogonal(nn.Linear(kwargs["output_shape"], kwargs["output_shape"])) 
        self.layer4 = nn.Linear(32,kwargs["input_shape"])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layer1(x)
        #x = self.tanh(x)
        #x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.relu(x)
        #x1 = x.T@x + torch.eye(x.shape[1]).to(x.device)*1e-7                             
        #l = torch.cholesky(x1)
        #x = x@((x.shape[0])**(1/2)*l.inverse().T)
        #x, _ = torch.qr(x)
        return x

    def get_embeddings(self,x):
        x = self.layer3(self.layer1(x))
        return x

class MLP1(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer1 = nn.Linear(32, kwargs["output_shape"])
        self.layer2 = orthogonal(nn.Linear(kwargs["output_shape"], kwargs["output_shape"]))
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layer1(x)
        #x = self.relu(x)
        x = self.layer2(x)
        #x = self.relu(x)
        #x1 = x.T@x + torch.eye(x.shape[1]).to(x.device)*1e-7
        #l = torch.cholesky(x1)
        #x = x@((x.shape[0])**(1/2)*l.inverse().T)
        #x, _ = torch.qr(x)
        return x

