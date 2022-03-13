import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DeepSetHVC(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSetHVC, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(       # \phi
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(       # \eta
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))
                #nn.Sigmoid())
        self.dec1 = nn.Sequential(      # \rho
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))
                #nn.Sigmoid())
        #self.dec2 = nn.Softmax(dim=1)
        self.dec2 = nn.Sigmoid()

    def forward(self, X):       # X [bs(1), n, 3]
        X = self.enc(X)         # X [bs(1), n, 128]
        Y = X.sum(-2)           # Y [bs(1), 128]
        X = self.dec(X)         # X [bs(1), n, 1]
        Y = self.dec1(Y)        # Y [bs(1), 1]
        X = X+Y                 # X [bs(1), n, 1]
        X = self.dec2(X)        # X [bs(1), n, 1]
        #X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

    def forward_allow_nan(self, X):     # X [bs, 100, 3]
        X = self.enc(X)
        X = torch.where(torch.isnan(X), torch.zeros(1, 1).to(self.device), X)      # all nan becomes 0
        Y = X.sum(-2)
        X = self.dec(X)
        Y = self.dec1(Y)
        X = X+Y
        X = self.dec2(X)
        return X


class DeepSet(nn.Module):
    def __init__(self, device, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.device = device
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output),
                nn.Sigmoid())

    def forward(self, X):
        X = self.enc(X).sum(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

    def forward_allow_nan(self, X):     # X [bs, 100, 3]
        X = self.enc(X)
        X = torch.where(torch.isnan(X), torch.zeros(1, 1).to(self.device), X)      # all nan becomes 0
        X = X.sum(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X
