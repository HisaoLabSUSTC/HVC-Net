import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DeepSetHVC_old(nn.Module):
    def __init__(self, device, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSetHVC_old, self).__init__()
        self.device = device
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
        # self.dec2 = nn.Softmax(dim=1)
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


class DeepSetHVC(nn.Module):
    def __init__(self, device, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSetHVC, self).__init__()
        self.device = device
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.num_blocks = 3

        self.phi = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.LayerNorm(dim_hidden))
        self.etas = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.LayerNorm(dim_hidden))
            for _ in range(self.num_blocks)
        ])
        self.rho = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output))
            # nn.Sigmoid())
        self.activation = nn.Sigmoid()
        # self.activation = nn.Softmax(dim=-2)      # if use softmax, batch with different num of nan is not allowed.

    def forward(self, X, allow_nan=False):      # X [bs(1), n, 3]
        """
        using X_0, nan remains in X.
        since output can contain nan,
        if activation is Softmax(dim=-2), batch with diff num of nan is not allowed.
        :param X:
        :param allow_nan:
        :return:
        """
        X = self.phi(X)                         # X [bs(1), n, 128]
        num_valids = torch.sum(~torch.isnan(X), dim=-2)  # [bs(1), 128]
        # if allow_nan:
        #     X = torch.where(torch.isnan(X), torch.zeros(1, 1).to(self.device), X)  # all nan becomes 0
        for i in range(self.num_blocks):
            X_0 = torch.where(torch.isnan(X), torch.zeros(1, 1).to(self.device), X)  # all nan becomes 0
            Y = X_0.sum(-2)                     # Y [bs(1), 128]
            Y = Y / num_valids                  # X.mean(-2)
            Y = self.etas[i](Y)                 # Y [bs(1), 128]
            Y = torch.stack([Y for _ in range(X.shape[-2])], dim=-2)    # Y [bs(1), n, 128]
            X = X+Y                             # X [bs(1), n, 128]
        X = self.rho(X)                         # X [bs(1), n, 1]
        return self.activation(X)

    def forward_allow_nan(self, X):
        return self.forward(X, allow_nan=True)


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
