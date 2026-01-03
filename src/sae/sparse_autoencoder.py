import torch
from torch.nn import Module, Linear, ReLU, SELU


class SparseAutoencoder(Module):

    def __init__(self, in_dim, h_dim, activation):
        super(SparseAutoencoder, self).__init__()

        self.encoder = Linear(in_dim, h_dim)
        self.decoder = Linear(h_dim, in_dim)
        self.activ = activation

    def forward(self, x):

        hidden = self.activ(self.encoder(x))
        decoded = self.decoder(hidden)

        return hidden, decoded
