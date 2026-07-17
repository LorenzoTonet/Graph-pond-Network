import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import GCNConv

class MLP(nn.Module):
    '''
        Simple 3-layer multi layer perceptron.

        Parameters
        ----------
        n_input : int
            Size of the input.

        n_hidden : int
            Number of units of the hidden layer.

        n_ouptut : int
            Size of the output.
    '''

    def __init__(self, n_input, n_hidden, n_output):
        super(MLP, self).__init__()
        self.i2h = nn.Linear(n_input, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        '''forward pass'''
        x = F.relu(self.i2h(x))
        x = self.droput(x)
        x = F.relu(self.h2o(x))
        return x


class GCNet_baseline(torch.nn.Module):
    """Multi-layer Graph Convolutional Network."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.classifier = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.dropout(x)

        embeddings = x
        output = self.classifier(x)

        return output
