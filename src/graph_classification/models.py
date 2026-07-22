import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

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

class GCNet_baseline_g_classification(torch.nn.Module):
    """Multi-layer Graph Convolutional Network for graph classification"""
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
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)

        output = self.classifier(x)

        return output, x

class GCPondNet_g_classification(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        embedding_dim: int,
        n_classes: int,
        dropout: float = 0.5,
        max_steps: int = 50,
        n_hidden_lin: int = 2,
        num_layers_step: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_steps = max_steps

        # Initial convolutional layer to transform input features to embedding dimension
        self.convs_in = GCNConv(in_dim, embedding_dim)

        # Some layers of GraphConv (this represent the step function)
        self.convs_step = torch.nn.ModuleList()

        for _ in range(num_layers_step):
            self.convs_step.append(GCNConv(embedding_dim, embedding_dim))
        
        # MLP that takes the output of Message passing layer and creates an embedding based on that
        self.mlp = MLP(n_input=2 * embedding_dim, n_hidden=n_hidden_lin, n_output=embedding_dim)

        #Linear layer that predicts the label embedding -> logits
        self.classifier = torch.nn.Linear(embedding_dim, n_classes)

        #Linear layer that predicts the CONDITIONAL halting probability lambda
        self.lambda_layer = torch.nn.Linear(embedding_dim, 1)

        #regularization
        self.dropout = torch.nn.Dropout(dropout)

    def _apply_conv_block(self, convs, x, edge_index):
        """Applica in sequenza una lista di GCNConv con ReLU + dropout tra un layer e l'altro."""
        for conv in convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return x

    def forward(self, x, edge_index, batch):
        num_graphs = int(batch.max().item()) + 1

        # message passing iniziale a livello di nodo
        embedding = self.convs_in(x, edge_index)

        # readout iniziale: pooling nodo -> grafo
        graph_embedding = global_mean_pool(embedding, batch)

        h = x.new_zeros((num_graphs, self.embedding_dim))
        concat = torch.cat([graph_embedding, h], 1)
        h = self.mlp(concat)

        p, y = [], []
        un_halted_prob = h.new_ones((num_graphs,))
        halting_step = h.new_zeros((num_graphs,), dtype=torch.float)
        is_halted = h.new_zeros((num_graphs,), dtype=torch.bool)

        for n in range(1, self.max_steps + 1):
            if n == self.max_steps:
                lambda_n = h.new_ones(num_graphs)
            else:
                lambda_n = torch.sigmoid(self.lambda_layer(h)).squeeze(-1)

            y_n = self.classifier(h)          # [num_graphs, n_classes]
            p_n = un_halted_prob * lambda_n   # [num_graphs]
            p.append(p_n)
            y.append(y_n)

            newly_halted = (halting_step == 0) * torch.bernoulli(lambda_n).to(torch.bool)
            halting_step = torch.maximum(n * newly_halted.to(torch.long), halting_step)
            is_halted = is_halted | newly_halted
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # step di message passing a livello di nodo
            new_embedding = self._apply_conv_block(self.convs_step, embedding, edge_index)

            # espandi la maschera "halted" da grafo a nodo, per congelare l'embedding
            # dei nodi appartenenti a grafi già fermi
            is_halted_nodes = is_halted[batch]  # [num_nodes]
            embedding = torch.where(is_halted_nodes.unsqueeze(1), embedding, new_embedding)

            new_graph_embedding = global_mean_pool(embedding, batch)

            concat = torch.cat([new_graph_embedding, h], 1)
            new_h = self.mlp(concat)
            h = torch.where(is_halted.unsqueeze(1), h, new_h)  # qui is_halted è già a livello di grafo, ok diretto

            if not self.training and (halting_step > 0).sum() == num_graphs:
                break

        last_embeddings = h.clone()
        return torch.stack(y), torch.stack(p), halting_step, last_embeddings
