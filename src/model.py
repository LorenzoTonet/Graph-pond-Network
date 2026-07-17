import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.baseline import MLP

class GCPondNet(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        embedding_dim: int,
        n_classes: int,
        dropout: float = 0.5,
        max_steps: int = 5,
        n_hidden_lin: int = 64,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_steps = max_steps

        # un GCNConv per step di pondering, invece di num_layers fissi a priori
        self.conv_in = GCNConv(in_dim, embedding_dim)
        self.conv_step = GCNConv(embedding_dim, embedding_dim)  # riusato ad ogni hop

        self.mlp = MLP(n_input=2 * embedding_dim, n_hidden=n_hidden_lin, n_output=embedding_dim)
        self.classifier = torch.nn.Linear(embedding_dim, n_classes)
        self.lambda_layer = torch.nn.Linear(embedding_dim, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]

        # hop 0: embedding iniziale
        embedding = F.relu(self.conv_in(x, edge_index))
        embedding = self.dropout(embedding)

        h = x.new_zeros((num_nodes, self.embedding_dim))
        concat = torch.cat([embedding, h], 1)
        h = self.mlp(concat)

        p, y = [], []
        un_halted_prob = h.new_ones((num_nodes,))
        halting_step = h.new_zeros((num_nodes,), dtype=torch.long)

        for n in range(1, self.max_steps + 1):
            if n == self.max_steps:
                lambda_n = h.new_ones(num_nodes)
            else: 
                torch.sigmoid(self.lambda_layer(h)).squeeze(-1)

            y_n = self.classifier(h)
            p_n = un_halted_prob * lambda_n
            p.append(p_n)
            y.append(y_n)

            newly_halted = (halting_step == 0) * torch.bernoulli(lambda_n).to(torch.bool)
            halting_step = torch.maximum(n * newly_halted.to(torch.long), halting_step)

            is_halted = is_halted | newly_halted  # <-- accumula chi si è fermato

            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # message passing su TUTTO il grafo (serve per i vicini attivi)
            new_embedding = F.relu(self.conv_step(embedding, edge_index))
            new_embedding = self.dropout(new_embedding)

            # ma i nodi già fermati NON aggiornano il proprio embedding:
            # mantengono il valore congelato, mentre i non-fermati prendono il nuovo
            embedding = torch.where(is_halted.unsqueeze(1), embedding, new_embedding)

            concat = torch.cat([embedding, h], 1)
            new_h = self.mlp(concat)
            h = torch.where(is_halted.unsqueeze(1), h, new_h)  # congela anche h

            if not self.training and is_halted.all():
                break

        return torch.stack(y), torch.stack(p), halting_step