import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from src.baseline import MLP

class GCPondNet(torch.nn.Module):
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

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]

        #initial conv
        embedding = self.convs_in(x, edge_index)

        h = x.new_zeros((num_nodes, self.embedding_dim))
        concat = torch.cat([embedding, h], 1)

        # The MLP takes both the output of convolutions and the current embeddings
        h = self.mlp(concat)

        p, y = [], []

        # probability of not being halted yet. at the beginning they are all ones, then it becomes (1-lambda1)(1-lambda2)....
        un_halted_prob = h.new_ones((num_nodes,))
        # step of halting of every node (0 = not halted)
        halting_step = h.new_zeros((num_nodes,), dtype=torch.float)
        # boolean mask for halted nodes
        is_halted = h.new_zeros((num_nodes,), dtype=torch.bool)

        # apply n times the step func
        for n in range(1, self.max_steps + 1):
            if n == self.max_steps:
                #if last step -> all conditional halting probabilities are 1
                lambda_n = h.new_ones(num_nodes)
            else:
                #evaluate that prob with the model
                lambda_n = torch.sigmoid(self.lambda_layer(h)).squeeze(-1)

            # predict labels based on current embeddings
            y_n = self.classifier(h)

            # marginal probability of halting: the probability of halting here times the probability of not having halted before (used in training)
            p_n = un_halted_prob * lambda_n

            p.append(p_n)
            y.append(y_n)

            # sample from a bernoulli of p = lambda_n (if 1 -> halt that node)
            newly_halted = (halting_step == 0) * torch.bernoulli(lambda_n).to(torch.bool)

            # update halting step for nodes that have just halted (if they haven't halted before)
            halting_step = torch.maximum(n * newly_halted.to(torch.long), halting_step)
            is_halted = is_halted | newly_halted

            # update the un_halted probabilities for all
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # apply step function
            new_embedding = self._apply_conv_block(self.convs_step, embedding, edge_index)

            # update embedding based on who has not halted
            embedding = torch.where(is_halted.unsqueeze(1), embedding, new_embedding)

            concat = torch.cat([embedding, h], 1)
            new_h = self.mlp(concat)
            h = torch.where(is_halted.unsqueeze(1), h, new_h)

            if not self.training and is_halted.all():
                break

        last_embeddings = h.clone()
        return torch.stack(y), torch.stack(p), halting_step, last_embeddings

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

            # step di message passing a livello di nodo (continua per tutto il batch)
            embedding = self._apply_conv_block(self.convs_step, embedding, edge_index)
            new_graph_embedding = global_mean_pool(embedding, batch)

            concat = torch.cat([new_graph_embedding, h], 1)
            new_h = self.mlp(concat)
            h = torch.where(is_halted.unsqueeze(1), h, new_h)  # freeze se il grafo ha già halted

            if not self.training and is_halted.all():
                break

        last_embeddings = h.clone()
        return torch.stack(y), torch.stack(p), halting_step, last_embeddings

class GCPondNet_SoftHalting(torch.nn.Module):
    """
    Variante 'soft halting': il message passing prosegue per TUTTI i nodi
    ad ogni step, senza congelamento. halting_step indica solo a quale step
    la rappresentazione di un nodo è considerata sufficientemente stabile
    per il readout — non corrisponde a un risparmio computazionale reale.

    Utile come baseline di confronto rispetto alla versione con freezing
    (GCPondNet), per capire quanto del segnale halting↔topologia dipende
    dal vero arresto del calcolo vs. dalla sola scelta di quando "leggere"
    l'output.
    """
    def __init__(
        self,
        in_dim: int,
        embedding_dim: int,
        n_classes: int,
        dropout: float = 0.5,
        max_steps: int = 5,
        n_hidden_lin: int = 64,
        num_layers_in: int = 2,
        num_layers_step: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_steps = max_steps

        self.convs_in = torch.nn.ModuleList()
        self.convs_in.append(GCNConv(in_dim, embedding_dim))
        for _ in range(num_layers_in - 1):
            self.convs_in.append(GCNConv(embedding_dim, embedding_dim))

        self.convs_step = torch.nn.ModuleList()
        for _ in range(num_layers_step):
            self.convs_step.append(GCNConv(embedding_dim, embedding_dim))

        self.mlp = MLP(n_input=2 * embedding_dim, n_hidden=n_hidden_lin, n_output=embedding_dim)
        self.classifier = torch.nn.Linear(embedding_dim, n_classes)
        self.lambda_layer = torch.nn.Linear(embedding_dim, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def _apply_conv_block(self, convs, x, edge_index):
        for conv in convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return x

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]

        embedding = self._apply_conv_block(self.convs_in, x, edge_index)

        h = x.new_zeros((num_nodes, self.embedding_dim))
        concat = torch.cat([embedding, h], 1)
        h = self.mlp(concat)

        p, y = [], []
        un_halted_prob = h.new_ones((num_nodes,))
        halting_step = h.new_zeros((num_nodes,), dtype=torch.float)

        for n in range(1, self.max_steps + 1):
            if n == self.max_steps:
                lambda_n = h.new_ones(num_nodes)
            else:
                lambda_n = torch.sigmoid(self.lambda_layer(h)).squeeze(-1)

            y_n = self.classifier(h)
            p_n = un_halted_prob * lambda_n
            p.append(p_n)
            y.append(y_n)

            # nessun freeze: aggiorna halting_step solo come "primo step in
            # cui il nodo avrebbe deciso di fermarsi", ma il calcolo prosegue
            newly_halted = (halting_step == 0) * torch.bernoulli(lambda_n).to(torch.bool)
            halting_step = torch.maximum(n * newly_halted.to(torch.long), halting_step)

            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # message passing SENZA maschera: tutti i nodi si aggiornano sempre
            embedding = self._apply_conv_block(self.convs_step, embedding, edge_index)

            concat = torch.cat([embedding, h], 1)
            h = self.mlp(concat)

            # niente early break basato su is_halted: qui ha senso interromperlo
            # solo se TUTTI i nodi hanno un halting_step assegnato, il calcolo
            # infatti prosegue comunque per tutti fino a quel punto
            if not self.training and (halting_step > 0).all():
                break

        last_embeddings = h.clone()
        return torch.stack(y), torch.stack(p), halting_step, last_embeddings