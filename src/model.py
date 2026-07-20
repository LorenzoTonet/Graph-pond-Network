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





class MyPondNet(torch.nn.Module):
    def __init__(self,
                 feature_dimension: int, 
                 embedding_dim: int = 64,
                 l_hidden_dim: int = 128, 
                 max_n: int = 50):
        super().__init__()
        self.feature_dimension = feature_dimension
        self.embedding_dim = embedding_dim
        self.l_hidden_dim = l_hidden_dim
        self.max_n = max_n

        self.conv_layer_init = GCNConv(in_channels=feature_dimension, out_channels=embedding_dim)
        self.conv_layer_step = GCNConv(in_channels=embedding_dim, out_channels=embedding_dim)

        self.conv_to_vec = MLP(n_input= 2 * embedding_dim, n_hidden=l_hidden_dim, n_output= embedding_dim)

        self.classifier = torch.nn.Linear(in_features=embedding_dim, out_features=7)
        self.lambda_layer = torch.nn.Sigmoid(torch.nn.Linear(in_features=embedding_dim, out_features=1))

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p = 0.2)

    def forward(self, x, edge_list):
        n_nodes = x.shape[0]

        h_0 = torch.ones((n_nodes, self.embedding_dim))
        
        conv_e = self.conv_layer_init(x, edge_list)
        
        to_mlp = torch.cat([h_0, conv_e], dim=1)
        embedding = MLP(to_mlp)

        logits_0 = self.classifier(embedding)

        lambda_0 = self.lambda_layer(embedding)

        y_logits = [logits_0]
        lambdas = [lambda_0]
        h_s = h_0

        un_halted_prob = torch.ones((n_nodes, ))
        halted_nodes = torch.zeros((n_nodes, ))

        for i in range(1, self.max_n +1):
            if i == self.max_n:
                pass

        return y_logits, lambdas, h_s


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