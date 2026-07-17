import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.utils import to_undirected, degree
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv
import pickle as pkl

from src.model import GCPondNet
from src.baseline import GCNet_baseline
from src.plots import *

INFO = True
DEBUG = True

if INFO: print(f"[INFO] All libraries imported successfully.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if INFO: print(f"[INFO] Device: {device}")

dataset = Planetoid(root="./data/Planetoid", name="Cora")
data = dataset[0]

if DEBUG: print(f"[DEBUG] dataset info: {dataset} | num_classes: {dataset.num_classes} | num_node_features: {dataset.num_node_features}")
if INFO: print(f"[INFO] Dataset loaded: {dataset.name} with {len(dataset)} graphs.")


model = GCPondNet(
    in_dim=dataset.num_node_features,
    embedding_dim=12,
    n_classes=dataset.num_classes,
    dropout=0.3,
    max_steps=30,
    n_hidden_lin=64,
).to(device)

baseline_model = GCNet_baseline(
    in_dim=dataset.num_node_features,
    hidden_dim=64,
    out_dim=dataset.num_classes,
    num_layers=4,
    dropout=0.5,
).to(device)

if INFO: print(f"[INFO] Models instanciated successfully.")

y, p, step, emb = model(data.x.to(device), data.edge_index.to(device))

print(y.shape, p.shape, step.shape)
print(y[0].shape, p[0].shape)


def pondering_loss(p, y, labels, step, mask, beta=0.01, prior_lambda=1/50, eps=1e-10):
    device = p.device
    n_steps = p.shape[0]

    # applica la maschera SOLO sulla dimensione dei nodi
    p_masked = p[:, mask]          # [n_steps, num_train_nodes]
    y_masked = y[:, mask, :]       # [n_steps, num_train_nodes, n_classes]
    labels_masked = labels[mask]   # [num_train_nodes]
    num_nodes = p_masked.shape[1]

    ce_per_step = torch.stack([
        F.cross_entropy(y_masked[n], labels_masked, reduction='none')
        for n in range(n_steps)
    ])

    rec_loss = (p_masked * ce_per_step).sum(dim=0).mean()

    prior = torch.tensor(
        [prior_lambda * (1 - prior_lambda) ** n for n in range(n_steps)],
        device=device
    )
    prior = prior / prior.sum()
    prior_expanded = prior.unsqueeze(1).expand(n_steps, num_nodes)

    kl_reg = F.kl_div(
        torch.log(prior_expanded + eps),
        p_masked + eps,
        reduction='none'
    ).sum(dim=0).mean()

    total_loss = rec_loss + beta * kl_reg
    return total_loss, rec_loss, kl_reg

def compute_pondering_accuracy(y, halting_step, labels, mask=None):
    """
    Accuracy per il pondering model: per ogni nodo, usa la predizione
    prodotta ALLO STEP in cui quel nodo ha effettivamente haltato,
    non l'ultimo step in assoluto.

    Parameters
    ----------
    y : torch.Tensor
        Shape [n_steps, num_nodes, n_classes] (output di model())
    halting_step : torch.Tensor
        Shape [num_nodes], halting step per nodo (1-indexed, come nel tuo codice)
    labels : torch.Tensor
        Shape [num_nodes]
    mask : torch.Tensor, optional
        Maschera booleana [num_nodes]
    """
    n_steps, num_nodes, n_classes = y.shape

    # halting_step è 1-indexed (n va da 1 a max_steps), y è 0-indexed (y[0] = step 1)
    step_idx = (halting_step.long() - 1).clamp(min=0, max=n_steps - 1)

    # seleziona, per ogni nodo, il logit prodotto al proprio halting step
    node_idx = torch.arange(num_nodes, device=y.device)
    y_at_halt = y[step_idx, node_idx]  # shape [num_nodes, n_classes]

    return compute_accuracy(y_at_halt, labels, mask)

def compute_accuracy(logits, labels, mask=None):
    """
    Accuracy standard per un output di classificazione singolo.

    Parameters
    ----------
    logits : torch.Tensor
        Shape [num_nodes, n_classes]
    labels : torch.Tensor
        Shape [num_nodes]
    mask : torch.Tensor, optional
        Maschera booleana [num_nodes] (es. data.val_mask, data.test_mask).
        Se None, calcola su tutti i nodi.
    """
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]

    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.shape[0]
    return correct / total

n_epochs = 100

train_loss_list = []
val_loss_list = []
baseline_train_loss_list = []
baseline_val_loss_list = []
accuracy_list = []
baseline_accuracy_list = []

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
baseline_optimizer = Adam(baseline_model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()
baseline_model.train()

for i in range(n_epochs):
    optimizer.zero_grad()
    baseline_optimizer.zero_grad()
    y, p, step, emb = model(data.x.to(device), data.edge_index.to(device))
    baseline_y = baseline_model(data.x.to(device), data.edge_index.to(device))

    loss, rec_loss, kl_reg = pondering_loss(p, y, data.y.to(device), step, data.train_mask.to(device), beta=0.1, prior_lambda=1/10, eps=1e-4)
    baseline_loss = F.cross_entropy(baseline_y[data.train_mask], data.y[data.train_mask].to(device))

    accuracy = compute_pondering_accuracy(y, step, data.y.to(device), data.train_mask.to(device))
    baseline_accuracy = compute_accuracy(baseline_y, data.y.to(device), data.train_mask.to(device))

    loss.backward()
    baseline_loss.backward()

    optimizer.step()
    baseline_optimizer.step()

    if INFO:
        print(f"[INFO] Epoch {i+1}/{n_epochs} | Pondering Loss: {loss.item():.4f} | Rec Loss: {rec_loss.item():.4f} | KL Reg: {kl_reg.item():.4f}")
        print(f"[INFO] Epoch {i+1}/{n_epochs} | Baseline Loss: {baseline_loss.item():.4f}")
    
    if i%10 == 0:
        model.eval()
        baseline_model.eval()
        with torch.no_grad():
            y, p, step, emb = model(data.x.to(device), data.edge_index.to(device))
            baseline_y = baseline_model(data.x.to(device), data.edge_index.to(device))

            val_loss, val_rec_loss, val_kl_reg = pondering_loss(p, y, data.y.to(device), step, data.val_mask.to(device))
            baseline_val_loss = F.cross_entropy(baseline_y[data.val_mask], data.y[data.val_mask].to(device))


            if INFO:
                print(f"[INFO] Validation | Pondering Loss: {val_loss.item():.4f} | Rec Loss: {val_rec_loss.item():.4f} | KL Reg: {val_kl_reg.item():.4f}")
                print(f"[INFO] Validation | Baseline Loss: {baseline_val_loss.item():.4f}")

    
    val_loss_list.append(val_loss.item())
    train_loss_list.append(loss.item())
    baseline_val_loss_list.append(baseline_val_loss.item())
    baseline_train_loss_list.append(baseline_loss.item())
    baseline_accuracy_list.append(baseline_accuracy)
    accuracy_list.append(accuracy)


compute_pondering_accuracy(y, step, data.y.to(device), data.test_mask.to(device))

plt.figure(figsize=(12, 6))
plt.plot(train_loss_list, label='Train Loss', color='blue')
plt.plot(val_loss_list, label='Validation Loss', color='orange')
plt.plot(baseline_train_loss_list, label='Baseline Train Loss', color='green')
plt.plot(baseline_val_loss_list, label='Baseline Validation Loss', color='red')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.figure(figsize=(12, 6))
plt.plot(accuracy_list, label='Pondering Model Accuracy', color='blue')
plt.plot(baseline_accuracy_list, label='Baseline Model Accuracy', color='orange')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

y, p, halting_step, emb = model(data.x.to(device), data.edge_index.to(device))


with open("pd_df_graph_topological_metrics.pkl", "rb") as file:
    df = pkl.load(file)

plot_halting_distribution(df)
plot_graph_with_halting(data, halting_step)
plot_halting_correlation_grid(df)
plot_correlation_heatmap(df)

plt.show()