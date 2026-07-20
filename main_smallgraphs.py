import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, TUDataset
from torch_geometric.utils import to_undirected, degree
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv
import pickle as pkl

from src.model import GCPondNet_g_classification
from src.baseline import GCNet_baseline_g_classification
from src.plots import *
from src.losses import pondering_loss
from src.evaluation import compute_pondering_accuracy, compute_accuracy


def pondering_loss_g_classification(logits, labels, p_n, beta, prior_lambda, eps=1e-10, direct_kl=True):
    max_steps, batch_size = p_n.shape

    rec_loss = 0.0
    for i in range(max_steps):
        ce = F.cross_entropy(logits[i], labels, reduction='none')
        rec_loss += (ce * p_n[i]).mean()

    steps = torch.arange(max_steps, device=p_n.device, dtype=p_n.dtype)
    log_prior = torch.log(torch.tensor(prior_lambda)) + steps * torch.log(torch.tensor(1 - prior_lambda))
    log_prior = torch.log_softmax(log_prior, dim=0)  # prior normalizzato in log-spazio

    log_p = torch.log(p_n + eps)

    reg_loss = 0.0
    for i in range(batch_size):
        if direct_kl:
            reg_loss += F.kl_div(log_p[:, i], log_prior, log_target=True, reduction='sum')
        else:
            reg_loss += F.kl_div(log_prior, log_p[:, i], log_target=True, reduction='sum')

    reg_loss = reg_loss / batch_size
    total_loss = rec_loss + beta * reg_loss
    return total_loss, rec_loss, reg_loss


INFO = True
DEBUG = True

if INFO: print(f"[INFO] All libraries imported successfully.")

n_epochs = 100

#Ponder model hyperparameters
embedding_dim = 16
max_steps = 8
num_layers_step = 1
dropout = 0.3
hidden_dim_lin = 64

#GCN baseline hyperparameters
baseline_hidden_dim = 32
baseline_num_layers = 2

# loss hyperparameters
beta = 0.1
prior_lambda = 1/5
eps = 1e-10

#optimizer hyperparameters
learning_rate = 0.001
weight_decay = 5e-4
gradient_clipping = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if INFO: print(f"[INFO] Device: {device}")

dataset = TUDataset(root="./data/TUDataset", name="ENZYMES", transform=None)

if DEBUG: print(f"[DEBUG] dataset info: {dataset} | num_classes: {dataset.num_classes} | num_node_features: {dataset.num_node_features}")
if INFO: print(f"[INFO] Dataset loaded: {dataset.name} with {len(dataset)} graphs.")
dataset = dataset.shuffle()

n = len(dataset)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)

train_dataset = dataset[:n_train]
val_dataset   = dataset[n_train:n_train + n_val]
test_dataset  = dataset[n_train + n_val:]

if INFO:
    print(f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

model = GCPondNet_g_classification(
    in_dim=dataset.num_node_features,
    embedding_dim=embedding_dim,
    n_classes=dataset.num_classes,
    dropout=dropout,
    max_steps=max_steps,
    n_hidden_lin=hidden_dim_lin,
).to(device)

baseline_model = GCNet_baseline_g_classification(
    in_dim=dataset.num_node_features,
    hidden_dim=baseline_hidden_dim,
    out_dim=dataset.num_classes,
    num_layers=baseline_num_layers,
    dropout=dropout,
).to(device)

if INFO: print(f"[INFO] Models instanciated successfully.")

train_loss_list = []
val_loss_list = []
accuracy_list = []

baseline_train_loss_list = []
baseline_val_loss_list = []
baseline_accuracy_list = []

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
baseline_optimizer = Adam(baseline_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

model.train()
baseline_model.train()

for i in range(n_epochs):
    for batch in train_loader:
        batch = batch.to(device)

        y, p, step, emb = model(batch.x, batch.edge_index, batch.batch)
        baseline_y, _ = baseline_model(batch.x, batch.edge_index, batch.batch)

        loss, rec_loss, kl_reg = pondering_loss_g_classification(logits=y, labels=batch.y, p_n = p, beta=beta, prior_lambda=prior_lambda, eps=eps)
        baseline_loss = F.cross_entropy(baseline_y, batch.y)

        #accuracy = compute_pondering_accuracy(y, step, batch.y)
        #baseline_accuracy = compute_accuracy(baseline_y, batch.y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = gradient_clipping)
        optimizer.step()

        
        baseline_optimizer.zero_grad()
        baseline_loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), max_norm = gradient_clipping)
        baseline_optimizer.step()

    if INFO:
        print(f"[INFO] Epoch {i+1}/{n_epochs} | Pondering Loss: {loss.item():.4f} | Rec Loss: {rec_loss.item():.4f} | KL Reg: {kl_reg.item():.4f}")
        print(f"[INFO] Epoch {i+1}/{n_epochs} | Baseline Loss: {baseline_loss.item():.4f}")
    
    if i%10 == 0:
        
        model.eval()
        baseline_model.eval()
        for batch in val_loader:
            with torch.no_grad():
                y, p, step, emb = model(batch.x, batch.edge_index, batch.batch)
                baseline_y, _ = baseline_model(batch.x, batch.edge_index, batch.batch)

                val_loss, val_rec_loss, val_kl_reg = pondering_loss_g_classification(logits=y, labels=batch.y, p_n = p, beta=beta, prior_lambda=prior_lambda, eps=eps)
                baseline_val_loss = F.cross_entropy(baseline_y, batch.y)


                if INFO:
                    print(f"[INFO] Validation | Pondering Loss: {val_loss.item():.4f} | Rec Loss: {val_rec_loss.item():.4f} | KL Reg: {val_kl_reg.item():.4f}")
                    print(f"[INFO] Validation | Baseline Loss: {baseline_val_loss.item():.4f}")
        
        model.train()
        baseline_model.train()

    val_loss_list.append(val_loss.item())
    train_loss_list.append(loss.item())
    baseline_val_loss_list.append(baseline_val_loss.item())
    baseline_train_loss_list.append(baseline_loss.item())
    #baseline_accuracy_list.append(baseline_accuracy)
    #accuracy_list.append(accuracy)

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

plt.show()
