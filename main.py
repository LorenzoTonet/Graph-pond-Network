import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from torch_geometric.utils import to_undirected, degree
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv
import pickle as pkl

from src.model import GCPondNet, GCPondNet_SoftHalting
from src.baseline import GCNet_baseline
from src.plots import *
from src.losses import pondering_loss
from src.evaluation import compute_pondering_accuracy, compute_accuracy

INFO = True
DEBUG = True

if INFO: print(f"[INFO] All libraries imported successfully.")

n_epochs = 100

#Ponder model hyperparameters
embedding_dim = 32
max_steps = 8
num_layers_step = 1
dropout = 0.01
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

dataset = Planetoid(root="./data/Planetoid", name="CiteSeer", transform=None)
data = dataset[0]

if DEBUG: print(f"[DEBUG] dataset info: {dataset} | num_classes: {dataset.num_classes} | num_node_features: {dataset.num_node_features}")
if INFO: print(f"[INFO] Dataset loaded: {dataset.name} with {len(dataset)} graphs.")


model = GCPondNet(
    in_dim=dataset.num_node_features,
    embedding_dim=embedding_dim,
    n_classes=dataset.num_classes,
    dropout=dropout,
    max_steps=max_steps,
    n_hidden_lin=hidden_dim_lin,
).to(device)

baseline_model = GCNet_baseline(
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

    y, p, step, emb = model(data.x.to(device), data.edge_index.to(device))
    baseline_y, _ = baseline_model(data.x.to(device), data.edge_index.to(device))


    loss, rec_loss, kl_reg = pondering_loss(p, y, data.y.to(device), data.train_mask.to(device), beta=beta, prior_lambda=prior_lambda, eps=eps, direct_kl=True)
    baseline_loss = F.cross_entropy(baseline_y[data.train_mask], data.y[data.train_mask].to(device))

    accuracy = compute_pondering_accuracy(y, step, data.y.to(device), data.train_mask.to(device))
    baseline_accuracy = compute_accuracy(baseline_y, data.y.to(device), data.train_mask.to(device))

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

        with torch.no_grad():
            y, p, step, emb = model(data.x.to(device), data.edge_index.to(device))
            baseline_y, _ = baseline_model(data.x.to(device), data.edge_index.to(device))

            val_loss, val_rec_loss, val_kl_reg = pondering_loss(p, y, data.y.to(device), data.val_mask.to(device), beta=beta, prior_lambda=prior_lambda, eps=eps, direct_kl=True)
            baseline_val_loss = F.cross_entropy(baseline_y[data.val_mask], data.y[data.val_mask].to(device))


            if INFO:
                print(f"[INFO] Validation | Pondering Loss: {val_loss.item():.4f} | Rec Loss: {val_rec_loss.item():.4f} | KL Reg: {val_kl_reg.item():.4f}")
                print(f"[INFO] Validation | Baseline Loss: {baseline_val_loss.item():.4f}")
        
        model.train()
        baseline_model.train()

    val_loss_list.append(val_loss.item())
    train_loss_list.append(loss.item())
    baseline_val_loss_list.append(baseline_val_loss.item())
    baseline_train_loss_list.append(baseline_loss.item())
    baseline_accuracy_list.append(baseline_accuracy)
    accuracy_list.append(accuracy)

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

model.eval()
baseline_model.eval()
y, p, halting_step, emb = model(data.x.to(device), data.edge_index.to(device))
_, base_emb = baseline_model(data.x.to(device), data.edge_index.to(device))

print(emb.shape)

print(f"Average halting step:  {torch.mean(halting_step)}")
print(y.shape)
print(p.shape)
print(halting_step.shape)
print(emb.shape)
print(p[:, 1])

# with open("pd_df_graph_topological_metrics.pkl", "rb") as file:
#     df = pkl.load(file)


from src.metrics import oversmoothing_metric

adj_torch_sparse = torch.sparse_coo_tensor(
    data.edge_index, 
    torch.ones(data.edge_index.shape[1]), 
    (data.num_nodes, data.num_nodes)
)


print(f"Oversmoothing metric for GraphPond: {oversmoothing_metric(emb, adj_torch_sparse)}")
print(f"Oversmoothing metric for base GNN: {oversmoothing_metric(base_emb, adj_torch_sparse)}")
df = compute_topological_metrics(halting_step=halting_step, data = data)
plot_halting_distribution(df)
plot_graph_with_halting(data, halting_step)
plot_halting_correlation_grid(df)
plot_correlation_heatmap(df)

plt.show()