import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from src.graph_classification.models import GCPondNet_g_classification, GCNet_baseline_g_classification
from src.graph_classification.loss import pondering_loss_g_classification
from src.graph_classification.evaluation import evaluate_baseline_accuracy, evaluate_pondering_accuracy

INFO = True
DEBUG = True

if INFO: print(f"[INFO] All libraries imported successfully.")

n_epochs = 1

#Ponder model hyperparameters
embedding_dim = 32
max_steps = 8
num_layers_step = 2
dropout = 0.3
hidden_dim_lin = 64

#GCN baseline hyperparameters
baseline_hidden_dim = 32
baseline_num_layers = 2

# loss hyperparameters
beta = 0.01
prior_lambda = 1/5
eps = 1e-10

#optimizer hyperparameters
learning_rate = 0.001
weight_decay = 5e-4
gradient_clipping = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if INFO: print(f"[INFO] Device: {device}")

dataset = TUDataset(root="./data/TUDataset", name="MUTAG", transform=None, use_node_attr=True)

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

avg_halting_step = []

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
baseline_optimizer = Adam(baseline_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def train_one_epoch_ponder(model, train_loader, optimizer, gradient_clipping, device = "cpu"):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)

        y, p, step, emb = model(batch.x, batch.edge_index, batch.batch)
        loss, rec_loss, kl_reg = pondering_loss_g_classification(logits=y, labels=batch.y, p_n = p, beta=beta, prior_lambda=prior_lambda, eps=eps, direct_kl=False)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = gradient_clipping)
        optimizer.step()

    return loss, rec_loss, kl_reg

def train_one_epoch_baseline(model, train_loader, optimizer, gradient_clipping, device = "cpu"):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)

        baseline_y, _ = model(batch.x, batch.edge_index, batch.batch)

        baseline_loss = F.cross_entropy(baseline_y, batch.y)
        
        optimizer.zero_grad()
        baseline_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = gradient_clipping)
        optimizer.step()
 
    return baseline_loss

for i in range(n_epochs):
    loss, rec_loss, kl_reg = train_one_epoch_ponder(model=model, train_loader=train_loader, optimizer=optimizer, gradient_clipping = gradient_clipping)
    train_accuracy, avg_step = evaluate_pondering_accuracy(model=model, loader=train_loader, device=device)

    baseline_loss = train_one_epoch_baseline(model= baseline_model, train_loader = train_loader, optimizer = baseline_optimizer,
                                                                 gradient_clipping=gradient_clipping, device = "cpu")
    baseline_train_accuracy = evaluate_baseline_accuracy(model=baseline_model, loader=train_loader, device=device)

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

                val_loss, val_rec_loss, val_kl_reg = pondering_loss_g_classification(logits=y, labels=batch.y, p_n = p, beta=beta, prior_lambda=prior_lambda, eps=eps,direct_kl=False)
                baseline_val_loss = F.cross_entropy(baseline_y, batch.y)

                if INFO:
                    print(f"[INFO] Validation | Pondering Loss: {val_loss.item():.4f} | Rec Loss: {val_rec_loss.item():.4f} | KL Reg: {val_kl_reg.item():.4f}")
                    print(f"[INFO] Validation | Baseline Loss: {baseline_val_loss.item():.4f}")

        model.train()
        baseline_model.train()

    
    avg_halting_step.append(avg_step)
    val_loss_list.append(val_loss.item())
    baseline_val_loss_list.append(baseline_val_loss.item())
    train_loss_list.append(loss.item())
    baseline_train_loss_list.append(baseline_loss.item())
    baseline_accuracy_list.append(baseline_train_accuracy)
    accuracy_list.append(train_accuracy)

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

graph = next(iter(test_loader))[1]

print(graph.x)
print(graph.edge_index)
print(graph.y)
G = to_networkx(graph)

layout = nx.kamada_kawai_layout(G)

nx.draw(G, pos = layout)

plt.show()
