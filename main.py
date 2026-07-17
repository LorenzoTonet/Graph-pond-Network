import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, degree
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv


INFO = True
DEBUG = True

if INFO: print(f"[INFO] All libraries imported successfully.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if INFO: print(f"[INFO] Device: {device}")




dataset = Planetoid(root="./data/Planetoid", name="Cora")
data = dataset[0]

if DEBUG: print(f"[DEBUG] dataset info: {dataset} | num_classes: {dataset.num_classes} | num_node_features: {dataset.num_node_features}")
if INFO: print(f"[INFO] Dataset loaded: {dataset.name} with {len(dataset)} graphs.")
