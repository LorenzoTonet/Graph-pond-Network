import torch

def oversmoothing_metric(X, adj):
    """
    Computes the normalized oversmoothing metric for a GNN layer.

    Parameters:
    - X: Tensor of shape (num_nodes, feature_dim), node embeddings.
    - adj: Sparse adjacency matrix (num_nodes, num_nodes).

    Returns:
    - normalized_metric: Normalized oversmoothing metric in [0, 1].
    """
    # Normalize the embeddings (cosine similarity requires normalization)
    X_norm = X / torch.norm(X, dim=1, keepdim=True)

    # Compute pairwise cosine similarities for neighbors
    cosine_sim = torch.mm(X_norm, X_norm.T)  # (num_nodes, num_nodes)
    
    # Element-wise 1 - cosine similarity
    dissimilarity = 1 - cosine_sim

    # Mask with the adjacency matrix to only consider neighbors
    neighbor_dissimilarity = dissimilarity * adj.to_dense()

    # Compute the unnormalized metric
    num_nodes = X.size(0)
    metric = neighbor_dissimilarity.sum() / num_nodes

    # Normalize by the maximum possible value (mu_max)
    mu_max = 2 * adj.sum() / num_nodes  # Maximum dissimilarity
    normalized_metric = metric / mu_max

    return normalized_metric