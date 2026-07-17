import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from torch_geometric.utils import to_networkx


# ---------------------------------------------------------------------------
# 1. Calcolo delle metriche topologiche (una volta sola, poi riusate ovunque)
# ---------------------------------------------------------------------------

def compute_topological_metrics(data, halting_step):
    """
    Calcola le metriche topologiche per ogni nodo e le mette in un DataFrame
    insieme all'halting step.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Il grafo (serve data.edge_index, opzionalmente data.y per homophily)
    halting_step : torch.Tensor
        Halting step per nodo, shape [num_nodes]

    Returns
    -------
    pd.DataFrame con una riga per nodo
    """
    G = to_networkx(data, to_undirected=True)

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    clustering = nx.clustering(G)
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

    # local homophily: frazione di vicini con la stessa label del nodo
    local_homophily = {}
    if hasattr(data, 'y') and data.y is not None:
        labels = data.y.cpu().numpy()
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 0:
                local_homophily[node] = np.nan
            else:
                same = sum(labels[n] == labels[node] for n in neighbors)
                local_homophily[node] = same / len(neighbors)

    df = pd.DataFrame({
        'node': list(G.nodes()),
        'halting_step': halting_step.cpu().numpy(),
        'degree': [degree[n] for n in G.nodes()],
        'betweenness': [betweenness[n] for n in G.nodes()],
        'closeness': [closeness[n] for n in G.nodes()],
        'clustering': [clustering[n] for n in G.nodes()],
        'eigenvector': [eigenvector[n] for n in G.nodes()],
    })

    if local_homophily:
        df['local_homophily'] = [local_homophily[n] for n in G.nodes()]

    return df


# ---------------------------------------------------------------------------
# 2. Distribuzione degli halting step
# ---------------------------------------------------------------------------

def plot_halting_distribution(df, ax=None):
    """Istogramma di quanti nodi si fermano ad ogni step."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    max_step = int(df['halting_step'].max())
    sns.histplot(df['halting_step'], bins=range(1, max_step + 2),
                 discrete=True, ax=ax, color='steelblue')
    ax.set_xlabel('Halting step')
    ax.set_ylabel('Numero di nodi')
    ax.set_title('Distribuzione degli halting step')
    ax.set_xticks(range(1, max_step + 1))
    return ax


# ---------------------------------------------------------------------------
# 3. Halting step visualizzato sul grafo (color-coded)
# ---------------------------------------------------------------------------

def plot_graph_with_halting(data, halting_step, layout='spring', seed=42, ax=None):
    """
    Disegna il grafo colorando ogni nodo in base al suo halting step.
    Attenzione: su grafi grandi (>1-2k nodi) diventa illeggibile, usa
    eventualmente un sottografo (vedi plot_ego_subgraph sotto).
    """
    G = to_networkx(data, to_undirected=True)
    steps = halting_step.cpu().numpy()

    if layout == 'spring':
        pos = nx.spring_layout(G, seed=seed)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError(f"layout '{layout}' non supportato")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=steps, cmap='viridis',
        node_size=40, ax=ax
    )
    nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax, width=0.5)
    plt.colorbar(nodes, ax=ax, label='Halting step')
    ax.set_title('Grafo colorato per halting step')
    ax.axis('off')
    return ax


def plot_ego_subgraph(data, halting_step, center_node, radius=2, ax=None):
    """
    Versione leggibile per grafi grandi: mostra solo l'ego-network
    di un nodo (utile per ispezionare bridge nodes o hub specifici).
    """
    G = to_networkx(data, to_undirected=True)
    sub_nodes = nx.ego_graph(G, center_node, radius=radius).nodes()
    subG = G.subgraph(sub_nodes)
    steps = halting_step.cpu().numpy()

    pos = nx.spring_layout(subG, seed=42)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    node_colors = [steps[n] for n in subG.nodes()]
    nodes = nx.draw_networkx_nodes(
        subG, pos, node_color=node_colors, cmap='viridis',
        node_size=200, ax=ax
    )
    nx.draw_networkx_labels(subG, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(subG, pos, alpha=0.3, ax=ax)
    plt.colorbar(nodes, ax=ax, label='Halting step')
    ax.set_title(f'Ego-network del nodo {center_node} (raggio {radius})')
    ax.axis('off')
    return ax


# ---------------------------------------------------------------------------
# 4. Scatter halting step vs singola metrica topologica (con correlazione)
# ---------------------------------------------------------------------------

def plot_halting_vs_metric(df, metric, ax=None, log_x=False):
    """
    Scatter + regressione lineare + coefficiente di correlazione di Spearman
    tra halting_step e una metrica topologica.
    Spearman invece di Pearson perché la relazione potrebbe non essere lineare
    e halting_step è discreto/ordinale.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    x = df[metric].values
    y = df['halting_step'].values

    valid = ~np.isnan(x)
    x, y = x[valid], y[valid]

    rho, pval = stats.spearmanr(x, y)

    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha': 0.4, 's': 15},
                line_kws={'color': 'red'})
    if log_x:
        ax.set_xscale('log')

    ax.set_xlabel(metric)
    ax.set_ylabel('Halting step')
    ax.set_title(f'{metric} vs Halting step\nSpearman ρ={rho:.3f}, p={pval:.1e}')
    return ax


# ---------------------------------------------------------------------------
# 5. Griglia riassuntiva: halting vs tutte le metriche insieme
# ---------------------------------------------------------------------------

def plot_halting_correlation_grid(df):
    """
    Una griglia di scatter plot: halting_step vs ciascuna metrica topologica.
    """
    metrics = [c for c in df.columns if c not in ('node', 'halting_step')]
    n = len(metrics)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        plot_halting_vs_metric(df, metric, ax=axes[i])

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Heatmap di correlazione (Spearman) tra tutte le metriche + halting
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df, ax=None):
    """Matrice di correlazione di Spearman tra halting_step e tutte le metriche."""
    cols = [c for c in df.columns if c != 'node']
    corr = df[cols].corr(method='spearman')

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlazione (Spearman) tra metriche topologiche e halting step')
    return ax


# ---------------------------------------------------------------------------
# 7. Boxplot halting step raggruppato per bucket di degree (o altra metrica)
# ---------------------------------------------------------------------------

def plot_halting_by_binned_metric(df, metric, n_bins=5, ax=None):
    """
    Utile quando lo scatter è troppo denso: raggruppa i nodi in bin
    (es. quantili di degree) e mostra un boxplot dell'halting step per bin.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    df = df.copy()
    df['bin'] = pd.qcut(df[metric], q=n_bins, duplicates='drop')

    sns.boxplot(data=df, x='bin', y='halting_step', ax=ax, color='steelblue')
    ax.set_xlabel(f'{metric} (bin)')
    ax.set_ylabel('Halting step')
    ax.set_title(f'Halting step per bin di {metric}')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    return ax


# ---------------------------------------------------------------------------
# Esempio d'uso end-to-end
# ---------------------------------------------------------------------------

# y, p, halting_step = model(data.x.to(device), data.edge_index.to(device))
#
# df = compute_topological_metrics(data, halting_step)
#
# plot_halting_distribution(df)
# plot_graph_with_halting(data, halting_step)
# plot_halting_correlation_grid(df)
# plot_correlation_heatmap(df)
# plot_halting_by_binned_metric(df, metric='degree')
# plot_halting_by_binned_metric(df, metric='betweenness')