# Pondering Graph Neural Networks

Adapting [PonderNet](https://arxiv.org/abs/2107.05407) (Banino et al., 2021) to Graph Neural Networks, so that the amount of message-passing computation — and therefore each node's or graph's effective receptive field — is learned rather than fixed in advance.

## Motivation

A standard GCN performs a **fixed number of message-passing layers** for every node/graph, regardless of how much structural context is actually needed to make a good prediction. This work asks: *what if the model could learn how much computation a given input needs, and adapt its receptive field accordingly?*

PonderNet answers this for standard feed-forward networks by treating the number of computation steps as a learned, stochastic quantity, optimized end-to-end via a halting probability distribution regularized toward a geometric prior. This repo combines that mechanism with message passing: **each ponder step corresponds to one additional hop of graph propagation.**

## Core idea

Message passing is inherently local (each layer expands the receptive field by exactly one hop), but the amount of context needed to solve a task can vary a lot — across nodes within the same graph, or across graphs of different sizes/topologies. The framework is applied to two settings:

- **Node classification** — each node has its own halting step; once a node halts, its embedding stops updating, but message passing keeps propagating information for the rest of the graph.
- **Graph classification** — halting is defined at the graph level (via pooling), so the "receptive field" is relative to the entire graph rather than to individual nodes.

## Architecture

Each ponder step is a small recursive unit:

1. A message-passing layer updates the node embeddings.
2. An MLP consumes the updated embeddings (plus the running state) and produces:
   - the class prediction `y` for this step,
   - the halting logit, from which the conditional halting probability `λ` is derived.
3. A Bernoulli sample on `λ` decides whether to stop.

For node classification, halting a node means freezing its embedding — message passing continues for the rest of the graph, but the halted node no longer updates.

## Training objective

Training unrolls a fixed maximum number of steps `n_max` and optimizes:

```
L(λ_p, β) = Σ_{i=1}^{n_max} p_n(i) · L(y, y_true) + D_KL[ p_n || q(λ_p) ]
```

- `p_n(i) = λ_i · Π_{k<i} (1 - λ_k)` — the marginal probability of having halted at step `i`; this is why every step must be computed during training, even ones that will be sampled as "already halted".
- The KL term regularizes the learned halting distribution `p_n` toward a **geometric prior** parameterized by `λ_p`, encouraging the model not to over- or under-compute relative to a controllable expected number of steps.

## Results

### Graph classification (ENZYMES, MUTAG, NCI1, PROTEINS)

Pondering vs. a depth-matched GCN baseline: results are dataset-dependent — the pondering model outperforms the baseline on some datasets and underperforms on others, with no single consistent winner across the board. The average halting step and its correlation with graph-level structural statistics (number of nodes/edges, density, average degree, clustering, diameter) is analyzed per dataset.

### Node classification (CiteSeer, Roman-Empire)

Test accuracy as a function of the ponder step used for the readout ("what if every node were read out at step *n*?") shows opposite trends on the two datasets — accuracy increases with more steps on CiteSeer, but decreases on Roman-Empire — suggesting the effective receptive field required is genuinely dataset-dependent, not something the model would benefit from maximizing uniformly. On both datasets, no strong correlation is found between the average halting step and standard node centrality measures (degree, clustering, betweenness, closeness, eigenvector centrality), and the pondering model does not consistently beat a plain baseline of comparable depth — adapting per-node depth appears to be a harder optimization problem than learning a single fixed depth for the whole graph.

## Repository structure

```
src/
├── node_classification/
│   ├── models.py       # GCPondNet, GCNet_baseline
│   ├── loss.py          # pondering_loss
│   ├── evaluation.py    # accuracy / halting-step utilities
│   └── plots.py
├── graph_classification/
│   ├── models.py         # GCPondNet_g_classification, GCNet_baseline_g_classification
│   ├── loss.py           # pondering_loss_g_classification
│   └── evaluation.py
└── metrics.py             # oversmoothing_metric and other shared metrics
```

## Author

**Tonet Lorenzo** — Deep Learning, Final Project


<img width="1402" height="1122" alt="image" src="https://github.com/user-attachments/assets/9a28f3e2-f08d-4886-a648-72fe649f3327" />
