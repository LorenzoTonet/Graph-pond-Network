import torch

def compute_pondering_accuracy_graph(y, halting_step, labels):
    """
    y            : [n_steps, num_graphs, n_classes] - logits ad ogni step
    halting_step : [num_graphs] - step (1-indexed) in cui ogni grafo si è fermato
    labels       : [num_graphs] - label vere

    Ritorna (num_corretti, num_totali) cosi' da poter accumulare su piu' batch.
    """
    n_steps, num_graphs, n_classes = y.shape

    # halting_step è 1-indexed -> passa a 0-indexed, clamp per sicurezza
    idx = (halting_step.long() - 1).clamp(min=0, max=n_steps - 1)

    # gather lungo la dimensione degli step: per ogni grafo prendo y[idx[i], i, :]
    idx_exp = idx.view(1, num_graphs, 1).expand(1, num_graphs, n_classes)
    y_at_halt = y.gather(0, idx_exp).squeeze(0)  # [num_graphs, n_classes]

    preds = y_at_halt.argmax(dim=-1)
    correct = (preds == labels).sum().item()

    return correct, num_graphs


@torch.no_grad()
def evaluate_pondering_accuracy(model, loader, device):
    """
    Fa girare il modello di ponder su tutto il loader, fermandosi (per ogni grafo)
    al proprio halting_step, e calcola l'accuracy aggregata.

    Ritorna (accuracy, avg_halting_step).
    """
    model.eval()

    total_correct = 0
    total_graphs = 0
    halting_steps_all = []

    for batch in loader:
        batch = batch.to(device)

        y, p, halting_step, emb = model(batch.x, batch.edge_index, batch.batch)

        correct, n = compute_pondering_accuracy_graph(y, halting_step, batch.y)
        total_correct += correct
        total_graphs += n
        halting_steps_all.append(halting_step)

    accuracy = total_correct / total_graphs
    avg_halting_step = torch.cat(halting_steps_all).float().mean().item()

    return accuracy, avg_halting_step

@torch.no_grad()
def evaluate_baseline_accuracy(model, loader, device):
    model.eval()

    total_correct = 0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        logits, emb = model(batch.x, batch.edge_index, batch.batch)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == batch.y).sum().item()
        total_graphs += batch.y.shape[0]

    return total_correct / total_graphs