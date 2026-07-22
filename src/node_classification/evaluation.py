import torch
import torch.nn.functional as F
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
