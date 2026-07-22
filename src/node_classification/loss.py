import torch
import torch.nn.functional as F

def pondering_loss(p, y, labels, mask, beta=0.01, prior_lambda=1/30, eps=1e-10, direct_kl = True):
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
    
    if direct_kl:
        kl_reg = F.kl_div(
            torch.log(p_masked + eps),
            torch.log(prior_expanded + eps),
            log_target=True,
            reduction='none'
        ).sum(dim=0).mean()
    else:
        kl_reg = F.kl_div(
            torch.log(prior_expanded + eps),
            torch.log(p_masked + eps),
            reduction='none',
            log_target=True,
        ).sum(dim=0).mean()

    total_loss = rec_loss + beta * kl_reg

    return total_loss, rec_loss, kl_reg


