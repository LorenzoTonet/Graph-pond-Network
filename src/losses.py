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
            p_masked + eps,
            torch.log(prior_expanded + eps),
            reduction='none'
        ).sum(dim=0).mean()
    else:
        kl_reg = F.kl_div(
            torch.log(prior_expanded + eps),
            p_masked + eps,
            reduction='none'
        ).sum(dim=0).mean()

    total_loss = rec_loss + beta * kl_reg

    return total_loss, rec_loss, kl_reg


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
