import torch
import torch.nn.functional as F

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
