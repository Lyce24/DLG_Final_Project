# for self-supervised learning, we consider cox loss and c-index loss
import torch
import torch.nn.functional as F

def c_index_ranking_loss(risk, time, event, margin=0.0):
    """
    Optimized, vectorized pairwise ranking loss for c-index optimization.
    
    Parameters:
    risk: (batch,) tensor of predicted risk scores.
    time: (batch,) tensor of survival times (can be log-transformed or raw).
    event: (batch,) tensor of event indicators (1 if event occurred, 0 if censored).
    margin: a float margin to enforce separation (default is 0.0).
    
    Returns:
    A scalar loss value that approximates a differentiable c-index loss.
    """
    # Compute pairwise differences:
    # diff[i,j] = risk[j] - risk[i] - margin
    diff = risk.unsqueeze(0) - risk.unsqueeze(1) - margin  # shape: (batch, batch)
    
    # Build a mask for valid pairs:
    # Valid if sample i had an event (event[i]==1) and time[i] < time[j]
    valid_mask = (time.unsqueeze(1) < time.unsqueeze(0)) & (event.unsqueeze(1) == 1)
    
    # Compute a smooth approximation of the indicator function:
    # Using the softplus function: softplus(x) = log(1 + exp(x))
    loss_matrix = F.softplus(diff)  # shape: (batch, batch)
    
    # Select only the valid pairs:
    valid_loss = loss_matrix[valid_mask]
    count = valid_loss.numel()
    
    if count > 0:
        loss = valid_loss.sum() / count
    else:
        # If there are no valid pairs, return a zero loss (with gradient).
        loss = torch.tensor(0.0, device=risk.device, requires_grad=True)
    return loss

def cox_partial_likelihood_loss(risk, time, event):
    """
    Compute the negative partial likelihood of the Cox proportional hazards model.
    
    Parameters:
    risk: Predicted risk score from the model, shape [N].
    time: Survival time (log-transformed), shape [N].
    event: Event indicator (1 if event occurred, 0 if censored), shape [N].
    """
    # Sort by descending survival time
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    time = time[order]
    event = event[order]
    
    exp_risk = torch.exp(risk)
    # The cumulative sum gives the sum of exp(risk) for the risk set
    cum_sum = torch.cumsum(exp_risk, dim=0)
    
    # For each event, compute risk - log(sum_{j in risk set} exp(risk_j))
    diff = risk - torch.log(cum_sum)
    loss = -torch.sum(diff * event) / (torch.sum(event) + 1e-8)
    return loss

# Example Cox loss computed on a mini-batch
def cox_partial_likelihood_loss_batch(risk, time, event):
    """
    Compute the negative partial likelihood of the Cox model for a mini-batch.
    Note: The risk set is limited to the mini-batch, which is an approximation.
    """
    # Sort mini-batch by descending time
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    time = time[order]
    event = event[order]
    
    exp_risk = torch.exp(risk)
    cum_sum = torch.cumsum(exp_risk, dim=0)
    
    diff = risk - torch.log(cum_sum + 1e-8)
    loss = -torch.sum(diff * event) / (torch.sum(event) + 1e-8)
    return loss

def cox_loss_vectorized(risk, time, event):
    """
    Vectorized Cox partial likelihood loss.
    
    risk: (batch,) tensor of predicted risk scores.
    time: (batch,) tensor of survival times (e.g., log-transformed).
    event: (batch,) tensor of event indicators (1 if event occurred, 0 if censored).
    """
    # Sort in descending order of time
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    time = time[order]
    event = event[order]
    
    exp_risk = torch.exp(risk)
    # Cumulative sum over the risk set (for each sample i, sum exp(risk[j]) for j >= i)
    cum_sum = torch.cumsum(exp_risk, dim=0)
    diff = risk - torch.log(cum_sum + 1e-8)
    loss = -torch.sum(diff * event) / (torch.sum(event) + 1e-8)
    return loss

def c_index_ranking_loss_vectorized(risk, time, event, margin=0.0):
    """
    Optimized, vectorized pairwise ranking loss that approximates the c-index.
    
    risk: (batch,) predicted risk scores.
    time: (batch,) survival times.
    event: (batch,) event indicators.
    margin: margin for the loss (default=0.0).
    """
    # Compute pairwise differences: diff[i,j] = risk[j] - risk[i] - margin.
    diff = risk.unsqueeze(0) - risk.unsqueeze(1) - margin  # shape: (batch, batch)
    
    # Valid pair mask: valid if sample i had an event and time[i] < time[j]
    valid_mask = (time.unsqueeze(1) < time.unsqueeze(0)) & (event.unsqueeze(1) == 1)
    
    # Use softplus as a smooth approximation: log(1 + exp(diff))
    loss_matrix = F.softplus(diff)
    
    valid_loss = loss_matrix[valid_mask]
    count = valid_loss.numel()
    
    if count > 0:
        return valid_loss.sum() / count
    else:
        return torch.tensor(0.0, device=risk.device, requires_grad=True)
    
def hybrid_survival_loss(risk, time, event, alpha=0.5, margin=0.0):
    """
    Hybrid loss combining Cox partial likelihood and ranking loss.
    
    alpha: weight for the Cox loss (0 <= alpha <= 1).
        (1 - alpha) is the weight for the ranking loss.
    """
    loss_cox = cox_loss_vectorized(risk, time, event)
    loss_rank = c_index_ranking_loss_vectorized(risk, time, event, margin)
    return alpha * loss_cox + (1 - alpha) * loss_rank