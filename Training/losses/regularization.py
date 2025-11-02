import torch
import torch.nn as nn


class KoleoLoss(nn.Module):
    """
    Koleo regularization (Kozachenko-Leonenko entropy estimator)
    Encourages feature diversity by maximizing entropy
    Reference: Delattre & Fournier (2017)
    """
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        """
        Args:
            x: Features of shape [batch_size, feature_dim]
        Returns:
            Koleo loss (negative entropy estimate)
        """
        # Normalize features
        x = nn.functional.normalize(x, dim=-1)
        
        # Compute pairwise distances
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            pdist = torch.cdist(x, x, p=2)
            
            # For each point, find distance to nearest neighbor (excluding self)
            pdist = pdist + torch.eye(pdist.size(0), device=pdist.device) * 1e6
            min_dists = pdist.min(dim=-1)[0]
            
            # Koleo loss: negative log of geometric mean of nearest neighbor distances
            loss = -torch.log(min_dists + self.eps).mean()
        
        return loss

