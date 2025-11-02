import torch
import torch.nn as nn
import torch.nn.functional as F
from .dino import sinkhorn_knopp


class iBOTLoss(nn.Module):
    """
    iBOT (Image BERT pre-training with Online Tokenizer) Loss
    Masked patch prediction using student-teacher framework
    """
    
    def __init__(self, out_dim, teacher_temp, student_temp, patch_out_dim=1024):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.patch_out_dim = patch_out_dim
    
    def forward(self, student_patches, teacher_patches, mask):
        """
        Args:
            student_patches: [batch, num_patches, out_dim] student patch features
            teacher_patches: [batch, num_patches, out_dim] teacher patch features
            mask: [batch, num_patches] boolean mask (True = masked)
        Returns:
            iBOT loss (only on masked patches)
        """
        # Apply mask - only compute loss on masked patches
        student_masked = student_patches[mask]  # [num_masked, out_dim]
        teacher_masked = teacher_patches[mask]  # [num_masked, out_dim]
        
        if student_masked.size(0) == 0:
            return torch.tensor(0.0, device=student_patches.device)
        
        # Temperature scaling
        teacher_logits = teacher_masked / self.teacher_temp
        student_logits = student_masked / self.student_temp
        
        # Sinkhorn-Knopp on teacher
        teacher_probs = sinkhorn_knopp(teacher_logits)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        
        # Cross-entropy loss
        loss = torch.sum(-teacher_probs * student_log_probs, dim=-1).mean()
        
        return loss


def random_masking(x, mask_ratio):
    """
    Random masking of patches
    Args:
        x: [batch, num_patches, dim]
        mask_ratio: fraction of patches to mask
    Returns:
        mask: [batch, num_patches] boolean tensor
    """
    batch_size, num_patches, _ = x.shape
    num_masked = int(mask_ratio * num_patches)
    
    # Random noise for shuffling
    noise = torch.rand(batch_size, num_patches, device=x.device)
    
    # Sort noise to get random indices
    ids_shuffle = torch.argsort(noise, dim=1)
    
    # Create mask: True for masked patches
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=x.device)
    mask[torch.arange(batch_size).unsqueeze(1), ids_shuffle[:, :num_masked]] = True
    
    return mask

