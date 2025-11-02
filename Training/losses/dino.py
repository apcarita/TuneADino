import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp(Q, n_iters=3):
    """
    Sinkhorn-Knopp algorithm for optimal transport
    Normalizes Q to have equal row and column sums
    
    Args:
        Q: [batch_size, num_prototypes] logits
        n_iters: number of iterations
    Returns:
        Normalized distribution
    """
    Q = torch.exp(Q).T  # [num_prototypes, batch_size]
    B = Q.shape[1]
    K = Q.shape[0]
    
    # Make Q doubly-stochastic
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    
    for _ in range(n_iters):
        # Normalize rows
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K
        
        # Normalize columns
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    
    Q *= B
    return Q.T


class DINOLoss(nn.Module):
    """
    DINO loss with Sinkhorn-Knopp centering
    Based on ExPLoRA / DINOv2 implementation
    """
    
    def __init__(self, out_dim, teacher_temp, student_temp, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
    
    def forward(self, student_output, teacher_output, use_sinkhorn=True):
        """
        Args:
            student_output: List of student predictions (all crops)
            teacher_output: List of teacher predictions (global crops only)
            use_sinkhorn: Use Sinkhorn-Knopp instead of simple centering
        """
        # Apply centering to teacher
        teacher_out_centered = [(t - self.center) / self.teacher_temp 
                                for t in teacher_output]
        
        if use_sinkhorn:
            teacher_probs = [sinkhorn_knopp(t) for t in teacher_out_centered]
        else:
            teacher_probs = [F.softmax(t, dim=-1) for t in teacher_out_centered]
        
        # Student logits
        student_logits = [s / self.student_temp for s in student_output]
        student_log_probs = [F.log_softmax(s, dim=-1) for s in student_logits]
        
        # Cross-entropy between all teacher-student pairs (except self-pairs)
        total_loss = 0
        n_loss_terms = 0
        
        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_log_prob in enumerate(student_log_probs):
                # Skip self-pairs (global crop i -> global crop i)
                if s_idx < 2 and t_idx == s_idx:
                    continue
                
                loss = torch.sum(-t_prob * s_log_prob, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        
        # Update center with EMA
        self.update_center(teacher_output)
        
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center using exponential moving average"""
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                     batch_center * (1 - self.center_momentum)

