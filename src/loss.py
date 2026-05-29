import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Computes the InfoNCE (Information Noise-Contrastive Estimation) loss.
    Can be run as asymmetric (student to teacher) or symmetric (bi-directional).
    """
    def __init__(self, temperature=0.07, symmetric=True):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        
    def forward(self, v_pred, v_teacher):
        # Normalize the embeddings to unit vectors
        v_pred_norm = F.normalize(v_pred, p=2, dim=-1)
        v_teacher_norm = F.normalize(v_teacher, p=2, dim=-1)
        
        # Compute cosine similarity matrix
        # Shape: [B, B]
        similarity_matrix = torch.matmul(v_pred_norm, v_teacher_norm.T) / self.temperature
        
        # Targets are the diagonal indices (each prediction matches its corresponding teacher embedding)
        batch_size = v_pred.size(0)
        targets = torch.arange(batch_size, device=v_pred.device)
        
        if self.symmetric:
            # Loss in both directions
            loss_a = F.cross_entropy(similarity_matrix, targets)
            loss_b = F.cross_entropy(similarity_matrix.T, targets)
            return (loss_a + loss_b) / 2.0
        else:
            return F.cross_entropy(similarity_matrix, targets)
