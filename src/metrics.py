import torch
import numpy as np

def calculate_proximity(v_pred, v_teacher):
    """
    Computes latent space proximity metrics between predicted and teacher embeddings.
    """
    v_pred_norm = torch.nn.functional.normalize(v_pred, p=2, dim=-1)
    v_teacher_norm = torch.nn.functional.normalize(v_teacher, p=2, dim=-1)
    
    # Cosine similarities
    cos_sims = torch.sum(v_pred_norm * v_teacher_norm, dim=-1).cpu().numpy()
    mean_cos = float(np.mean(cos_sims))
    
    # Mean Squared Error (MSE)
    mse = float(torch.nn.functional.mse_loss(v_pred, v_teacher).item())
    
    return {
        "mean_cosine_similarity": mean_cos,
        "mse": mse,
        "cosine_similarities": cos_sims
    }

def calculate_retrieval_metrics(v_pred, v_teacher):
    """
    Computes Recall@K (R@1, R@5, R@10) and Median Rank (MedR).
    Query: v_pred
    Gallery: v_teacher
    """
    # Normalize embeddings
    v_pred_norm = torch.nn.functional.normalize(v_pred, p=2, dim=-1)
    v_teacher_norm = torch.nn.functional.normalize(v_teacher, p=2, dim=-1)
    
    # Compute similarity matrix of shape [Num_Queries, Num_Gallery]
    sim_matrix = torch.matmul(v_pred_norm, v_teacher_norm.T).cpu().numpy()
    
    num_queries = sim_matrix.shape[0]
    ranks = []
    
    for i in range(num_queries):
        # Get similarities for query i
        sims = sim_matrix[i]
        
        # Correct item index is i
        correct_sim = sims[i]
        
        # Calculate rank: count how many items have similarity >= correct item similarity
        # (1-indexed: if it's the highest, rank is 1)
        # To handle exact ties conservatively, we check sims > correct_sim and add 1
        rank = np.sum(sims > correct_sim) + 1
        ranks.append(rank)
        
    ranks = np.array(ranks)
    
    # Calculate Recall@K
    r1 = float(np.mean(ranks <= 1)) * 100
    r5 = float(np.mean(ranks <= 5)) * 100
    r10 = float(np.mean(ranks <= 10)) * 100
    
    # Median Rank
    med_r = float(np.median(ranks))
    
    return {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "MedR": med_r,
        "ranks": ranks.tolist()
    }
