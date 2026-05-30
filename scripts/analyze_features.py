import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Setup paths
    features_path = "data/test_features.pt" if os.path.exists("data/test_features.pt") else "features/test_features.pt"
    output_dir = "data/analysis_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ANALYZING MULTIMODAL FEATURES")
    print(f"Loading: {features_path}")
    print("=" * 60)
    
    if not os.path.exists(features_path):
        print(f"Error: Feature file not found at {features_path}")
        return
        
    # Load dataset dictionary
    data = torch.load(features_path, map_location="cpu")
    
    # 2. Basic Inspection
    print(f"Keys in feature file: {list(data.keys())}")
    
    z_img = data['z_img']
    z_aud = data['z_aud']
    v_teacher = data['v_teacher']
    video_ids = data.get('video_ids', [])
    
    num_samples = len(z_img)
    print(f"Number of samples: {num_samples}")
    print(f"z_img (CLIP Vision Student) shape:     {z_img.shape}  | dtype: {z_img.dtype}")
    print(f"z_aud (VGGish Audio Student) shape:    {z_aud.shape}  | dtype: {z_aud.dtype}")
    print(f"v_teacher (ImageBind Video) shape:    {v_teacher.shape} | dtype: {v_teacher.dtype}")
    print(f"Number of video IDs:                   {len(video_ids)}")
    print("-" * 60)
    
    # Cast float16 tensors to float32 for math consistency
    z_img = z_img.to(torch.float32)
    z_aud = z_aud.to(torch.float32)
    v_teacher = v_teacher.to(torch.float32)
    
    # 3. Data Integrity & Value Distribution Checks
    print("Checking for anomalies (NaNs, Infs, zero vectors):")
    for name, tensor in [('z_img', z_img), ('z_aud', z_aud), ('v_teacher', v_teacher)]:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        
        # Check for zero vectors (row-wise L2 norm is very close to 0)
        norms = torch.norm(tensor, p=2, dim=1)
        zeros_count = (norms < 1e-5).sum().item()
        
        print(f"  {name:10s}: NaNs={nan_count}, Infs={inf_count}, ZeroVectors={zeros_count}/{num_samples} ({zeros_count/num_samples*100:.1f}%)")
        print(f"              Min={tensor.min().item():.4f}, Max={tensor.max().item():.4f}, Mean={tensor.mean().item():.4f}, Std={tensor.std().item():.4f}")
    
    print("-" * 60)
    
    # 4. L2 Norm analysis
    print("L2 Norm analysis (Mean +/- Std):")
    norm_img = torch.norm(z_img, p=2, dim=1)
    norm_aud = torch.norm(z_aud, p=2, dim=1)
    norm_teacher = torch.norm(v_teacher, p=2, dim=1)
    
    print(f"  z_img L2 norms:     {norm_img.mean().item():.4f} +/- {norm_img.std().item():.4f}")
    print(f"  z_aud L2 norms:     {norm_aud.mean().item():.4f} +/- {norm_aud.std().item():.4f} (including silent)")
    # Filter out zero vectors for audio L2 norm stats
    non_zero_aud = norm_aud[norm_aud > 1e-5]
    if len(non_zero_aud) > 0:
        print(f"  z_aud L2 (non-zero): {non_zero_aud.mean().item():.4f} +/- {non_zero_aud.std().item():.4f}")
    print(f"  v_teacher L2 norms: {norm_teacher.mean().item():.4f} +/- {norm_teacher.std().item():.4f}")
    
    print("-" * 60)
    
    # 5. Within-modality Self-similarity check
    # Check self-similarity to see if features are collapsed or well-distributed
    z_img_norm = F.normalize(z_img, p=2, dim=1)
    v_teacher_norm = F.normalize(v_teacher, p=2, dim=1)
    
    sim_img_img = torch.mm(z_img_norm, z_img_norm.t())
    sim_teacher_teacher = torch.mm(v_teacher_norm, v_teacher_norm.t())
    
    off_diag_mask = ~torch.eye(num_samples, dtype=torch.bool)
    off_diag_sim_img = sim_img_img[off_diag_mask]
    off_diag_sim_teacher = sim_teacher_teacher[off_diag_mask]
    
    print("Within-Modality Self-Similarity (Pairwise Cosine Similarity between different samples):")
    print(f"  z_img (CLIP)  - Mean Off-Diag Sim: {off_diag_sim_img.mean().item():.4f} | Std: {off_diag_sim_img.std().item():.4f} | Min/Max: {off_diag_sim_img.min().item():.4f}/{off_diag_sim_img.max().item():.4f}")
    print(f"  v_teacher     - Mean Off-Diag Sim: {off_diag_sim_teacher.mean().item():.4f} | Std: {off_diag_sim_teacher.std().item():.4f} | Min/Max: {off_diag_sim_teacher.min().item():.4f}/{off_diag_sim_teacher.max().item():.4f}")
    
    valid_aud_idx = (norm_aud > 1e-5).nonzero().squeeze(1)
    if len(valid_aud_idx) > 1:
        z_aud_valid = F.normalize(z_aud[valid_aud_idx], p=2, dim=1)
        sim_aud_aud = torch.mm(z_aud_valid, z_aud_valid.t())
        off_diag_mask_aud = ~torch.eye(len(valid_aud_idx), dtype=torch.bool)
        off_diag_sim_aud = sim_aud_aud[off_diag_mask_aud]
        print(f"  z_aud (VGGish)- Mean Off-Diag Sim: {off_diag_sim_aud.mean().item():.4f} | Std: {off_diag_sim_aud.std().item():.4f} | Min/Max: {off_diag_sim_aud.min().item():.4f}/{off_diag_sim_aud.max().item():.4f}")
    else:
        print("  z_aud (VGGish)- Skipping pairwise sim check (all audio features are zero).")
        
    print("=" * 60)
    
    # 6. GENERATE VISUALS
    print("Generating and saving visuals to 'features/analysis_plots/'...")
    
    # Style configuration for premium design
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.facecolor'] = '#121212'
    plt.rcParams['figure.facecolor'] = '#121212'
    plt.rcParams['text.color'] = '#E0E0E0'
    plt.rcParams['axes.labelcolor'] = '#E0E0E0'
    plt.rcParams['xtick.color'] = '#B0B0B0'
    plt.rcParams['ytick.color'] = '#B0B0B0'
    plt.rcParams['grid.color'] = '#333333'
    
    # Color palette
    colors = {
        'z_img': '#00ADB5',      # Cyan
        'z_aud': '#FF2E93',      # Hot Pink
        'v_teacher': '#FFD369',  # Pastel Yellow
    }
    
    # --- PLOT 1: L2 Norms Distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(norm_img.numpy(), bins=30, alpha=0.7, label='z_img (CLIP Vision)', color=colors['z_img'])
    # Filter zeros for audio histogram if not all zeros
    non_zero_aud_norms = norm_aud[norm_aud > 1e-5]
    if len(non_zero_aud_norms) > 0:
        ax.hist(non_zero_aud_norms.numpy(), bins=30, alpha=0.7, label='z_aud (VGGish)', color=colors['z_aud'])
    else:
        ax.axvline(0.0, color=colors['z_aud'], linestyle='--', linewidth=2, label='z_aud (All-Zero)')
    ax.hist(norm_teacher.numpy(), bins=30, alpha=0.7, label='v_teacher (ImageBind)', color=colors['v_teacher'])
    ax.set_title("Embedding L2 Norm Distributions", fontsize=14, pad=15, color='#FFFFFF', weight='bold')
    ax.set_xlabel("L2 Norm Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(facecolor='#1E1E1E', edgecolor='#333333', labelcolor='#E0E0E0')
    ax.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/l2_norms.png", dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    # --- PLOT 2: Pairwise Self-Similarity Heatmaps (First 50 samples) ---
    subset_size = min(50, num_samples)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Image self-similarity
    sim_img_subset = sim_img_img[:subset_size, :subset_size].numpy()
    im1 = axes[0].imshow(sim_img_subset, cmap='magma', aspect='equal', vmin=0, vmax=1)
    axes[0].set_title(f"z_img (CLIP) Pairwise Cosine Similarity (First {subset_size})", fontsize=12, pad=12, color='#FFFFFF', weight='bold')
    axes[0].set_xlabel("Sample Index", fontsize=10)
    axes[0].set_ylabel("Sample Index", fontsize=10)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Teacher video self-similarity
    sim_teacher_subset = sim_teacher_teacher[:subset_size, :subset_size].numpy()
    im2 = axes[1].imshow(sim_teacher_subset, cmap='magma', aspect='equal', vmin=0, vmax=1)
    axes[1].set_title(f"v_teacher (ImageBind) Pairwise Cosine Similarity (First {subset_size})", fontsize=12, pad=12, color='#FFFFFF', weight='bold')
    axes[1].set_xlabel("Sample Index", fontsize=10)
    axes[1].set_ylabel("Sample Index", fontsize=10)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/self_similarity_heatmaps.png", dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    # --- PLOT 3: Value Distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, tensor, col) in zip(axes, [
        ('z_img (CLIP)', z_img, colors['z_img']),
        ('z_aud (VGGish)', z_aud, colors['z_aud']),
        ('v_teacher (ImageBind)', v_teacher, colors['v_teacher'])
    ]):
        vals = tensor.numpy().flatten()
        if len(vals) > 100000:
            vals = np.random.choice(vals, 100000, replace=False)
        ax.hist(vals, bins=50, color=col, alpha=0.7)
        ax.set_title(f"Feature Value Distribution\n({name})", fontsize=12, color='#FFFFFF', weight='bold', pad=10)
        ax.set_xlabel("Value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_value_distributions.png", dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    # --- PLOT 4: PCA Projections ---
    def get_pca_2d(tensor):
        mean = tensor.mean(dim=0, keepdim=True)
        centered = tensor - mean
        # SVD
        _, _, V = torch.svd(centered)
        projected = torch.mm(centered, V[:, :2])
        return projected.numpy()
        
    try:
        pca_img = get_pca_2d(z_img)
        pca_teacher = get_pca_2d(v_teacher)
        
        has_aud = len(valid_aud_idx) > 1
        num_subplots = 3 if has_aud else 2
        fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 5.5))
        
        # Plot CLIP
        axes[0].scatter(pca_img[:, 0], pca_img[:, 1], c=colors['z_img'], alpha=0.6, edgecolors='none', s=20)
        axes[0].set_title("PCA: z_img (CLIP 512d -> 2d)", fontsize=13, color='#FFFFFF', weight='bold', pad=10)
        axes[0].grid(True, linestyle='--', alpha=0.15)
        
        # Plot VGGish
        idx_teacher = 1
        if has_aud:
            pca_aud = get_pca_2d(z_aud[valid_aud_idx])
            axes[1].scatter(pca_aud[:, 0], pca_aud[:, 1], c=colors['z_aud'], alpha=0.6, edgecolors='none', s=20)
            axes[1].set_title("PCA: z_aud (VGGish 128d -> 2d)", fontsize=13, color='#FFFFFF', weight='bold', pad=10)
            axes[1].grid(True, linestyle='--', alpha=0.15)
            idx_teacher = 2
            
        # Plot ImageBind Video
        axes[idx_teacher].scatter(pca_teacher[:, 0], pca_teacher[:, 1], c=colors['v_teacher'], alpha=0.6, edgecolors='none', s=20)
        axes[idx_teacher].set_title("PCA: v_teacher (ImageBind 1024d -> 2d)", fontsize=13, color='#FFFFFF', weight='bold', pad=10)
        axes[idx_teacher].grid(True, linestyle='--', alpha=0.15)
        
        plt.suptitle("2D PCA Projections of Embedding Spaces", fontsize=16, color='#FFFFFF', weight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_projections.png", dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        print("Plots generated successfully!")
    except Exception as e:
        print(f"Error generating PCA plots: {e}")
        
    print("=" * 60)
    print("Feature analysis completed!")
    print(f"Check visuals in: {os.path.abspath(output_dir)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
