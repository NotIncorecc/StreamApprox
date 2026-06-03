"""
Analysis script for V2-patched features (data_V2_patched/).

V2-patched schema (same as V2 but audio is now real, not zeros):
  z_img     : [N, 3, 512]   - CLIP ViT-B/32, 3 frames (early/mid/late)
  z_aud     : [N, 128]      - VGGish audio (zeros when silent)
  v_teacher : [N, 1024]     - ImageBind multimodal teacher (L2-normalized)
  has_audio : [N]           - bool flag, True when IB audio was fused into teacher
  video_ids : list[str]
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.facecolor": "#121212",
    "figure.facecolor": "#121212",
    "savefig.facecolor": "#121212",
    "text.color": "#E0E0E0",
    "axes.labelcolor": "#E0E0E0",
    "xtick.color": "#B0B0B0",
    "ytick.color": "#B0B0B0",
    "grid.color": "#333333",
    "axes.edgecolor": "#444444",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "z_img":      "#00ADB5",
    "z_aud":      "#FF2E93",
    "v_teacher":  "#FFD369",
    "early":      "#4FC3F7",
    "mid":        "#00ADB5",
    "late":       "#006064",
    "train":      "#A5D6A7",
    "test":       "#EF9A9A",
}

PCA_SUBSET = 2000
SIM_SUBSET = 1000


def load(path):
    print(f"  Loading {path} ...")
    d = torch.load(path, map_location="cpu", weights_only=False)
    d["z_img"]     = d["z_img"].to(torch.float32)
    d["z_aud"]     = d["z_aud"].to(torch.float32)
    d["v_teacher"] = d["v_teacher"].to(torch.float32)
    d["has_audio"] = d["has_audio"].to(torch.bool)
    return d


def pca_2d(tensor, n=PCA_SUBSET):
    x = tensor[:n].clone()
    mu = x.mean(0, keepdim=True)
    x -= mu
    _, S, V = torch.linalg.svd(x, full_matrices=False)
    var_ratio = (S[:2] ** 2) / (S ** 2).sum()
    proj = x @ V[:2].T
    return proj.numpy(), var_ratio.numpy()


def pca_scree(tensor, k=20, n=PCA_SUBSET):
    x = tensor[:n].clone()
    x -= x.mean(0, keepdim=True)
    _, S, _ = torch.linalg.svd(x, full_matrices=False)
    var_ratio = (S[:k] ** 2) / (S ** 2).sum()
    return var_ratio.numpy()


def pairwise_cosine(tensor, n=SIM_SUBSET):
    t = F.normalize(tensor[:n], dim=1)
    return torch.mm(t, t.T).numpy()


def fmt(x):
    return f"{x:.4f}"


def banner(title):
    w = 64
    print("\n" + "=" * w)
    pad = (w - len(title) - 2) // 2
    print(" " * pad + title)
    print("=" * w)


def print_stats(name, tensor):
    norms = torch.norm(tensor, p=2, dim=-1).flatten()
    nan_n = torch.isnan(tensor).sum().item()
    inf_n = torch.isinf(tensor).sum().item()
    zero_n = (norms < 1e-5).sum().item()
    n = len(norms)
    print(
        f"  {name:25s}  shape={list(tensor.shape)}"
        f"  NaN={nan_n}  Inf={inf_n}  Zero={zero_n}/{n}"
        f"  L2 {norms.mean():.3f}+/-{norms.std():.3f}"
        f"  val [{tensor.min():.3f}, {tensor.max():.3f}]"
        f"  mean={tensor.mean():.4f}  std={tensor.std():.4f}"
    )
    return norms


def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_l2_norms(norms_dict, audio_mask, split, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, norms in norms_dict.items():
        if key == "z_aud":
            norms = norms[audio_mask]
        ax.hist(norms.numpy(), bins=40, alpha=0.7,
                label=key if key != "z_aud" else "z_aud (audio-only)",
                color=COLORS[key])
    ax.set_title(f"L2 Norm Distributions — {split}", fontsize=13,
                 color="#FFFFFF", weight="bold", pad=12)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")
    ax.legend(facecolor="#1E1E1E", edgecolor="#333333", labelcolor="#E0E0E0")
    ax.grid(True, linestyle="--", alpha=0.2)
    savefig(fig, os.path.join(out, f"{split}_l2_norms.png"))


def plot_value_histograms(data_dict, audio_mask, split, out):
    triplets = [
        ("z_img (CLIP, flat 3 frames)", data_dict["z_img"].reshape(-1, 512),  COLORS["z_img"]),
        ("z_aud (VGGish, audio-only)",  data_dict["z_aud"][audio_mask],        COLORS["z_aud"]),
        ("v_teacher (ImageBind fused)", data_dict["v_teacher"],                COLORS["v_teacher"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, t, col) in zip(axes, triplets):
        vals = t.numpy().flatten()
        if len(vals) > 200_000:
            rng = np.random.default_rng(0)
            vals = rng.choice(vals, 200_000, replace=False)
        ax.hist(vals, bins=60, color=col, alpha=0.75)
        ax.set_title(f"{name}", fontsize=10, color="#FFFFFF", weight="bold", pad=8)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.2)
    fig.suptitle(f"Feature Value Distributions — {split}", fontsize=13,
                 color="#FFFFFF", weight="bold", y=1.02)
    plt.tight_layout()
    savefig(fig, os.path.join(out, f"{split}_value_histograms.png"))


def plot_per_frame_norms(z_img, split, out):
    labels = ["Early (idx 1)", "Middle (idx N//2)", "Late (idx -2)"]
    frame_colors = [COLORS["early"], COLORS["mid"], COLORS["late"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, (ax, lbl, col) in enumerate(zip(axes, labels, frame_colors)):
        norms = torch.norm(z_img[:, i, :], p=2, dim=1).numpy()
        ax.hist(norms, bins=40, color=col, alpha=0.8)
        ax.set_title(lbl, fontsize=11, color="#FFFFFF", weight="bold", pad=8)
        ax.set_xlabel("L2 Norm")
        ax.set_ylabel("Count")
        ax.axvline(norms.mean(), color="#FFFFFF", linestyle="--", linewidth=1.2,
                   label=f"mean={norms.mean():.2f}")
        ax.legend(facecolor="#1E1E1E", edgecolor="#333333", labelcolor="#E0E0E0", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.2)
    fig.suptitle(f"Per-Frame CLIP Norm Distributions — {split}", fontsize=13,
                 color="#FFFFFF", weight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(out, f"{split}_per_frame_norms.png"))


def plot_inter_frame_cosine(z_img, split, out):
    pairs = [(0, 1, "early↔mid"), (0, 2, "early↔late"), (1, 2, "mid↔late")]
    pair_colors = ["#80CBC4", "#4DD0E1", "#00ACC1"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (i, j, lbl), col in zip(axes, pairs, pair_colors):
        sims = F.cosine_similarity(z_img[:, i, :], z_img[:, j, :], dim=1).numpy()
        ax.hist(sims, bins=50, color=col, alpha=0.8)
        ax.set_title(f"Cosine Sim: {lbl}", fontsize=11, color="#FFFFFF", weight="bold", pad=8)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.axvline(sims.mean(), color="#FFFFFF", linestyle="--", linewidth=1.2,
                   label=f"mean={sims.mean():.3f}")
        ax.legend(facecolor="#1E1E1E", edgecolor="#333333", labelcolor="#E0E0E0", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.2)
    fig.suptitle(f"Inter-Frame Cosine Similarity — {split}", fontsize=13,
                 color="#FFFFFF", weight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(out, f"{split}_inter_frame_cosine.png"))


def plot_audio_coverage(has_audio_dict, out):
    """Bar chart comparing audio coverage between train and test."""
    fig, ax = plt.subplots(figsize=(6, 5))
    splits = list(has_audio_dict.keys())
    counts_with = [has_audio_dict[s].sum().item() for s in splits]
    counts_total = [len(has_audio_dict[s]) for s in splits]
    counts_without = [t - w for t, w in zip(counts_total, counts_with)]

    x = np.arange(len(splits))
    w = 0.35
    bars_with    = ax.bar(x - w/2, counts_with,    w, label="Has Audio",    color=COLORS["z_aud"],     alpha=0.85)
    bars_without = ax.bar(x + w/2, counts_without, w, label="Silent/No Audio", color=COLORS["z_img"], alpha=0.85)

    for bar, cnt, tot in zip(bars_with, counts_with, counts_total):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{100*cnt/tot:.1f}%", ha="center", va="bottom", fontsize=9, color="#E0E0E0")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Number of Samples")
    ax.set_title("Audio Coverage — Train vs Test", fontsize=13,
                 color="#FFFFFF", weight="bold", pad=12)
    ax.legend(facecolor="#1E1E1E", edgecolor="#333333", labelcolor="#E0E0E0")
    ax.grid(True, linestyle="--", alpha=0.2, axis="y")
    savefig(fig, os.path.join(out, "audio_coverage.png"))


def plot_cross_modal_analysis(z_img, z_aud, v_teacher, audio_mask, split, out):
    z_img_mean = z_img.mean(dim=1)
    norm_img = torch.norm(z_img_mean, p=2, dim=1).numpy()
    norm_t   = torch.norm(v_teacher, p=2, dim=1).numpy()

    r_img_t = float(np.corrcoef(norm_img, norm_t)[0, 1])

    n_s = min(500, len(z_img_mean))
    sm_img = pairwise_cosine(z_img_mean, n_s)
    sm_t   = pairwise_cosine(v_teacher,  n_s)
    mask   = ~np.eye(n_s, dtype=bool)
    r_struct = float(np.corrcoef(sm_img[mask], sm_t[mask])[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].scatter(norm_img, norm_t, s=6, alpha=0.35, color=COLORS["z_img"], edgecolors="none")
    axes[0].set_title(f"L2 Norm Correlation\n(Pearson r = {r_img_t:.4f})",
                      fontsize=11, color="#FFFFFF", weight="bold", pad=8)
    axes[0].set_xlabel("z_img mean L2 norm (CLIP 512d)")
    axes[0].set_ylabel("v_teacher L2 norm (IB 1024d)")
    axes[0].grid(True, linestyle="--", alpha=0.2)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(sm_img[mask]), size=min(20_000, len(sm_img[mask])), replace=False)
    axes[1].scatter(sm_img[mask][idx], sm_t[mask][idx], s=3, alpha=0.2,
                    color=COLORS["v_teacher"], edgecolors="none")
    axes[1].set_title(
        f"Pairwise Similarity Structure\n(Pearson r = {r_struct:.4f}, first {n_s} samples)",
        fontsize=11, color="#FFFFFF", weight="bold", pad=8
    )
    axes[1].set_xlabel("CLIP pairwise cosine sim")
    axes[1].set_ylabel("ImageBind pairwise cosine sim")
    axes[1].grid(True, linestyle="--", alpha=0.2)

    if audio_mask.sum() > 10:
        norm_aud = torch.norm(z_aud[audio_mask], p=2, dim=1).numpy()
        norm_t_a = norm_t[audio_mask.numpy()]
        r_aud_t = float(np.corrcoef(norm_aud, norm_t_a)[0, 1])
        axes[2].scatter(norm_aud, norm_t_a, s=6, alpha=0.35, color=COLORS["z_aud"], edgecolors="none")
        axes[2].set_title(f"z_aud vs v_teacher L2 Norms\n(audio samples, r = {r_aud_t:.4f})",
                          fontsize=11, color="#FFFFFF", weight="bold", pad=8)
        axes[2].set_xlabel("z_aud L2 norm (VGGish 128d)")
        axes[2].set_ylabel("v_teacher L2 norm (IB 1024d)")
        axes[2].grid(True, linestyle="--", alpha=0.2)
    else:
        axes[2].text(0.5, 0.5, "No audio samples\nin this split",
                     ha="center", va="center", color="#B0B0B0", fontsize=12,
                     transform=axes[2].transAxes)
        axes[2].set_title("z_aud vs v_teacher (no audio)", fontsize=11,
                          color="#FFFFFF", weight="bold", pad=8)

    fig.suptitle(f"Cross-Modal Correlation Analysis — {split}", fontsize=13,
                 color="#FFFFFF", weight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(out, f"{split}_cross_modal_correlation.png"))
    return r_img_t, r_struct


def plot_self_similarity_heatmaps(z_img_mean, v_teacher, split, out, n=50):
    n = min(n, len(z_img_mean))
    sim_img = pairwise_cosine(z_img_mean, n)
    sim_t   = pairwise_cosine(v_teacher, n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title in zip(axes,
                              [sim_img, sim_t],
                              [f"z_img mean (CLIP) — first {n}",
                               f"v_teacher (ImageBind) — first {n}"]):
        im = ax.imshow(mat, cmap="magma", aspect="equal", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, color="#FFFFFF", weight="bold", pad=10)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Sample Index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Within-Modality Self-Similarity Heatmaps — {split}", fontsize=13,
                 color="#FFFFFF", weight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(out, f"{split}_self_similarity_heatmaps.png"))


def plot_pca(data_dict, audio_mask, split, out):
    z_img_mean = data_dict["z_img"].mean(dim=1)
    z_aud_valid = data_dict["z_aud"][audio_mask]
    v_teacher   = data_dict["v_teacher"]

    n_subplots = 3 + (1 if audio_mask.sum() > 10 else 0)
    fig = plt.figure(figsize=(6 * n_subplots, 5.5))
    gs  = gridspec.GridSpec(1, n_subplots, figure=fig)

    modalities = [
        ("z_img mean (CLIP 512d→2d)", z_img_mean, COLORS["z_img"]),
        ("v_teacher (ImageBind 1024d→2d)", v_teacher, COLORS["v_teacher"]),
    ]
    if audio_mask.sum() > 10:
        modalities.insert(1, ("z_aud (VGGish 128d→2d)", z_aud_valid, COLORS["z_aud"]))

    for idx, (title, tensor, col) in enumerate(modalities):
        ax = fig.add_subplot(gs[idx])
        proj, var_ratio = pca_2d(tensor)
        ax.scatter(proj[:, 0], proj[:, 1], c=col, alpha=0.55, s=12, edgecolors="none")
        ax.set_title(
            f"PCA: {title}\n(PC1={var_ratio[0]*100:.1f}%, PC2={var_ratio[1]*100:.1f}%)",
            fontsize=10, color="#FFFFFF", weight="bold", pad=8
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle="--", alpha=0.15)

    ax_scree = fig.add_subplot(gs[-1])
    k = 20
    for title_short, tensor, col in [
        ("z_img mean", z_img_mean, COLORS["z_img"]),
        ("v_teacher",  v_teacher,  COLORS["v_teacher"]),
    ]:
        ev = pca_scree(tensor, k=k)
        cumev = np.cumsum(ev)
        ax_scree.plot(range(1, k + 1), cumev * 100, color=col, marker="o",
                      markersize=4, linewidth=1.5, label=title_short)
    ax_scree.set_title("Cumulative PCA Explained Variance", fontsize=10,
                       color="#FFFFFF", weight="bold", pad=8)
    ax_scree.set_xlabel("# Principal Components")
    ax_scree.set_ylabel("Cumulative Explained Variance (%)")
    ax_scree.legend(facecolor="#1E1E1E", edgecolor="#333333", labelcolor="#E0E0E0", fontsize=8)
    ax_scree.grid(True, linestyle="--", alpha=0.2)
    ax_scree.set_xlim(1, k)

    fig.suptitle(f"PCA Projections & Scree — {split}", fontsize=14,
                 color="#FFFFFF", weight="bold", y=1.01)
    plt.tight_layout()
    savefig(fig, os.path.join(out, f"{split}_pca.png"))


def plot_train_test_comparison(train_d, test_d, train_mask, test_mask, out):
    def img_mean_norms(d):
        return torch.norm(d["z_img"].mean(dim=1), p=2, dim=1).numpy()

    def teacher_norms(d):
        return torch.norm(d["v_teacher"], p=2, dim=1).numpy()

    def inter_frame_cosine(d):
        return F.cosine_similarity(d["z_img"][:, 0, :], d["z_img"][:, 2, :], dim=1).numpy()

    metrics = [
        ("z_img mean L2 Norm",           img_mean_norms,    "z_img"),
        ("v_teacher L2 Norm",            teacher_norms,     "v_teacher"),
        ("Inter-frame cos (early-late)", inter_frame_cosine, "z_img"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (title, fn, key) in zip(axes, metrics):
        tr_vals = fn(train_d)
        te_vals = fn(test_d)
        lo = min(tr_vals.min(), te_vals.min())
        hi = max(tr_vals.max(), te_vals.max())
        bins = np.linspace(lo, hi, 50)
        ax.hist(tr_vals, bins=bins, alpha=0.65, color=COLORS["train"], label="Train", density=True)
        ax.hist(te_vals, bins=bins, alpha=0.65, color=COLORS["test"],  label="Test",  density=True)
        ax.set_title(title, fontsize=11, color="#FFFFFF", weight="bold", pad=8)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(facecolor="#1E1E1E", edgecolor="#333333", labelcolor="#E0E0E0")
        ax.grid(True, linestyle="--", alpha=0.2)

    fig.suptitle("Train vs Test Feature Distributions", fontsize=13,
                 color="#FFFFFF", weight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(out, "train_vs_test_comparison.png"))


def analyze_split(d, split, out):
    banner(f"ANALYZING {split.upper()}")
    z_img     = d["z_img"]
    z_aud     = d["z_aud"]
    v_teacher = d["v_teacher"]
    has_audio = d["has_audio"]
    video_ids = d["video_ids"]

    N = len(z_img)
    print(f"  Samples: {N}  |  Video IDs: {len(video_ids)}")
    print(f"  Audio coverage: {has_audio.sum().item()}/{N} ({100*has_audio.float().mean():.1f}%)")
    print()

    z_img_mean = z_img.mean(dim=1)
    norms = {
        "z_img":     print_stats("z_img (3-frame stack)", z_img.reshape(-1, 512)),
        "z_aud":     print_stats("z_aud (VGGish)",        z_aud),
        "v_teacher": print_stats("v_teacher (IB fused)",  v_teacher),
    }
    print_stats("z_img mean-pooled", z_img_mean)
    print()

    for i, lbl in enumerate(["early", "mid", "late"]):
        f = z_img[:, i, :]
        n = torch.norm(f, p=2, dim=1)
        print(f"  Frame {lbl:5s}: L2 {n.mean():.3f}+/-{n.std():.3f}")

    print("\n  Inter-frame cosine (mean +/- std):")
    for i, j, lbl in [(0, 1, "early<->mid"), (0, 2, "early<->late"), (1, 2, "mid<->late")]:
        s = F.cosine_similarity(z_img[:, i, :], z_img[:, j, :], dim=1)
        print(f"    {lbl}: {s.mean():.4f} +/- {s.std():.4f}  [min={s.min():.3f} max={s.max():.3f}]")

    aud_mask = has_audio
    norm_img_np = torch.norm(z_img_mean, p=2, dim=1).numpy()
    norm_t_np   = torch.norm(v_teacher, p=2, dim=1).numpy()
    r_norm = float(np.corrcoef(norm_img_np, norm_t_np)[0, 1])
    print(f"\n  Cross-modal L2 norm Pearson r (img_mean vs teacher): {r_norm:.4f}")

    n_sub = min(SIM_SUBSET, N)
    print(f"\n  Within-modality pairwise cosine (first {n_sub}):")
    for name, t in [("z_img mean", z_img_mean), ("v_teacher", v_teacher)]:
        mat = pairwise_cosine(t, n_sub)
        off = mat[~np.eye(n_sub, dtype=bool)]
        print(f"    {name:20s}: mean={off.mean():.4f}  std={off.std():.4f}  [{off.min():.3f}, {off.max():.3f}]")

    print(f"\n  Generating plots for {split}...")
    plot_l2_norms(
        {"z_img": torch.norm(z_img_mean, p=2, dim=1),
         "z_aud": torch.norm(z_aud, p=2, dim=1),
         "v_teacher": torch.norm(v_teacher, p=2, dim=1)},
        aud_mask, split, out
    )
    plot_value_histograms(d, aud_mask, split, out)
    plot_per_frame_norms(z_img, split, out)
    plot_inter_frame_cosine(z_img, split, out)
    plot_cross_modal_analysis(z_img, z_aud, v_teacher, aud_mask, split, out)
    plot_self_similarity_heatmaps(z_img_mean, v_teacher, split, out)
    plot_pca(d, aud_mask, split, out)


def main():
    data_dir = "data_V2_patched"
    out_dir  = os.path.join(data_dir, "analysis_plots")
    os.makedirs(out_dir, exist_ok=True)

    banner("V2-PATCHED FEATURE ANALYSIS")

    train_path = os.path.join(data_dir, "train_features_v2_patched.pt")
    test_path  = os.path.join(data_dir, "test_features_v2_patched.pt")

    missing = [p for p in [train_path, test_path] if not os.path.exists(p)]
    if missing:
        print(f"Missing files: {missing}")
        return

    train_d = load(train_path)
    test_d  = load(test_path)

    train_mask = train_d["has_audio"]
    test_mask  = test_d["has_audio"]

    # audio coverage comparison
    plot_audio_coverage({"train": train_mask, "test": test_mask}, out_dir)

    analyze_split(train_d, "train", out_dir)
    analyze_split(test_d,  "test",  out_dir)

    banner("TRAIN vs TEST COMPARISON")
    plot_train_test_comparison(train_d, test_d, train_mask, test_mask, out_dir)

    banner("DONE")
    print(f"  All plots saved to: {os.path.abspath(out_dir)}")
    for f in sorted(os.listdir(out_dir)):
        print(f"    {f}")


if __name__ == "__main__":
    main()
