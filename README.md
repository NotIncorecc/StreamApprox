# Multimodal Embedding Approximator

This project implements an efficient **multimodal embedding approximator** designed for video retrieval. Running dense temporal video models (extracting features from dozens of frames across a video timeline) is computationally expensive. This project solves that bottleneck by approximating the high-quality embeddings of full-video models using only sparse inputs.

By independently extracting features from a **single video frame** (via CLIP) and a **short audio segment** (via VGGish), we train a lightweight fusion network (the "student") to map these sparse inputs into the dense embedding space of a heavy video encoder (the "teacher", e.g., ImageBind).

The goal is to achieve nearly the same semantic retrieval quality (Recall@K) as dense video processing, but with significantly lower latency, fewer parameters, and reduced FLOPs during inference.

### Key Features
- **Efficient Feature Extraction:** Utilizes off-the-shelf, frozen encoders (CLIP for vision, VGGish for audio) to generate unimodal embeddings cheaply.
- **Cross-Modal Distillation:** Trains a student model to predict dense temporal embeddings using contrastive alignment (InfoNCE) and regression.
- **Multiple Fusion Strategies:** Compares naive late fusion (MLP) against more advanced methods like Transformer-based cross-attention.
- **Retrieval Evaluation:** Benchmarked on standard metrics like Recall@K and Median Rank, demonstrating high-efficiency video retrieval without dense temporal sampling.

---

## Repository Structure

```
StreamApprox/
│
├── src/                              # Core Python library
│   ├── models.py                     # MLP and Transformer student model definitions
│   ├── loss.py                       # InfoNCE + cosine regression loss functions
│   ├── metrics.py                    # Recall@K, Median Rank evaluation metrics
│   ├── dataset.py                    # Dataset class for (CLIP, VGGish, ImageBind) triplets
│   ├── remote_runner.py              # Helper to launch training on remote/Kaggle kernels
│   └── sync_workspace.py             # Utility for syncing files with remote environments
│
├── notebooksV2/                      # Final end-to-end pipeline notebooks (run in order)
│   ├── 1_feature_extraction.ipynb    # Extract CLIP (video) + VGGish (audio) + ImageBind features
│   ├── 2_audio_patch_v2.ipynb        # Patch incomplete audio coverage in extracted features
│   ├── 2_training_mlp.ipynb          # Train MLP student (cosine + InfoNCE loss variants)
│   ├── 3_training_transformer.ipynb  # Train Transformer student (cross-attention fusion)
│   └── 4_evaluation.ipynb            # Evaluate all models; generate retrieval + latency plots
│
├── kaggle-mlp-infoNCE-training-opt/  # Final optimized MLP training (run on Kaggle GPU)
│   ├── multimodel-baseline-ml-training-opt.ipynb   # Optimized training with regularization
│   ├── multimodel-baseline-ml-training-opt1.ipynb  # Fine-tuning on best InfoNCE checkpoint
│   └── *.pt                          # Saved best/final checkpoints (cosine, InfoNCE, tuned)
│
├── kaggle-transformer-training-opt/  # Final optimized Transformer training (run on Kaggle GPU)
│   ├── multimodal-transformer-opt.ipynb   # Optimized training with corrected normalization
│   ├── multimodal-transformer-opt1.ipynb  # Final run with best hyperparameters
│   └── *.pt                               # Saved best/final checkpoints
│
├── models/                           # Best model checkpoints for inference
│   ├── mlp_cosine/mlp_cosine.pt      # MLP trained with cosine regression loss
│   ├── mlp_infonce/mlp_infonce.pt    # MLP trained with InfoNCE contrastive loss (best MLP)
│   └── transformer_fusion/           # Transformer cross-attention fusion model
│       └── transformer_fusion.pt
│
├── evaluation_V2/                    # Final evaluation outputs
│   ├── final-evaluations.ipynb       # Full evaluation notebook with all comparisons
│   ├── accuracy_results_v2.csv       # Recall@1/5/10 and Median Rank for all models
│   ├── retrieval_comparition.png     # Recall@K bar chart across models
│   ├── latency_comparition.png       # Inference latency comparison
│   ├── cosine_similiarity.png        # Cosine similarity distributions
│   ├── embeddings_pca.png            # PCA projection of embedding spaces
│   └── teacher_vs_pred_overlay.png   # Teacher vs. student embedding overlay
│
├── dataV2/                           # V2 extracted features (pre-patch)
│   ├── train_features_v2.pt          # Training set: CLIP + VGGish + ImageBind tensors
│   ├── test_features_v2.pt           # Test set: CLIP + VGGish + ImageBind tensors
│   └── analysis_plots/               # Distribution and PCA plots for V2 features
│
├── data_V2_patched/                  # Final features with audio coverage fix applied
│   ├── train_features_v2_patched.pt  # Patched training features (used for all final training)
│   ├── test_features_v2_patched.pt   # Patched test features
│   └── analysis_plots/               # Distribution plots confirming the patch
│
├── scripts/                          # Utility and data preparation scripts
│   ├── patch_audio_features.py       # Patches missing/zero audio frames in feature tensors
│   ├── analyze_features_v2.py        # Statistical analysis of V2 feature tensors
│   ├── analyze_features_v2_patched.py# Same analysis after patching
│   ├── analyze_features.py           # General feature analysis helper
│   ├── analyze_train_features.py     # Train-split specific analysis
│   ├── sync_models_kaggle.py         # Syncs trained .pt files from Kaggle back to local
│   ├── remote_setup_and_patch.py     # Sets up environment and applies audio patch on remote
│   ├── unzip_and_patch.py            # Unzips Kaggle dataset and applies audio patch
│   └── create_notebooks.py           # Script used to scaffold initial notebook templates
│
├── archive/                          # Superseded files kept for reference (not part of pipeline)
│   ├── v1_notebooks/                 # Original V1 notebooks (before audio patch + optimization)
│   ├── v1_features/                  # V1 extracted feature tensors + analysis plots
│   ├── v1_training/                  # Non-optimized Kaggle training notebooks + checkpoints
│   ├── v1_models/                    # First-run model checkpoints
│   ├── old_results/                  # Old accuracy CSVs and plots from early experiments
│   └── dev_notes/                    # Planning docs, day logs, and agent conversation logs
│
├── REPORT.md                         # Full written project report
├── REPORT.pdf                        # PDF version of the report
├── Multi-Modal AI.pdf                # Reference paper (Zhu et al., StreamApprox basis)
└── requirements.txt                  # Python dependencies
```

---

## Quickstart

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the pipeline end-to-end using the notebooks in `notebooksV2/` in order (1 → 4). Training notebooks (`2_training_mlp`, `3_training_transformer`) were run on Kaggle GPU kernels — the corresponding Kaggle notebooks with saved checkpoints are in `kaggle-mlp-infoNCE-training-opt/` and `kaggle-transformer-training-opt/`.

To evaluate pre-trained models directly, open `evaluation_V2/final-evaluations.ipynb` and point it at the checkpoints in `models/`.
