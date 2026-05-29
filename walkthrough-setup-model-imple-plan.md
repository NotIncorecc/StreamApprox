# Walkthrough: Kaggle GPU Setup & Multimodal Approximator Environment

We have successfully set up the local Python virtual environment, implemented the core modeling blocks, generated the four sequential Jupyter notebooks, and created utility scripts to sync files back and forth with Kaggle.

Here is the summary of what has been built and how to use it.

---

## 1. Project Structure

The project has been organized into modular Python files and sequential Jupyter notebooks:

- **Local Python Environment**: Activated under `.venv/` with all packages in `requirements.txt` installed.
- **Source Modules (`src/`)**:
  - [models.py](file:///d:/Multimodal-approximator/src/models.py): Implements `MLPApproximator` (Baseline 1 & Method A) and `TransformerFusionApproximator` (Method B).
  - [loss.py](file:///d:/Multimodal-approximator/src/loss.py): Implements symmetric InfoNCE contrastive loss.
  - [dataset.py](file:///d:/Multimodal-approximator/src/dataset.py): PyTorch dataset class for cached multimodal features.
  - [metrics.py](file:///d:/Multimodal-approximator/src/metrics.py): Logic for calculating cosine similarities, MSE, Recall@K, and Median Rank (MedR).
- **Sequential Notebooks (`notebooks/`)**:
  - [1_feature_extraction.ipynb](file:///d:/Multimodal-approximator/notebooks/1_feature_extraction.ipynb): Remote environment setup, MSR-VTT dataset download/unzipping from Hugging Face, pre-trained encoder extraction (CLIP, VGGish, ImageBind), and feature caching.
  - [2_training_mlp.ipynb](file:///d:/Multimodal-approximator/notebooks/2_training_mlp.ipynb): Trains Baseline 1 (MLP + Cosine) and Method A (MLP + InfoNCE).
  - [3_training_transformer.ipynb](file:///d:/Multimodal-approximator/notebooks/3_training_transformer.ipynb): Trains Method B (Tiny Transformer + Cosine/InfoNCE joint loss).
  - [4_evaluation.ipynb](file:///d:/Multimodal-approximator/notebooks/4_evaluation.ipynb): Compares latent proximity, retrieval accuracy (Recall@K, MedR), and inference latencies/footprints.
- **Synchronization Scripts**:
  - [remote_runner.py](file:///d:/Multimodal-approximator/src/remote_runner.py): Utility to execute inline python code or scripts on the remote Kaggle kernel.
  - [sync_workspace.py](file:///d:/Multimodal-approximator/src/sync_workspace.py): Tool to upload your local `src/` files and download trained assets (weights & features) from Kaggle.

---

## 2. Interactive Notebook Execution in VS Code

To run the notebooks on Kaggle's free GPU instances from your local VS Code:

1. **Connect Kernel**:
   - Open any notebook (e.g., `notebooks/1_feature_extraction.ipynb`) in VS Code.
   - Click on **Select Kernel** in the top-right corner.
   - Choose **Existing Jupyter Server**.
   - Paste the Kaggle connection URL you copied:
     ```text
     https://kkb-production.jupyter-proxy.kaggle.net/k/322999328/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..RWVOFOfFq8SNf9pNEAZBhw.sHXCgKpsUAke0rvUNYpQ3lWSW8uotciUtuchmDJZyGWaTPHGA6BvVK9ogtg9JhbAjNlX1qzdwuYiIy6V8p5kXtyB_LIrnOhKDOiisGqcKEHP7FoIE6DDC-s2-lq7XyhpCNqYv_e6mgWvrV80_5jH0SQDcM1d1qddv7PF42vRiXUVFj9ubAtHA5H4DijSfzB134lL3nyNIetmGIsp-ax8eSaHLrWn7XwisZBJlNGLB6Pr544OZIu0n1R3xqFfkGDp.wmkg8mB23Yx2ymuqGm2GZQ/proxy
     ```
   - Press **Enter** to connect.
2. **Execute Cells**: Run the cells sequentially. The code (including heavy video extraction and model training) will execute on the remote Tesla T4 GPUs.

---

## 3. Uploading Code & Downloading Assets

Since Kaggle container storage is ephemeral and expires when the session is idle, use `sync_workspace.py` to synchronize files back and forth:

> [!NOTE]
> All local source code files in `src/` have already been synchronized to your active Kaggle kernel!

### How to use `sync_workspace.py`

When running these commands, replace `<KAGGL_URL>` with your current Kaggle proxy URL.

* **Upload updated local source code** (run if you make edits to `models.py`, `loss.py`, etc.):
  ```bash
  .venv\Scripts\python.exe src/sync_workspace.py --url "<KAGGL_URL>" --action upload_src
  ```

* **Download cached features** (run after Notebook 1 completes to save the extracted `.pt` tensors locally):
  ```bash
  .venv\Scripts\python.exe src/sync_workspace.py --url "<KAGGL_URL>" --action download_features
  ```
  This downloads `train_features.pt` and `test_features.pt` to a local `data/` directory.

* **Download trained model weights** (run after Notebook 2 & 3 finish training):
  ```bash
  .venv\Scripts\python.exe src/sync_workspace.py --url "<KAGGL_URL>" --action download_models
  ```
  This downloads `mlp_cosine.pt`, `mlp_infonce.pt`, and `transformer_fusion.pt` to a local `models/` directory.
