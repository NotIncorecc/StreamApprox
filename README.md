# Multimodal Embedding Approximator

This project implements an efficient **multimodal embedding approximator** designed for video retrieval. Running dense temporal video models (extracting features from dozens of frames across a video timeline) is computationally expensive. This project solves that bottleneck by approximating the high-quality embeddings of full-video models using only sparse inputs.

By independently extracting features from a **single video frame** (via CLIP) and a **short audio segment** (via VGGish), we train a lightweight fusion network (the "student") to map these sparse inputs into the dense embedding space of a heavy video encoder (the "teacher", e.g., ImageBind). 

The goal is to achieve nearly the same semantic retrieval quality (Recall@K) as dense video processing, but with significantly lower latency, fewer parameters, and reduced FLOPs during inference.

### 🌟 Key Features
- **Efficient Feature Extraction:** Utilizes off-the-shelf, frozen encoders (CLIP for vision, VGGish for audio) to generate unimodal embeddings cheaply.
- **Cross-Modal Distillation:** Trains a student model to predict dense temporal embeddings using contrastive alignment (InfoNCE) and regression.
- **Multiple Fusion Strategies:** Compares naive late fusion (MLP) against more advanced methods like Transformer-based cross-attention.
- **Retrieval Evaluation:** Benchmarked on standard metrics like Recall@K and Median Rank, demonstrating high-efficiency video retrieval without dense temporal sampling.
