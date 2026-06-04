You have **6 days** (from **Thu, May 28, 2026** to **Wed, June 3, 2026 11:59**). With that timeline, you should optimize for: **(a) a clean, reproducible pipeline, (b) 2 methods + 2 baselines, (c) strong evaluation tables**, and avoid anything that requires heavy training or large-scale video decoding at runtime.

Below is a realistic “ship-it” plan that can still look serious.

---

## Strategy (what to simplify without failing the spec)

### Pick a dataset that is easy + standard for retrieval
Choose **MSR-VTT** (text–video retrieval). It’s widely used, has standard splits, and you can:
- treat **text** as an evaluation/query modality (very convenient)
- extract **one keyframe** per video (e.g., middle frame or precomputed frame)
- extract **audio** from the video (or if that’s too slow, use short audio segments)

If MSR-VTT audio is painful in your environment, an alternative is:
- a dataset where audio is already available as files, or
- use an already-preprocessed MSR-VTT variant / features (acceptable if documented)

### Define a single teacher “ground-truth video embedding”
To satisfy “dense temporal baseline”, do this:

- **Teacher (GT):** a **pretrained video-text model** that uses multiple frames (dense temporal).  
  You compute and cache `E_video_teacher(video)` offline once.

Then your student predicts this teacher embedding from `(frame + audio)`.

If you can’t run a true multi-frame model fast enough, you can still be transparent:
- use “N-frame teacher” (e.g., 8 frames) as dense temporal baseline
- even 8 frames qualifies as temporal processing compared to 1 frame

### Keep encoders frozen
- Image encoder frozen (CLIP vision)
- Audio encoder frozen (AST or VGGish)
- Train only a small fusion head (MLP / small Transformer)

This is the fastest path.

---

## Minimum required models (to meet the “two methods” + baselines)

You can frame it like this:

### Baseline 1 (required): Naive Late Fusion + MLP regression
- `z_img = CLIP(image)`
- `z_aud = AST(audio)`
- `z = MLP([z_img || z_aud])`
- Train with **cosine loss** to teacher: maximize cos(z, v_teacher) (or use MSE on normalized vectors)

### Method A: Contrastive alignment (InfoNCE) to teacher embeddings
Same architecture as baseline (or slightly better MLP), but train with **InfoNCE**:
- positives: (z_pred for video i, v_teacher for video i)
- negatives: (z_pred for video i, v_teacher for other videos in batch)

This often improves retrieval without heavy architecture changes.

### Method B: Tiny Transformer fusion (or gated fusion) + distillation loss
To keep it fast, do one of these:

**Option 1 (fastest “Transformer-based fusion” that still counts):**
- Treat `z_img` and `z_aud` as 2 tokens (+ a [CLS] token)
- 2-layer Transformer encoder
- output CLS → projected to teacher dim
- loss = cosine + (optional) InfoNCE

**Option 2 (even faster, arguably more effective than 2-token Transformer):**
- “Gated fusion”: learn weights α = sigmoid(W[z_img||z_aud])  
  then `z = α * z_img_proj + (1-α) * z_aud_proj`
- plus MLP to teacher dim
- loss = cosine + InfoNCE

If you’re worried about whether “gated fusion” counts as a distinct methodology, pick the **tiny Transformer**, because it matches the wording exactly.

### Baseline 2 (required benchmark): ImageBind (zero-shot)
- Use ImageBind to compute:
  - `v_imgbind_video(video)` (teacher-like)
  - `v_imgbind_img(frame)` and `v_imgbind_audio(audio)`
- Evaluate retrieval using ImageBind’s own embeddings (or its fused approach if available)
- You do **not** train it; you just report it as a reference.

Even if you can’t run ImageBind end-to-end due to environment, you can still try—just don’t leave it to the last day. If it fails, be honest and report that you couldn’t run it; but ideally you do run it.

---

## Concrete 6-day schedule (what to do each day)

### Day 1 — Thu, May 28: Lock scope + dataset + skeleton repo
- Decide dataset (recommend: **MSR-VTT**)
- Create repo structure:
  - `extract/` (feature extraction)
  - `train/` (training scripts)
  - `eval/` (metrics, retrieval)
  - `configs/`
- Implement retrieval metrics: R@1/5/10, MedR
- Write README “how to run” with placeholders

### Day 2 — Fri, May 29: Feature extraction (cached)
- Extract and cache:
  - frame embedding (CLIP)
  - audio embedding (AST/VGGish)
  - teacher video embedding (multi-frame model)
- Save to disk as one file per split (train/val/test) with IDs

Goal: by end of day, you can train without touching raw video again.

### Day 3 — Sat, May 30: Baseline 1 training + evaluation
- Train MLP regression baseline
- Evaluate:
  - cosine similarity to teacher
  - retrieval against teacher gallery (student query → teacher gallery)
  - latency of student inference (no teacher)

Make the first result table.

### Day 4 — Sun, May 31: Method A (InfoNCE) + Method B (tiny Transformer)
- Train contrastive (InfoNCE) version
- Train Transformer fusion version
- Evaluate the same metrics, fill tables

### Day 5 — Mon, June 1: ImageBind benchmark + efficiency numbers
- Run ImageBind baseline
- Measure:
  - student latency vs teacher latency
  - params for student heads
  - optionally FLOPs (if too hard, report params + latency clearly)

### Day 6 — Tue, June 2 (and morning of Wed, June 3): Report + cleanup
- Write report with:
  - problem statement
  - methods
  - training details
  - results tables (proximity + retrieval + efficiency)
  - discussion: where it works/fails
- Clean code, add scripts, add fixed seeds, add “download + preprocess” notes
- Ensure you can reproduce from scratch (or from cached features)

---

## What I need from you to make this plan *fully concrete* (no guessing)
Answer these and I’ll tell you exactly what models/tools to use:

1) **Do you have a GPU?** If yes, which one (even roughly: T4/3060/3090/A100)?  
2) **Can you use MSR-VTT**, or did they specify another dataset?  
3) What environment: **Colab / local / Kaggle / university server**?  
4) Are you allowed to use **pretrained models** freely (CLIP/AST/ImageBind), and is internet access allowed during training?

Once you reply, I’ll propose:
- exact encoders (with dimensions),
- the simplest teacher model that still counts as “dense temporal”,
- exact loss formulas,
- and the minimum set of experiments that will look credible by **June 3, 2026 11:59**.