Use **MSR-VTT** and structure the whole project around **“teacher video embedding (dense temporal) → student predicts from (1 frame + audio)”** with two lightweight students (MLP vs tiny Transformer) plus an InfoNCE variant. On Kaggle/Colab this is feasible by **June 3, 2026 11:59** if you aggressively cache features and avoid decoding videos during training.

Below is a concrete, minimal-risk implementation plan with specific model choices and what to measure.

---

## 1) Dataset choice (fast + standard)
### Recommended: **MSR-VTT (video–text retrieval)**
Why it’s ideal for your deadline:
- widely accepted benchmark
- standard retrieval metrics (R@K, MedR) are common there
- you can treat **teacher video embedding** as “ground truth”
- you can use **text** as an additional evaluation/query modality if needed (optional)

### Practical note on audio
MSR-VTT videos contain audio, but extracting it can be annoying/slow. Your strategy should be:

- **Extract everything once**, store as `.pt` / `.npy` on Kaggle dataset storage or Google Drive
- During training, load cached embeddings only (no ffmpeg, no decoding)

If audio extraction becomes a bottleneck, you can:
- use **short audio** (e.g., first 5–10 seconds) to speed up extraction
- or downsample to 16 kHz mono

---

## 2) Define “ground truth” properly (teacher)
You need a teacher that uses **temporal frames** (dense-ish temporal processing).

### Teacher option A (recommended): **ImageBind video embedding**
- Teacher: `v_teacher = ImageBind(video)`
- Student input: `z_img = CLIP(frame)`, `z_aud = AST(audio)` (or ImageBind audio/image too, but then student = ImageBind-lite; still okay, but less “independent encoders” feel)

Pros: ImageBind naturally aligns video/audio/image; strong semantics.
Cons: Running ImageBind video encoder may be heavy, but if you do it offline once, it’s fine.

### Teacher option B (safer if ImageBind video is too heavy): a smaller pretrained video model (8–16 frames)
Pick any pretrained video encoder that takes multiple frames (8–16) and outputs a vector. Cache once.

Pros: clearer “dense temporal baseline” vs your 1-frame method.
Cons: you must integrate another model.

**Given your time, I’d try Teacher A first**. If it’s too slow, fall back to Teacher B.

---

## 3) What you will actually train (two methods + baselines)

### Baseline 1 (must-have): Naive late fusion + MLP regression
Inputs:
- `z_img` from frozen image encoder (CLIP ViT-B/32 is fine)
- `z_aud` from frozen audio encoder (AST is strong; VGGish is simpler)

Model:
- `h = concat(z_img, z_aud)`
- `v_pred = MLP(h) -> d_teacher`
- normalize `v_pred` and `v_teacher`

Loss:
- `L = 1 - cosine(v_pred, v_teacher)` (simple, stable)

### Method 1 (counts as distinct): **InfoNCE contrastive alignment** to teacher
Same backbone (MLP or slightly deeper MLP), but train with InfoNCE:

For a batch of size B:
- similarity matrix `S_ij = cos(v_pred_i, v_teacher_j) / tau`
- `L = cross_entropy(S, targets=diag_indices)`

This usually improves retrieval because it’s directly optimizing ranking.

### Method 2 (counts as distinct): **Tiny Transformer multimodal fusion**
Inputs:
- treat `z_img` and `z_aud` as tokens
- add learnable `[CLS]` token
- 2-layer Transformer encoder (very small)
- output CLS → linear → teacher dim

Loss:
- either cosine regression, or cosine + InfoNCE (best)

This satisfies “Transformer-based multimodal fusion” explicitly.

### Baseline 2 (benchmark): ImageBind / AudioCLIP zero-shot
If you use ImageBind as teacher, you still report:
- ImageBind’s own **(image+audio)** embedding ability (if available) or compare image-only/audio-only/video embeddings in its space for retrieval.
If you don’t use it as teacher, then it’s a clean “upper bound” benchmark.

---

## 4) Evaluation you can finish on time (and that will look correct)

### A) Latent proximity
Compute on test set:
- mean cosine similarity: `mean cos(v_pred, v_teacher)`
- optionally distribution plot (histogram)

If you do regression with MSE, also report MSE, but cosine is enough and aligns with retrieval.

### B) Retrieval (the key)
You have a choice of retrieval protocol:

**Protocol 1 (cleanest for this project):**
- Query = `v_pred` (from frame+audio)
- Gallery = `v_teacher` for all test videos
- Similarity = cosine
- Compute R@1, R@5, R@10, MedR

This directly answers: “can my cheap embedding retrieve the correct video semantics in the teacher space?”

**Protocol 2 (optional): text-to-video retrieval**
If you also compute text embeddings (CLIP text or teacher text encoder), you can show:
- text query → retrieve student embeddings vs teacher embeddings
But don’t add this unless you have time.

### C) Efficiency (must report something concrete)
Measure:
- student inference latency for one sample (frame+audio embeddings already computed OR include encoders—be explicit which)
- teacher inference latency (full video model on N frames)
- parameter count of fusion head
- (optional) FLOPs; if too hard, skip FLOPs but be honest

**Important:** Be explicit about what you time:
- End-to-end including encoders? or only fusion head?
Given the project goal, it’s more convincing to time:
- Teacher: video decoding + N-frame video encoder
- Student: 1 frame encoder + audio encoder + fusion head  
But if decoding dominates on Colab, time “model forward only” with cached tensors and mention it.

---

## 5) What to do in the next 6 days (specific tasks)

### Today (May 28): lock design + start extraction notebook
- Create Kaggle notebook:
  - download MSR-VTT
  - extract 1 frame/video (middle frame)
  - extract audio waveform (first 10s)
- Verify you can load 100 videos end-to-end without crashing

### May 29: cache embeddings (most important)
- Compute and save:
  - `z_img` for all videos (train/val/test)
  - `z_aud` for all videos
  - `v_teacher` for all videos
- Save as one file per split: `train.pt`, `val.pt`, `test.pt` with dict:
  - video_id, z_img, z_aud, v_teacher

### May 30: train Baseline 1 + eval
- Train MLP cosine regression
- Evaluate retrieval metrics + latent cosine

### May 31: train Method 1 (InfoNCE)
- Implement InfoNCE with batch negatives
- Evaluate and compare to baseline

### June 1: train Method 2 (tiny Transformer)
- Evaluate
- Collect efficiency numbers

### June 2–June 3: run ImageBind benchmark + write report + cleanup
- ImageBind as benchmark (or as teacher if you chose it)
- finalize plots/tables
- ensure repo reproducible: `extract.py`, `train.py`, `eval.py`, config

---

## 6) Two decisions you must make now (to avoid wasting time)

### Decision A: Teacher choice
Pick one:
1) **ImageBind video embedding** as teacher (strong, aligned, but might be heavier)
2) **Any 8–16 frame pretrained video encoder** as teacher (lighter maybe, but integration work)

If you tell me which direction you prefer, I can suggest an exact teacher model and the exact embedding dims you’ll get.

### Decision B: Audio encoder choice (speed vs quality)
- **VGGish**: simpler, older, often easiest
- **AST**: stronger, but heavier

If Kaggle/Colab time is tight, start with **VGGish** for speed; you can always swap to AST later.

---

## Quick recommendation given your hardware/time
- Use Colab/Kaggle GPU for extraction + training.
- Use **1 frame** + **first 10 seconds audio**.
- Start with **VGGish + CLIP + MLP** baseline (get a working pipeline fast).
- Then add **InfoNCE** (almost no extra compute).
- Then add **tiny Transformer** (small extra compute).
- Try ImageBind as benchmark last; don’t let it block your core deliverables.

---

## Tell me these 3 things and I’ll finalize exact model picks + code structure
1) Do you prefer **ImageBind as teacher**, or would you rather use a “classic” video model teacher with 8–16 frames?  
2) Are you okay using **VGGish** first (faster), or do you want **AST** from the start?  
3) Do you need the solution to run fully on **Kaggle**, or is **Colab** fine for extraction and Kaggle for training (or vice versa)?