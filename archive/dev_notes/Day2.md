Fixed an error in the feature_extraction.ipynb notebook.

The issue occurs because of a mismatch in the directory structure of the extracted Hugging Face dataset zip file:

Expected Path: The notebook was configured to look for videos inside msrvtt/MSRVTT_Videos/ (e.g. msrvtt/MSRVTT_Videos/video7020.mp4).
Actual Path: The MSRVTT_Videos.zip file extracts its videos into a nested folder named video in lowercase (e.g. msrvtt/video/video7020.mp4).
Outcome: Because the directory msrvtt/MSRVTT_Videos didn't exist, the loop skipped all 1,000 video files during processing. The lists of extracted features (z_imgs, z_auds, v_teachers) remained empty, causing torch.stack to raise: RuntimeError: stack expects a non-empty TensorList

But still:

Extracting features for 1000 videos from msrvtt/msrvtt_test_1k.json...
  1%|          | 9/1000 [00:05<07:10,  2.30it/s]  
Error processing video7021: 'Tensor' object has no attribute 'asnumpy'
Error processing video7024: 'Tensor' object has no attribute 'asnumpy'
Error processing video7025: 'Tensor' object has no attribute 'asnumpy'

Turns out:

During the loading/initialization of the model libraries (such as imagebind), the environment sets decord's framework bridging system to PyTorch: decord.bridge.set_bridge('torch')

This configures VideoReader to return PyTorch Tensor objects directly instead of its native NDArray objects. Since PyTorch Tensor objects do not have the .asnumpy() method (only .numpy() or .cpu().numpy()), calling .asnumpy() raises the error: 'Tensor' object has no attribute 'asnumpy'

Fixed this by Modifying the extract_clip_frame function definition inside 
scripts/create_notebooks.py
 to check for the .asnumpy attribute and fallback to .cpu().numpy() if the frame is a PyTorch tensor.

Executed the script create_notebooks.py to write the corrected code into the notebooks.

Step 5: Define Extraction Helper Functions (Updated extract_clip_frame)

```python
def extract_clip_frame(video_path):
    # Reads the middle frame and passes it through CLIP Vision
    vr = VideoReader(video_path, ctx=cpu(0))
    mid_idx = len(vr) // 2
    frame = vr[mid_idx]
    if hasattr(frame, 'asnumpy'):
        frame = frame.asnumpy()
    else:
        frame = frame.cpu().numpy()
    pil_img = Image.fromarray(frame)
    img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(img_tensor)
    return feat.squeeze(0).cpu()
```

After the test_features.pt was generated, I decided to do a gut check on the embeddings, to see if they were actually generating meaningful information in
`scripts/analyze_features.py`. 

I found something that could have wasted my entire 1 hr of training
Analysis showed that ***100%*** of the vectors in z_aud are ***all-zeros***.

Possible issues:
- Missing System Libraries: soundfile requires the Ubuntu library libsndfile1 to read WAV files. If it is missing, vggish.forward throws a silent OSError.
- Missing ffmpeg: If ffmpeg is missing, the subprocess fails to write temp_audio.wav.

The Student visual frame embedding spaces and the teacher video embedding space look healthy and well-distributed, confirming that our dimensions match. You can inspect the distribution histograms, PCA projections, and similarity heatmaps in `features/analysis_plots_test_features`

But I had an idea, that we can just let the training continue with the empty audio embeddings and just patch the audio embeddings later. Gemini said that the audio embeddings are faster to generate, so we can just patch them later. Gemini also gave me some code for it. 

Python Code to Repair test_features.pt on Kaggle:
 (after installing libsndfile1 and ffmpeg and making sure your VGGish encoder is loaded):

```python
import os
import json
import torch
from tqdm import tqdm

# Ensure ffmpeg and libsndfile1 are installed first
# !apt-get update && apt-get install -y ffmpeg libsndfile1

# 1. Load the existing file
original_path = "test_features.pt"
if not os.path.exists(original_path):
    raise FileNotFoundError(f"Could not find {original_path}")

data = torch.load(original_path, map_location="cpu")
print(f"Loaded {original_path}. Shape of z_img: {data['z_img'].shape}")

# 2. Get video-ID to filename mappings from the JSON
with open("msrvtt/msrvtt_test_1k.json") as f:
    test_json = json.load(f)

video_id_to_filename = {}
for item in test_json:
    video_id_to_filename[item['video_id']] = item['video']

# 3. Extract ONLY audio embeddings in the exact order of original video_ids
repaired_z_aud = []
print("Extracting VGGish audio embeddings for test split...")

for vid in tqdm(data['video_ids']):
    video_file = os.path.join("msrvtt/video", video_id_to_filename[vid])
    
    # Uses your extract_vggish_audio function (make sure it's loaded)
    aud_emb = extract_vggish_audio(video_file) 
    repaired_z_aud.append(aud_emb)

# 4. Replace the old silent audio embeddings
data['z_aud'] = torch.stack(repaired_z_aud)

# 5. Save the repaired file
torch.save(data, "test_features.pt")
print("Repair completed successfully!")
print(f"New z_aud shape: {data['z_aud'].shape}")
```

Gemini said that it is actually more efficient to let 7k videos generations finish and then patch the audio rather than stopping and restarting from scratch.

Here is the math:

If you let it finish + patch (Recommended):
Remaining extraction time: ~3.4 hours.
Audio-only patch time: ~15 minutes.
Total GPU time from now: ~3.65 hours.
Why: You preserve the 1,111 videos already processed (which took ~40 minutes of GPU time). Since CLIP and ImageBind are the bottlenecks, running them once and post-patching the fast audio features is the fastest route.
If you stop and restart from scratch:
The 1,111 videos are lost because they are inside a function scope and haven't been saved to disk yet.
Fresh run extraction time: ~4.3 hours.
Total GPU time from now: ~4.3 hours.

Python Code to Patch train_features.pt (7K Videos):
This is the same code adjusted for the training file. Once the 7k run finishes, create a cell, install the dependencies, and run this:

```python
import os
import json
import torch
from tqdm import tqdm

# 1. Ensure audio system dependencies are installed
# !apt-get update && apt-get install -y ffmpeg libsndfile1

# 2. Load the train features
original_path = "train_features.pt"
if not os.path.exists(original_path):
    raise FileNotFoundError(f"Could not find {original_path}")

data = torch.load(original_path, map_location="cpu")
print(f"Loaded {original_path}. Shape of z_img: {data['z_img'].shape}")

# 3. Get video-ID to filename mappings from the 7k training JSON
with open("msrvtt/msrvtt_train_7k.json") as f:
    train_json = json.load(f)

video_id_to_filename = {}
for item in train_json:
    video_id_to_filename[item['video_id']] = item['video']

# 4. Extract ONLY audio embeddings
repaired_z_aud = []
print("Extracting VGGish audio embeddings for train split...")

for vid in tqdm(data['video_ids']):
    video_file = os.path.join("msrvtt/video", video_id_to_filename[vid])
    aud_emb = extract_vggish_audio(video_file) # Ensure extract_vggish_audio is defined
    repaired_z_aud.append(aud_emb)

# 5. Overwrite the old audio tensor and save
data['z_aud'] = torch.stack(repaired_z_aud)
torch.save(data, "train_features.pt")

print("Train features repair completed successfully!")
print(f"New z_aud shape: {data['z_aud'].shape}")
```

Now I can try audio patching the test_features.pt now, but unsure if I can do this on the same kaggle kernel, as it is generating the 7k videos. I should work on different kaggle account then