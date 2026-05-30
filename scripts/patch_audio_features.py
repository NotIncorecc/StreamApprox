import os
import sys
import json
import uuid
import torch
import subprocess
from tqdm import tqdm

def install_system_dependencies():
    """
    Checks for ffmpeg and installs system-level dependencies on Linux/Kaggle.
    """
    if sys.platform.startswith("linux"):
        print("Running on Linux/Kaggle. Checking system-level dependencies...")
        
        # Check ffmpeg
        ffmpeg_installed = subprocess.run("which ffmpeg", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not ffmpeg_installed:
            print("Installing ffmpeg and libsndfile1 (required for VGGish)...")
            try:
                subprocess.run("apt-get update -y && apt-get install -y ffmpeg libsndfile1", shell=True, check=True)
                print("System dependencies installed successfully.")
            except Exception as e:
                print(f"Warning: Failed to install system dependencies: {e}")
                print("Make sure you are running as root, or install ffmpeg and libsndfile1 manually.")
        else:
            print("ffmpeg is already installed.")
    else:
        print(f"Running on {sys.platform}. Please ensure ffmpeg is installed and added to your system PATH.")

def extract_vggish_audio(vggish_model, video_path, device="cpu"):
    """
    Extracts audio from video path and computes 128-dim VGGish embedding.
    """
    # Create a unique temp file name to avoid concurrency issues
    temp_wav = f"temp_audio_{os.getpid()}_{uuid.uuid4().hex[:8]}.wav"
    if os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
        except OSError:
            pass
            
    # Resample to 16kHz mono using ffmpeg
    cmd = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{temp_wav}"'
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # If ffmpeg failed, return zero vector
    if res.returncode != 0:
        # Check if it was because there was no audio stream in the file
        if "Output file is empty" in res.stderr or "Does not contain any stream" in res.stderr:
            # Silent video, return zero vector without logging as a hard error
            pass
        else:
            print(f"\nFFmpeg error for {os.path.basename(video_path)}: {res.stderr.strip()}")
        if os.path.exists(temp_wav):
            try: os.remove(temp_wav)
            except OSError: pass
        return torch.zeros(128)
        
    if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 1000:
        if os.path.exists(temp_wav):
            try: os.remove(temp_wav)
            except OSError: pass
        return torch.zeros(128)
        
    try:
        with torch.no_grad():
            feat = vggish_model.forward(temp_wav)
            if feat.ndim > 1:
                feat = feat.mean(dim=0) # Aggregate across frames
        try:
            os.remove(temp_wav)
        except OSError:
            pass
        return feat.cpu()
    except Exception as e:
        print(f"\nVGGish model extraction failed for {os.path.basename(video_path)}: {e}")
        if os.path.exists(temp_wav):
            try: os.remove(temp_wav)
            except OSError: pass
        return torch.zeros(128)

def patch_file(features_path, json_path, video_dir, vggish_model, device):
    """
    Loads features file, extracts audio embeddings, patches the z_aud entry, and saves it.
    """
    print(f"\nLoading features from: {features_path}")
    if not os.path.exists(features_path):
        print(f"Skipping: {features_path} does not exist.")
        return False
        
    data = torch.load(features_path, map_location="cpu")
    print(f"Successfully loaded. Contains keys: {list(data.keys())}")
    
    # Check if z_img and video_ids exist
    if 'z_img' not in data or 'video_ids' not in data:
        print("Error: Missing z_img or video_ids in features dictionary.")
        return False
        
    # Get shape and sample size
    num_samples = len(data['video_ids'])
    print(f"Found {num_samples} samples. Current z_aud shape: {data.get('z_aud', torch.zeros(1)).shape}")
    
    # Load video_id to filename mappings
    print(f"Loading split mapping from: {json_path}")
    if not os.path.exists(json_path):
        print(f"Error: Mapping JSON not found at {json_path}")
        return False
        
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    # JSON structure is a list of dicts with 'video_id' and 'video' (filename)
    video_map = {}
    for item in json_data:
        video_map[item['video_id']] = item['video']
        
    # Check if video directory exists
    if not os.path.exists(video_dir):
        print(f"Error: Video directory '{video_dir}' does not exist.")
        return False
        
    # Begin audio extraction
    repaired_z_aud = []
    print("Extracting audio embeddings...")
    
    for vid in tqdm(data['video_ids']):
        filename = video_map.get(vid)
        if not filename:
            print(f"\nWarning: Video ID {vid} not found in JSON mapping! Using zero vector.")
            repaired_z_aud.append(torch.zeros(128))
            continue
            
        video_path = os.path.join(video_dir, filename)
        if not os.path.exists(video_path):
            print(f"\nWarning: Video file {video_path} not found! Using zero vector.")
            repaired_z_aud.append(torch.zeros(128))
            continue
            
        # Extract audio embedding
        aud_emb = extract_vggish_audio(vggish_model, video_path, device)
        repaired_z_aud.append(aud_emb)
        
    # Collate and verify shapes
    new_z_aud = torch.stack(repaired_z_aud)
    
    # Check if they are all zeros
    norms = torch.norm(new_z_aud, p=2, dim=1)
    zeros_count = (norms < 1e-5).sum().item()
    print(f"\nExtraction completed: {zeros_count}/{num_samples} ({zeros_count/num_samples*100:.1f}%) were silent or failed.")
    
    # Update dict
    data['z_aud'] = new_z_aud
    
    # Save file back
    print(f"Saving patched features to: {features_path} ...")
    torch.save(data, features_path)
    print("Patched file saved successfully!")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate and patch VGGish audio embeddings in features files.")
    parser.add_argument("--features_dir", default=".", help="Directory where test_features.pt and train_features.pt are stored")
    parser.add_argument("--video_dir", default="msrvtt/video", help="Directory where msrvtt videos are stored")
    parser.add_argument("--json_dir", default="msrvtt", help="Directory where msrvtt_test_1k.json and msrvtt_train_7k.json are stored")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run VGGish on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Install dependencies
    install_system_dependencies()
    
    # Load VGGish
    print(f"Loading VGGish encoder onto device: {args.device}...")
    try:
        vggish = torch.hub.load('harritaylor/torchvggish', 'vggish', trust_repo=True)
        vggish.eval()
        vggish.to(args.device)
        print("VGGish loaded successfully!")
    except Exception as e:
        print(f"Critical Error: Failed to load VGGish encoder: {e}")
        sys.exit(1)
        
    # Check test_features.pt
    test_feat_path = os.path.join(args.features_dir, "test_features.pt")
    test_json_path = os.path.join(args.json_dir, "msrvtt_test_1k.json")
    
    if os.path.exists(test_feat_path):
        patch_file(
            features_path=test_feat_path,
            json_path=test_json_path,
            video_dir=args.video_dir,
            vggish_model=vggish,
            device=args.device
        )
    else:
        print(f"test_features.pt not found at '{test_feat_path}', skipping.")
        
    # Check train_features.pt
    train_feat_path = os.path.join(args.features_dir, "train_features.pt")
    train_json_path = os.path.join(args.json_dir, "msrvtt_train_7k.json")
    
    if os.path.exists(train_feat_path):
        patch_file(
            features_path=train_feat_path,
            json_path=train_json_path,
            video_dir=args.video_dir,
            vggish_model=vggish,
            device=args.device
        )
    else:
        print(f"train_features.pt not found at '{train_feat_path}', skipping.")
        
    print("\n" + "=" * 60)
    print("AUDIO PATCHING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
