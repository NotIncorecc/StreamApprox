import os
import zipfile
import urllib.request
import subprocess
import sys

def main():
    print("=" * 60)
    print("REMOTE UNZIP AND FEATURE PATCHING (MSR-VTT WITH AUDIO)")
    print("=" * 60)

    # 1. Install dependencies
    print("\n--- Step 1: Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "resampy", "soundfile"], check=True)
    if os.path.exists("/usr/bin/apt-get"):
        subprocess.run("apt-get update -y && apt-get install -y libsndfile1", shell=True, check=True)
    print("Dependencies installed successfully.")

    # 2. Download JSON mapping files
    print("\n--- Step 2: Downloading JSON mapping files...")
    os.makedirs("msrvtt", exist_ok=True)
    urls = {
        "msrvtt/msrvtt_train_7k.json": "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/msrvtt_train_7k.json",
        "msrvtt/msrvtt_test_1k.json": "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/msrvtt_test_1k.json"
    }
    for path, url in urls.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            urllib.request.urlretrieve(url, path)
            print(f"Finished downloading {path}")
        else:
            print(f"{path} already exists.")

    # 3. Unzip MSR-VTT videos with audio
    zip_path = "msrvtt.zip"
    video_dir = "MSRVTT/videos/all"
    if not os.path.exists(zip_path):
        print(f"\nError: {zip_path} not found. Please ensure nyhuka/msrvtt is downloaded on Kaggle.")
        sys.exit(1)
        
    if not os.path.exists(video_dir):
        print(f"\n--- Step 3: Extracting nyhuka/msrvtt dataset zip (approx. 6.55 GB)...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all
                zip_ref.extractall(".")
            print("Extraction completed successfully.")
        except Exception as e:
            print(f"Error extracting videos: {e}")
            sys.exit(1)
    else:
        print(f"\n--- Step 3: Video directory {video_dir} already exists, skipping extraction.")

    # 4. Invoke patch script
    print("\n--- Step 4: Invoking patch_audio_features.py on Kaggle GPU...")
    patch_script = "scripts/patch_audio_features.py"
    if not os.path.exists(patch_script):
        patch_script = "patch_audio_features.py"
        
    cmd = f"{sys.executable} {patch_script} --features_dir features --video_dir {video_dir} --json_dir msrvtt"
    print(f"Running command: {cmd}")
    
    # Run synchronously and stream stdout/stderr
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip(), flush=True)
            
    rc = process.poll()
    if rc == 0:
        print("\n" + "=" * 60)
        print("AUDIO EMBEDDINGS PATCHED SUCCESSFULLY!")
        print("=" * 60)
    else:
        print(f"\nCritical Error: Patch script failed with exit code: {rc}")
        sys.exit(rc)

if __name__ == "__main__":
    main()
