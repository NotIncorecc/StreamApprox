import os
import zipfile
import urllib.request
import subprocess
import sys

def main():
    print("=" * 60)
    print("REMOTE KAGGLE SETUP AND FEATURE PATCHING")
    print("=" * 60)

    # 1. Install pip packages
    print("\n--- Step 1: Installing python packages (resampy, soundfile)...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "resampy", "soundfile"], check=True)
        print("Python packages installed successfully.")
    except Exception as e:
        print(f"Error installing python packages: {e}")
        sys.exit(1)

    # 2. Install libsndfile1
    print("\n--- Step 2: Installing system audio libraries (libsndfile1)...")
    if os.path.exists("/usr/bin/apt-get"):
        try:
            subprocess.run("apt-get update -y && apt-get install -y libsndfile1", shell=True, check=True)
            print("System libraries installed successfully.")
        except Exception as e:
            print(f"Warning: Failed to install libsndfile1: {e}")
    else:
        print("Not running on Debian/Ubuntu system, skipping apt-get.")

    # 3. Create msrvtt directories
    print("\n--- Step 3: Downloading MSR-VTT dataset files...")
    os.makedirs("msrvtt", exist_ok=True)
    
    urls = {
        "msrvtt/msrvtt_train_7k.json": "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/msrvtt_train_7k.json",
        "msrvtt/msrvtt_test_1k.json": "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/msrvtt_test_1k.json",
        "msrvtt/MSRVTT_Videos.zip": "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/MSRVTT_Videos.zip"
    }

    for path, url in urls.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            urllib.request.urlretrieve(url, path)
            print(f"Finished downloading {path}")
        else:
            print(f"{path} already exists.")

    # Extract videos
    video_dir = "msrvtt/video"
    if not os.path.exists(video_dir):
        print("\n--- Step 4: Extracting videos zip (approx. 2.6 GB)...")
        try:
            with zipfile.ZipFile("msrvtt/MSRVTT_Videos.zip", 'r') as zip_ref:
                zip_ref.extractall("msrvtt")
            print("Videos extracted successfully.")
        except Exception as e:
            print(f"Error extracting videos: {e}")
            sys.exit(1)
    else:
        print("\n--- Step 4: Video directory already exists, skipping extraction.")

    # 4. Execute patching script
    print("\n--- Step 5: Invoking patch_audio_features.py on Kaggle GPU...")
    patch_script = "scripts/patch_audio_features.py"
    if not os.path.exists(patch_script):
        # Fallback if scripts folder structure is different
        patch_script = "patch_audio_features.py"
        
    cmd = f"{sys.executable} {patch_script} --features_dir features --video_dir msrvtt/video --json_dir msrvtt"
    print(f"Running command: {cmd}")
    
    # Run synchronously and output stdout/stderr
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Stream the output line-by-line
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip(), flush=True)
            
    rc = process.poll()
    if rc == 0:
        print("\n" + "=" * 60)
        print("REMOTE SETUP & PATCH RUN COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    else:
        print(f"\nCritical Error: Patch script failed with exit code: {rc}")
        sys.exit(rc)

if __name__ == "__main__":
    main()
