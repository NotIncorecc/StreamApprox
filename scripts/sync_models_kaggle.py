import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Download model weights from Kaggle using Kaggle API")
    parser.add_argument("--kernel", required=True, help="Kaggle kernel identifier (e.g. username/kernel-slug)")
    parser.add_argument("--dest", default="models", help="Local directory to save the models")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    
    print("=" * 60)
    print("SYNCING KAGGLE MODEL CHECKPOINTS LOCALLY")
    print(f"Kernel:      {args.kernel}")
    print(f"Destination: {args.dest}")
    print("=" * 60)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Error: The 'kaggle' package is not installed in this python environment.")
        print("Run 'pip install kaggle' or make sure requirements.txt is installed.")
        sys.exit(1)

    try:
        print("Authenticating with Kaggle API...")
        # Note: requires ~/.kaggle/kaggle.json or %USERPROFILE%\.kaggle\kaggle.json
        api = KaggleApi()
        api.authenticate()
        
        print(f"Fetching outputs for kernel '{args.kernel}'...")
        # Parse owner and kernel slug
        if "/" not in args.kernel:
            print("Error: Kernel identifier must be in the format 'username/kernel-slug'")
            sys.exit(1)
            
        owner, kernel_slug = args.kernel.split("/", 1)
        
        # Download output files
        api.kernels_output(owner, kernel_slug, path=args.dest)
        print("\n" + "=" * 60)
        print(f"SUCCESS: Models downloaded locally to: {os.path.abspath(args.dest)}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError downloading outputs from Kaggle: {e}")
        print("Please ensure your Kaggle credentials are configured correctly at ~/.kaggle/kaggle.json")
        sys.exit(1)

if __name__ == "__main__":
    main()
