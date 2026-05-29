import os
import requests
import json

def upload_file_to_jupyter(local_path, remote_path, server_url):
    """
    Uploads a local text file to the remote Jupyter server using the Contents API.
    """
    base_url = server_url.rstrip('/')
    api_url = f"{base_url}/api/contents/{remote_path}"
    
    with open(local_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    payload = {
        "name": os.path.basename(remote_path),
        "path": remote_path,
        "type": "file",
        "format": "text",
        "content": content
    }
    
    # Ensure parent directories exist
    parent_dir = os.path.dirname(remote_path)
    if parent_dir:
        dir_payload = {
            "name": os.path.basename(parent_dir),
            "path": parent_dir,
            "type": "directory"
        }
        create_dir_url = f"{base_url}/api/contents/{parent_dir}"
        requests.put(create_dir_url, json=dir_payload)
        
    r = requests.put(api_url, json=payload)
    r.raise_for_status()
    print(f"Successfully uploaded {local_path} to Kaggle: {remote_path}")

def download_file_from_jupyter(remote_path, local_path, server_url):
    """
    Downloads a binary/text file from the remote Jupyter server.
    """
    base_url = server_url.rstrip('/')
    file_url = f"{base_url}/files/{remote_path}"
    
    print(f"Downloading {remote_path} from Kaggle to {local_path}...")
    
    # Ensure local directory exists
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    
    r = requests.get(file_url, stream=True)
    r.raise_for_status()
    
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Download complete: {local_path}")

def sync_src_to_kaggle(server_url):
    """
    Uploads all Python files in the local src/ directory to Kaggle.
    """
    src_files = [f for f in os.listdir("src") if f.endswith(".py")]
    for filename in src_files:
        local_p = os.path.join("src", filename)
        remote_p = f"src/{filename}"
        upload_file_to_jupyter(local_p, remote_p, server_url)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync files with remote Kaggle Jupyter Server")
    parser.add_argument("--url", required=True, help="VS Code compatible connection URL from Kaggle")
    parser.add_argument("--action", choices=["upload_src", "download_features", "download_models"], required=True,
                        help="Action to perform: upload_src (local src/ -> Kaggle), download_features (Kaggle -> local data/), download_models (Kaggle -> local models/)")
    
    args = parser.parse_args()
    
    if args.action == "upload_src":
        sync_src_to_kaggle(args.url)
    elif args.action == "download_features":
        download_file_from_jupyter("train_features.pt", "data/train_features.pt", args.url)
        download_file_from_jupyter("test_features.pt", "data/test_features.pt", args.url)
    elif args.action == "download_models":
        download_file_from_jupyter("mlp_cosine.pt", "models/mlp_cosine.pt", args.url)
        download_file_from_jupyter("mlp_infonce.pt", "models/mlp_infonce.pt", args.url)
        download_file_from_jupyter("transformer_fusion.pt", "models/transformer_fusion.pt", args.url)
