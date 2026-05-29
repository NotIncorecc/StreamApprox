import json
import uuid
import requests
import websocket
import urllib.parse

def execute_on_kaggle(code, server_url, kernel_id=None):
    """
    Executes a string of python code on the remote Kaggle Jupyter server
    and streams back the stdout, stderr, and execution errors.
    """
    base_url = server_url.rstrip('/')
    
    # 1. Get list of kernels if kernel_id is not specified
    if not kernel_id:
        r = requests.get(f"{base_url}/api/kernels")
        r.raise_for_status()
        kernels = r.json()
        if not kernels:
            # Create a new kernel if none exist
            r = requests.post(f"{base_url}/api/kernels")
            r.raise_for_status()
            kernel_id = r.json()['id']
        else:
            kernel_id = kernels[0]['id']
            
    print(f"Connecting to Kaggle Kernel: {kernel_id}")
    
    # 2. Parse websocket URL from the base URL
    parsed_url = urllib.parse.urlparse(base_url)
    ws_scheme = "wss" if parsed_url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed_url.netloc}{parsed_url.path}/api/kernels/{kernel_id}/channels"
    
    # 3. Establish websocket connection
    ws = websocket.create_connection(ws_url, timeout=30)
    
    # 4. Prepare execute_request message
    session_id = uuid.uuid4().hex
    msg_id = uuid.uuid4().hex
    execute_msg = {
        "header": {
            "msg_id": msg_id,
            "username": "antigravity",
            "session": session_id,
            "msg_type": "execute_request",
            "version": "5.3"
        },
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True
        },
        "buffers": [],
        "parent_header": {}
    }
    
    # 5. Send message
    ws.send(json.dumps(execute_msg))
    
    # 6. Listen for outputs until execution finishes
    has_error = False
    while True:
        try:
            resp = json.loads(ws.recv())
            # Check if this message is in response to our request
            if resp.get("parent_header", {}).get("msg_id") != msg_id:
                continue
                
            msg_type = resp.get("msg_type")
            content = resp.get("content", {})
            
            if msg_type == "stream":
                text = content.get("text", "")
                try:
                    print(text, end="", flush=True)
                except UnicodeEncodeError:
                    import sys
                    enc = sys.stdout.encoding or "utf-8"
                    print(text.encode(enc, errors="replace").decode(enc), end="", flush=True)
            elif msg_type == "execute_result" or msg_type == "display_data":
                data = content.get("data", {})
                if "text/plain" in data:
                    text = data["text/plain"]
                    try:
                        print(text, flush=True)
                    except UnicodeEncodeError:
                        import sys
                        enc = sys.stdout.encoding or "utf-8"
                        print(text.encode(enc, errors="replace").decode(enc), flush=True)
            elif msg_type == "error":
                print(f"\nExecution Error: {content.get('ename')}: {content.get('evalue')}", flush=True)
                for line in content.get("traceback", []):
                    print(line, flush=True)
                has_error = True
            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break
        except Exception as e:
            print(f"\nWebsocket read error: {e}")
            break
            
    ws.close()
    if has_error:
        raise RuntimeError("Code execution failed on remote Kaggle kernel.")
    print("\nExecution completed successfully.")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Python code on remote Kaggle Jupyter Server")
    parser.add_argument("--url", required=True, help="VS Code compatible connection URL from Kaggle")
    parser.add_argument("--code", help="Inline Python code to execute")
    parser.add_argument("--file", help="Python file path to read code from")
    
    args = parser.parse_args()
    
    code_to_run = ""
    if args.code:
        code_to_run = args.code
    elif args.file:
        with open(args.file, "r") as f:
            code_to_run = f.read()
    else:
        print("Please provide --code or --file to run.")
        sys.exit(1)
        
    execute_on_kaggle(code_to_run, args.url)
