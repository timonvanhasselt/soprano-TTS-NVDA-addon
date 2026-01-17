import os
import requests
from logHandler import log

# The repository containing all necessary files
REPO_URL = "https://huggingface.co/KevinAHM/soprano-1.1-onnx/resolve/main"

# Files located in the 'onnx' subfolder on HuggingFace
ONNX_REMOTE_FILES = [
    "onnx/soprano_backbone_kv_int8.onnx",
    "onnx/soprano_decoder_int8.onnx"
]

# Files located in the root directory on HuggingFace
CONFIG_REMOTE_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json"
]

def download_file(url, target_path, session):
    if os.path.exists(target_path):
        # Check if the file is not empty (e.g., from a previously failed download)
        if os.path.getsize(target_path) > 0:
            log.info(f"Soprano TTS: {os.path.basename(target_path)} already exists.")
            return True
    
    log.info(f"Soprano TTS: Downloading {url}...")
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
        log.info(f"Soprano TTS: {os.path.basename(target_path)} saved successfully.")
        return True
    except Exception as e:
        log.error(f"Soprano TTS: Error downloading {url}: {e}")
        return False

def onInstall():
    base_dir = os.path.dirname(__file__)
    # We store everything flat in the 'models' folder for easy access in helper.py
    models_dir = os.path.join(base_dir, "synthDrivers", "soprano", "models")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    headers = {"User-Agent": "NVDA-Addon-Soprano-1.1"}
    
    with requests.Session() as session:
        session.headers.update(headers)
        
        # 1. Download ONNX models from the 'onnx/' subfolder
        for remote_path in ONNX_REMOTE_FILES:
            filename = os.path.basename(remote_path)
            url = f"{REPO_URL}/{remote_path}"
            download_file(url, os.path.join(models_dir, filename), session)
            
        # 2. Download Config & Tokenizer from the root
        for filename in CONFIG_REMOTE_FILES:
            url = f"{REPO_URL}/{filename}"
            download_file(url, os.path.join(models_dir, filename), session)

    log.info("Soprano TTS: Installation download task completed.")