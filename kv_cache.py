# kv_cache.py
import pickle
import logging
import hashlib
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cache_dir = "/home/amar/llama3_assistant/data/kv_cache"
target_file = "Top Attractions in Bengaluru .pdf"

def list_cache_files():
    """List all KV cache files in the directory"""
    try:
        files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
        logger.info(f"Found {len(files)} cache files in {cache_dir}:")
        for f in files:
            logger.info(f"  - {f}")
        return files
    except Exception as e:
        logger.error(f"Error listing cache files: {e}")
        return []

def find_cache_file(target_file_name):
    """Find the cache file for the target document"""
    cache_key = hashlib.md5(target_file_name.encode()).hexdigest()
    for user_id in ['default', 'None']:  # Check both possible user_ids
        cache_path = os.path.join(cache_dir, f"{user_id}_{cache_key}.pkl")
        if os.path.exists(cache_path):
            logger.info(f"Found cache file: {cache_path}")
            return cache_path
    logger.warning(f"No cache file found for {target_file_name}")
    return None

def read_kv_cache(cache_path):
    """Read and print KV cache contents"""
    try:
        with open(cache_path, 'rb') as f:
            kv_cache = pickle.load(f)
        logger.info(f"Loaded KV cache from {cache_path}")
        for entry in kv_cache:
            logger.info(f"Chunk {entry.get('chunk_id', 'Unknown')}:")
            logger.info(f"  File: {entry.get('file_name', 'Unknown')}")
            logger.info(f"  Text: {entry.get('text', 'No text')[:200]}...")
            logger.info(f"  KV State: {entry.get('kv_state', 'No KV state')}")
            logger.info(f"  Processed: {datetime.fromtimestamp(entry.get('processed_time', 0)) if entry.get('processed_time') else 'Unknown'}")
    except Exception as e:
        logger.error(f"Error reading KV cache: {e}")

if __name__ == "__main__":
    cache_files = list_cache_files()
    cache_path = find_cache_file(target_file)
    if cache_path:
        read_kv_cache(cache_path)
    else:
        logger.error(f"Cannot read KV cache. Please re-upload {target_file} in the app.")