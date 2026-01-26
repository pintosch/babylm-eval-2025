from datasets import load_dataset
import os
import shutil
import aiohttp

# Download VQAv2 dataset from Hugging Face Hub
print("Downloading VQAv2 dataset from Hugging Face Hub...")

# Specify custom cache directory on another disk
custom_cache_dir = (
    "/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge78jel2/tmp/HuggingfaceM4VQAv2"
)
os.makedirs(custom_cache_dir, exist_ok=True)

# Use cache_dir parameter to specify the storage location for this dataset only
# Increase timeout to handle large dataset downloads
print("Loading dataset...", flush=True)
dataset = load_dataset(
    "HuggingFaceM4/VQAv2",
    cache_dir=custom_cache_dir,
    # download_mode="force_redownload",
    storage_options={
        "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=7200)}
    },  # 2 hours timeout
)

print("Dataset downloaded successfully!", flush=True)
print(f"Stored in: {custom_cache_dir}", flush=True)
