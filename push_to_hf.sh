#!/bin/bash

# push_to_hf.sh: Uploads a saved model checkpoint to a Hugging Face repository, creating it if needed

# Usage: HF_TOKEN=hf_xxx ./push_to_hf.sh <local_checkpoint_dir> <hf_repo_id>

# Check if required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: HF_TOKEN=hf_xxx $0 <local_checkpoint_dir> <hf_repo_id>"
    echo "Example: HF_TOKEN=hf_xxx $0 /home/ubuntu/sft-lab/train_output/global_step_6 semantichealth/llama-3.1-sft-glaive"
    exit 1
fi

# Assign arguments
LOCAL_DIR="$1"
HF_REPO_ID="$2"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please set it like: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    exit 1
fi

# Check if the local directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo "Error: Directory $LOCAL_DIR does not exist."
    exit 1
fi

# Log in to Hugging Face and check/create the repository
echo "Checking if $HF_REPO_ID exists..."
python3 -c "
from huggingface_hub import login, HfApi
login(token='$HF_TOKEN')
api = HfApi()
try:
    api.repo_info(repo_id='$HF_REPO_ID', repo_type='model')
    print('Repository $HF_REPO_ID already exists.')
except:
    print('Repository $HF_REPO_ID not found. Creating it...')
    api.create_repo(repo_id='$HF_REPO_ID', repo_type='model', private=False)
    print('Repository $HF_REPO_ID created.')
"

# Check if repo check/creation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to verify or create repository $HF_REPO_ID"
    exit 1
fi

# Upload the folder
echo "Uploading $LOCAL_DIR to $HF_REPO_ID..."
python3 -c "from huggingface_hub import login, HfApi; login(token='$HF_TOKEN'); HfApi().upload_folder(folder_path='$LOCAL_DIR', repo_id='$HF_REPO_ID', repo_type='model', commit_message='Upload checkpoint from $LOCAL_DIR')"

# Check if the upload was successful
if [ $? -eq 0 ]; then
    echo "Successfully uploaded $LOCAL_DIR to $HF_REPO_ID"
else
    echo "Error: Failed to upload $LOCAL_DIR to $HF_REPO_ID"
    exit 1
fi