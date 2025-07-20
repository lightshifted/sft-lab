#!/bin/bash

# Accept terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the conda environment
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y

# Activate the conda environment
conda activate unsloth_env

# Install dependencies
pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
pip install --no-deps peft accelerate bitsandbytes
pip install diffusers
pip install vllm
pip install --upgrade pillow
pip install flash-attn --no-build-isolation
pip install -U scikit-learn
pip install liger-kernel

# Clone verl repository
git clone https://github.com/volcengine/verl.git && cd verl && git checkout v0.3.0.post0 && pip install -e .

# Patch vERL SFT Trainer
cp fsdp_sft_trainer.py ~/sft-lab/verl/verl/trainer/fsdp_sft_trainer.py

# Patch vERL dataset handler
cp sft_dataset.py ~/sft-lab/verl/verl/utils/dataset/sft_dataset.py
