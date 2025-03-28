#!/bin/bash

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

# Clone verl repository
git clone https://github.com/volcengine/verl.git && cd verl && pip install -e .