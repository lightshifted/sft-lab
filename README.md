# SFT Lab

## Install Anaconda
1. ```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh```
2. ```chmod +x Miniconda3-latest-Linux-x86_64.sh```
3. ```./Miniconda3-latest-Linux-x86_64.sh -b```

## Activate Conda Environment
4. ```source ~/miniconda3/bin/activate```

## Clone SFT Lab
5. ```git clone https://github.com/lightshifted/sft-lab.git```
6. ```cd sft-lab```

## Setup Environment with Dependencies
7. ```chmod +x setup_env.sh```
8. ```./setup_env.sh```

## Download Llama-3.1-8B-Instruct
9.  ```HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir ~/models/meta-llama3.1-8b-instruct \
    --local-dir-use-symlinks False```

## Start Training
10. ```chmod +x train.sh```
11. ```./train.sh```

# Heads-Up
- You'll want to modify `train.sh` to the specs of your training run
- Tracking the training run with Weights & Biases is recommended