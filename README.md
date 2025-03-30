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
    --local-dir ~/sft-lab/models/meta-llama3.1-8b-instruct \
    --local-dir-use-symlinks False```

## Prepare Training Data
10. ```python data_preprocess/reasoning_data.py --sample_size 300_000```

## Start Training
11. ```chmod +x train.sh```
12. ```./train.sh```

## Push SFT Model to Hugging Face
13. ```HF_TOKEN=<your_huggingface_token> ./push_to_hf.sh <path_to_checkpoint> <your_huggingface_repo>```

## Heads-Up

### `train.sh` configuration options:
```
data:
  train_batch_size: 64
  micro_batch_size: null 
  micro_batch_size_per_gpu: 2
  train_files: /home/ubuntu/sft-lab/data/glavieai/train.parquet
  val_files: /home/ubuntu/sft-lab/data/glavieai/test.parquet
  prompt_key: prompt
  response_key: response
  max_length: 8192
  truncation: error
  balance_dp_token: false
  chat_template: null

model:
  partial_pretrain: /home/ubuntu/sft-lab/models/meta-llama3.1-8b-instruct
  fsdp_config:
    wrap_policy:
      min_num_params: 0
    cpu_offload: false
    offload_params: false
  external_lib: null
  enable_gradient_checkpointing: true
  trust_remote_code: false
  lora_rank: 0
  lora_alpha: 16
  target_modules: all-linear
  use_liger: true

optim:
  lr: 1e-5
  betas: [0.9, 0.95]
  weight_decay: 0.01
  warmup_steps_ratio: 0.1
  clip_grad: 1.0

ulysses_sequence_parallel_size: 1
use_remove_padding: false

trainer:
  default_local_dir: /tmp/sft_model
  default_hdfs_dir: hdfs://user/verl/experiments/meta/llama-3.1-8B-Instruct/
  resume_path: null
  project_name: llama3.1-instruct-sft
  experiment_name: llama3.1-instruct-sft
  total_epochs: 1
  total_training_steps: null
  logger: [console, wandb]
  seed: 1
```
