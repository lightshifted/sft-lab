#!/bin/bash
set -x

export N_NODES=1
export N_GPUS_PER_NODE=4
export MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Optimize for multi-GPU
export NCCL_P2P_LEVEL=NVL             # Force NVLink usage
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Better error handling

torchrun \
    --nnodes=$N_NODES \
    --nproc_per_node=$N_GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/ubuntu/sft-lab/data/glavieai/train.parquet \
    data.val_files=/home/ubuntu/sft-lab/data/glavieai/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=8192 \
    model.partial_pretrain=/home/ubuntu/sft-lab/models/meta-llama3.1-8b-instruct \
    model.enable_gradient_checkpointing=true \
    trainer.default_hdfs_dir=hdfs://user/verl/experiments/meta/llama-3.1-8B-Instruct/ \
    trainer.project_name=llama3.1-instruct-sft \
    trainer.experiment_name=llama3.1-instruct-sft \
    trainer.total_epochs=2 \
    trainer.save_frequency=100 \
    trainer.test_freq=100 \
    trainer.remove_previous_ckpt_in_save=true \
    trainer.logger=['console','wandb']