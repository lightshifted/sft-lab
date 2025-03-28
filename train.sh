#!/bin/bash
set -x

export N_NODES=1
export N_GPUS_PER_NODE=2
export MASTER_PORT=29500

torchrun \
    --nnodes=$N_NODES \
    --nproc_per_node=$N_GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/ubuntu/data/reasoning/train.parquet \
    data.val_files=/home/ubuntu/data/reasoning/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=/home/ubuntu/models/meta-llama3.1-8b-instruct \
    trainer.default_hdfs_dir=hdfs://user/verl/experiments/meta/llama-3.1-8B-Instruct/ \
    trainer.project_name=llama3.1-sft \
    trainer.experiment_name=llama3.1-instruct-sft \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb']