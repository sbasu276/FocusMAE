#!/usr/bin/env bash
set -x

source activate videomae
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda-9.2
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

export MASTER_PORT=${MASTER_PORT:-12320}
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=7

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1


OUTPUT_DIR='pretrained_model_working_directory'
DATA_PATH='pretrained_data_train.txt'
MODEL_PATH='Pretrained_model_path'

# JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
N_NODES=${N_NODES:-1}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-1}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3} 
NODE_RANK=0

CPUS_PER_TASK=${CPUS_PER_TASK:-8}

PY_ARGS=${@:3}



torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node=${GPUS_PER_NODE} --nnodes=${N_NODES} --node_rank=0 run_mae_pretraining.py \
    --data_path ${DATA_PATH} \
    --mask_type ours \
    --device cuda \
    --mask_ratio 0.8 \
    --inflateroi 1.2 \
    --decoder_mask_type run_cell \
    --decoder_mask_ratio 0.5 \
    --delta 2 \
    --model pretrain_adamae_base_patch16_224 \
    --finetune ${MODEL_PATH} \
    --decoder_depth 10 \
    --no_save_ckpt \
    --batch_size 4\
    --input_size 224 \
    --save_ckpt_freq 10 \
    --num_sample 2 \
    --num_workers 8 \
    --tubelet_size 2 \
    --num_frames 16 \
    --sampling_rate 1 \
    --num_workers 10 \
    --warmup_lr 1e-3 \
    --lr 1e-3 \
    --min_lr 1e-12 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 1 \
    --save_ckpt_freq 20 \
    --epochs 500 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --enable_deepspeed \
    ${PY_ARGS}

