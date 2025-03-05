#!/usr/bin/env bash
set -x

# source activate focusmae

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MASTER_PORT=${MASTER_PORT:-12320}
export OMP_NUM_THREADS=1

MODEL_PATH=$1
OUTPUT_DIR=$2
DATA_PATH=$3
FRCNN_JSON_FILE=$4
DIR_PATH=$5

# JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
N_NODES=${N_NODES:-1}  # Number of nodes
# GPUS_PER_NODE=${GPUS_PER_NODE:-1}  # Number of GPUs in each node
GPUS_PER_NODE=1
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3} 
NODE_RANK=0

CPUS_PER_TASK=${CPUS_PER_TASK:-8}

PY_ARGS=${@:3}


torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node=${GPUS_PER_NODE} --nnodes=${N_NODES} --node_rank=0 \
    run_mae_pretraining.py \
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
    --json_file_path ${FRCNN_JSON_FILE} \
    --images_folder_path ${DIR_PATH} \
    --decoder_depth 10 \
    --no_save_ckpt \
    --batch_size 1 \
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
    --epochs 800 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --enable_deepspeed 

