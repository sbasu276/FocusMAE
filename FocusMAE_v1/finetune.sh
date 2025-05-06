#!/usr/bin/env bash
set -x

# source activate focusmae

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MASTER_PORT=${MASTER_PORT:-12320}
export OMP_NUM_THREADS=1

# Command-line arguments
MODEL_PATH=$1
OUTPUT_DIR=$2
DATA_PATH=$3

COLUMN_NAME='pred_column'
N_NODES=${N_NODES:-1}
GPUS_PER_NODE=1
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}  # Additional arguments
NODE_RANK=0
INPUT_SIZE=224

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=${GPUS_PER_NODE} \
       run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set GBC_Net \
        --auto_resume \
        --nb_classes 2 \
        --save_ckpt \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --input_size ${INPUT_SIZE} \
        --sampling_rate 3 \
        --num_sample 2 \
        --num_workers 4 \
        --opt adamw \
        --num_segment 1 \
        --lr 1e-3 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --epochs 59 \
        --test_randomization \
        --dist_eval --enable_deepspeed \
        --pred_column ${COLUMN_NAME} 
