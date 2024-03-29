#!/usr/bin/env bash
set -x
source activate videomae
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda-9.2
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export MASTER_PORT=${MASTER_PORT:-12320}
export OMP_NUM_THREADS=1


OUTPUT_DIR='Working directory for CT Data, use format path to CT_finetuning_checkpoint and place checkpoint-29.pth in the folder'
DATA_PATH='Path to folder containing train test val csv files for CT videos'
MODEL_PATH='Pretrained Model Path'

N_NODES=${N_NODES:-1}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-2}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3} 
NODE_RANK=0
# batch_size can be adjusted according to the graphics card


torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node=${GPUS_PER_NODE} \
      --nnodes=${N_NODES} \
        run_class_finetuning.py \
        --model vit_small_patch16_224 \
        --data_set Kinetics-400 \
        --auto_resume \
        --nb_classes 2 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 5 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 4 \
        --save_ckpt \
        --opt adamw \
        --num_segment 1 --lr 7e-3 \
        --warmup_lr 1e-2 \
        --opt_betas 0.9 0.9 \
        --weight_decay 0.001 \
        --clip_grad 5.0 \
        --min_lr 1e-5 \
        --warmup_epochs 1 \
        --epochs 30 \
        --test_num_segment 10 \
        --test_randomization \
        --test_num_crop 3  \
        --dist_eval \
        --eval_ct \
        ${PY_ARGS}

