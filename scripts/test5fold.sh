#!/usr/bin/env bash
set -x
source activate videomae
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda-9.2
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export MASTER_PORT=${MASTER_PORT:-12320}
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3

OUTPUT_DIR='working directory test_model_fold_3_path'
DATA_PATH='Data_path_fold_4'
MODEL_PATH='Pretrained_model_path'
COLUMN_NAME='pred_column'

N_NODES=${N_NODES:-1}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-2}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3} 
NODE_RANK=0
INPUT_SIZE=224


torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE} \
 run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set GBC_Net \
        --auto_resume \
        --nb_classes 2 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 2 \
        --input_size ${INPUT_SIZE} \
        --num_workers 10 \
        --sampling_rate 3 \
        --num_sample 4 \
        --num_workers 4 \
        --opt adamw \
        --num_segment 1 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --epochs 90 \
        --test_randomization \
        --dist_eval --enable_deepspeed \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}
        

OUTPUT_DIR='working directory test_model_fold_4_path'
DATA_PATH='Data_path_fold_4'

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE}\
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
        --test_num_segment 10 \
        --test_num_crop 3 \
        --epochs 59 \
        --test_randomization \
        --dist_eval --enable_deepspeed \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}

OUTPUT_DIR='working directory test_model_fold_2_path'
DATA_PATH='Data_path_fold_2'


torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE}\
 run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set GBC_Net \
        --auto_resume \
        --save_ckpt \
        --nb_classes 2 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --input_size ${INPUT_SIZE} \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 4 \
        --opt adamw \
        --num_segment 1 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --epochs 69 \
        --test_randomization \
        --dist_eval --enable_deepspeed \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}


OUTPUT_DIR='working directory test_model_fold_1_path'
DATA_PATH='Data_path_fold_1'
#90 epochs

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE}\
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
        --test_num_segment 5 \
        --test_num_crop 3 \
        --epochs 59 \
        --test_randomization \
        --dist_eval --enable_deepspeed \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}

OUTPUT_DIR='working directory test_model_fold_0_path'
DATA_PATH='Data_path_fold_0'

#90 epcohs

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE}\
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
        --test_num_segment 5 \
        --test_num_crop 3 \
        --epochs 59 \
        --test_randomization \
        --dist_eval --enable_deepspeed \
        --pred_column ${COLUMN_NAME} \
        ${PY_ARGS}