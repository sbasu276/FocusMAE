set -x
source activate videomae
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/usr/local/cuda-9.2
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MASTER_PORT=${MASTER_PORT:-12320}
export OMP_NUM_THREADS=1


OUTPUT_DIR='Working directory'
DATA_PATH='Path to csv file with video level annotations'
MODEL_PATH='Path to Pretrained model'

N_NODES=${N_NODES:-1}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-2}  # Number of GPUs in each node
PY_ARGS=${@:3} 
NODE_RANK=0

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1  --nproc_per_node=${GPUS_PER_NODE} \
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
        --input_size 224 \
        --short_side_size 224 \
        --num_frames 16 \
        --num_workers 10 \
        --sampling_rate 3 \
        --num_sample 2 \
        --num_workers 4 \
        --opt adamw \
        --num_segment 1 \
        --lr 7e-4 \
        --min_lr 1e-8 \
        --opt_betas 0.9 0.999 \
        --test_num_segment 10 \
        --test_num_crop 3 \
        --epochs 58 \
        --test_randomization \
        --dist_eval --enable_deepspeed \
        ${PY_ARGS}