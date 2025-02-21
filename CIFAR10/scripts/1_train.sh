gpu_ids=$1
main_process_port=$2
seed=$3

echo "gpu_ids: $gpu_ids"
echo "main_process_port: $main_process_port"
echo "seed: $seed"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HOME="~/codes/.cache/huggingface"

accelerate launch --gpu_ids $gpu_ids --main_process_port=$main_process_port --num_processes=1 1_train.py \
    --seed=$seed \
    --load-dataset \
    --dataset-dir "../Dataset/CIFAR10" \
    --resolution 32 \
    --shuffle \
    --batch-size 128 \
    --dataloader-num-workers 8 \
    --learning-rate 1e-4 \
    --adam-weight-decay 1e-6 \
    --num-epochs 200 \
    --checkpointing-steps -1 \
    --gradient-accumulation-steps 1 \
    --logger "tensorboard" \
    --index-path "./data/idx-train.pkl" \
    --save-dir "./saved/models/model-$seed"
