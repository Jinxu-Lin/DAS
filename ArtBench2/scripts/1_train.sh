gpu_ids=$1
main_process_port=$2
seed=$3

echo "gpu_ids: $gpu_ids"
echo "main_process_port: $main_process_port"
echo "seed: $seed"

export MODEL_NAME="lambdalabs/miniSD-diffusers"
export DATASET_NAME="data/artbench-10-imagefolder/**"
export HF_HOME="~/codes/.cache/huggingface"

accelerate launch --gpu_ids $gpu_ids --main_process_port=$main_process_port --num_processes 1 1_train.py \
    --seed $seed \
    --logger "wandb" \
    --wandb-name "Artbench2-train" \
    --dataset-dir "../Dataset/ArtBench10" \
    --index-path "./data/idx-train.pkl" \
    --resolution 256 \
    --shuffle \
    --center-crop \
    --random-flip \
    --batch-size 128 \
    --dataloader-num-workers 8 \
    --model-path "../Models/miniSD-diffusers" \
    --learning-rate 3e-04 \
    --adam-weight-decay 1e-06 \
    --lr-scheduler "cosine" \
    --mixed-precision "fp16" \
    --num-train-epochs 100 \
    --gradient-accumulation-steps 1 \
    --save-dir "./saved/models/model-$seed" \
    --checkpointing-steps -1 \




