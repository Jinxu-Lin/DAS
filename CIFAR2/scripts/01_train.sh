echo "gpu_ids: $1"
echo "main_process_port: $2"
echo "setting: $3"

export HF_HOME="~/codes/.cache/huggingface"

accelerate launch --num_processes=$1 --gpu_ids=$2 --main_process_port=$3 01_train.py \
    --seed=42 \
    --logger="wandb" \
    --wandb_name="balckboarderCIFAR10" \
    --model_config_name_or_path="config.json" \
    --dataset_name_or_path="../../../Resources/Datasets/maskCIFAR10" \
    --dataloader_num_workers=8 \
    --resolution=32 --center_crop --random_flip \
    --train_batch_size=128 \
    --num_epochs=200 \
    --checkpointing_steps=-1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --adam_weight_decay=1e-6 \
    --save_images_epochs=100000 \
    --save_dir=./saved/ddpm \
