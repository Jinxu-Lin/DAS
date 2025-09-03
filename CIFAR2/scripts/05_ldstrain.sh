echo "num_processes: $1"
echo "gpu_ids: $2"
echo "main_process_port: $3"

echo "start: $4"
echo "end: $5"

export HF_HOME="~/codes/.cache/huggingface"

for seed in `seq 0 2`
do
echo ${seed}
    for index in `seq $4 $5`
    do
    echo ${index}
    accelerate launch --num_processes=$1 --gpu_ids=$2 --main_process_port=$3 01_train.py \
    --seed=${seed} \
    --logger="wandb" \
    --wandb_name="CIFAR2-automobile-horse-lds-val" \
    --model_config_name_or_path="config.json" \
    --dataset_name_or_path="../../../Resources/Datasets/CIFAR10" \
    --index_path="./data/indices/lds-val/sub-idx-${index}.pkl" \
    --dataloader_num_workers=8 \
    --resolution=32 --center_crop --random_flip \
    --train_batch_size=128 \
    --num_epochs=200 \
    --checkpointing_steps=100000 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --adam_weight_decay=1e-6 \
    --save_images_epochs=100000 \
    --save_path="./saved/lds-val/ddpm-sub-${index}-${seed}" \
    done
done
