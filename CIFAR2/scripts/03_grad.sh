echo "gpu_ids: $1"
echo "e_seed: $2"

echo "dataset_type: $3"
echo "dataset_index: $4"
echo "split: $5"

echo "loss_function_type: $6"
echo "timestep_strategy: $7"
echo "num_timesteps_avg: $8"
echo "projection_dim: $9"

export HF_HOME="~/codes/.cache/huggingface"

CUDA_VISIBLE_DEVICES=$1 python 03_grad.py \
    --seed=42 \
    --e_seed=$2 \
    --model_config_name_or_path="config.json" \
    --model_path="./saved/ddpm" \
    --dataset_type=$3 \
    --index_path=./data/indices/$4 \
    --dataset_name_or_path="../../../Resources/Datasets/CIFAR10" \
    --gen_path="./saved/gen" \
    --dataloader_num_workers=8 \
    --resolution=32 --center_crop \
    --train_batch_size=16 \
    --split=$5 \
    --loss_function_type=$6 \
    --timestep_strategy=$7 \
    --num_timesteps_avg=$8 \
    --projection_dim=$9 \
    --save_path="./saved/grad_$2" \