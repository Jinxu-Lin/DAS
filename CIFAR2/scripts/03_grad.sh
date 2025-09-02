echo "gpu_ids: $1"
echo "e_seed: $2"
echo "setting: $3"

echo "index: $4"
echo "split: $5"

echo "ckpt: $6"

echo "loss_function_type: $7"
echo "timestep_strategy: $8"
echo "num_timesteps_avg: $9"
echo "projection_dim: ${10}"

export HF_HOME="~/codes/.cache/huggingface"

CUDA_VISIBLE_DEVICES=$1 python 03_grad.py \
    --dataset_name="cifar10" \
    --dataloader_num_workers=8 \
    --model_config_name_or_path="config.json" \
    --resolution=32 --center_crop \
    --train_batch_size=52 \
    --e_seed=$2 \
    --index_path=./data/indices/$3/$4 \
    --split=$5 \
    --output_dir=./saved/$3/$6 \
    --gen_path=./saved/$3/gen \
    --loss_function_type=$7 \
    --timestep_strategy=$8 \
    --num_timesteps_avg=$9 \
    --projection_dim=${10} \
    --seed=42
    