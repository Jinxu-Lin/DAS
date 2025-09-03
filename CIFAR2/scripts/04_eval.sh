echo "gpu_ids: $1"
echo "e_seed: $2"

echo "dataset_type: $3"
echo "dataset_index: $4"

export HF_HOME="~/codes/.cache/huggingface"
    
CUDA_VISIBLE_DEVICES=$1 python 04_eval.py \
    --seed=42 \
    --e_seed=$2 \
    --model_config_name_or_path="config.json" \
    --model_path="./saved/ddpm" \
    --dataset_type=$3 \
    --dataset_name_or_path="../../../Resources/Datasets/CIFAR10" \
    --index_path="./data/indices/$4" \
    --gen_path="./saved/gen" \
    --dataloader_num_workers=8 \
    --resolution=32 --center_crop \
    --train_batch_size=256 \
    --save_path=./saved/error_$2 \