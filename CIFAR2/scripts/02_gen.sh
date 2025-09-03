echo "gpu_ids: $1"
echo "gen_seed: $2"

export HF_HOME="~/codes/.cache/huggingface"

CUDA_VISIBLE_DEVICES=$1 python 02_gen.py \
    --model_config_name_or_path="config.json" \
    --train_batch_size=256 \
    --model_path="./saved/ddpm" \
    --gen_seed=$2 \
    --save_path="./saved/gen"

