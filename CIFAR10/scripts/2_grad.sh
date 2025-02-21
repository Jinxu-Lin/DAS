gpu_ids=$1
seed=$2
e_seed=$3
index_path=$4
selected_timesteps=$5
selected_timesteps_strategy=$6
output_type=$7
proj_dim=$8

echo "gpu_ids: $gpu_ids"
echo "seed: $seed"
echo "e_seed: $e_seed"
echo "index_path: $index_path"
echo "selected_timesteps: $selected_timesteps"
echo "selected_timesteps_strategy: $selected_timesteps_strategy"
echo "output_type: $output_type"
echo "proj_dim: $proj_dim"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HOME="~/codes/.cache/huggingface"

CUDA_VISIBLE_DEVICES=$gpu_ids python 2_grad.py \
    --seed=42 \
    --e_seed=$e_seed \
    --load-dataset \
    --dataset-dir "../Dataset/CIFAR10" \
    --index-path $index_path \
    --resolution 32 \
    --batch-size 32 \
    --dataloader-num-workers 0 \
    --model-dir "./saved/models/model-$seed" \
    --selected-timesteps $selected_timesteps \
    --selected-timesteps-strategy $selected_timesteps_strategy \
    --output-type $output_type \
    --proj-dim $proj_dim \
    --save-dir "./saved/grads/model-$seed" \
