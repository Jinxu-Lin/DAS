echo "gpu_ids: $1"

echo "dataset_type: $2"
echo "method: $3"

echo "loss_function_type: $4"
echo "grad_num_timesteps: $5"
echo "projection_dim: $6"
echo "timestep_strategy: $7"

CUDA_VISIBLE_DEVICES=$1 python 07_score.py \
    --dataset_type=$2 \
    --method=$3 \
    --loss_function_type=$4 \
    --grad_num_timesteps=$5 \
    --projection_dim=$6 \
    --timestep_strategy=$7 \