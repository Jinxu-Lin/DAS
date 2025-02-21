gpu_ids=$1
model_id=$2
e_seed=$3
index_path=$4

echo "gpu_ids: $gpu_ids"
echo "model_id: $model_id"
echo "e_seed: $e_seed"
echo "index_path: $index_path"

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
    --model-dir "./saved/models/model-$model_id" \
    --save-dir "./saved/grads/model-$model_id" \
