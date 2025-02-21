gpu_ids=$1
seed=$2
e_seed=$3
index_path=$4
start=$5
end=$6

echo "gpu_ids: $gpu_ids"
echo "seed: $seed"
echo "e_seed: $e_seed"
echo "index_path: $index_path"
echo "start: $start"
echo "end: $end"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HOME="~/codes/.cache/huggingface"


CUDA_VISIBLE_DEVICES=$gpu_ids python 3_error.py \
    --seed=$seed \
    --e_seed=$e_seed \
    --load-dataset \
    --dataset-dir "../Dataset/CIFAR10" \
    --index-path $index_path \
    --resolution 32 \
    --center-crop \
    --batch-size 32 \
    --dataloader-num-workers 0 \
    --model-dir "./saved/models/model-$seed" \
    --save-dir "./saved/errors/model-$seed" \
