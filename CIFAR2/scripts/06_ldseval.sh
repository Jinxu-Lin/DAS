echo "gpu_ids: $1"
echo "e_seed: $2"

echo "dataset_type: $3"
echo "dataset_index: $4"

echo "start: $5"
echo "end: $6"

export HF_HOME="~/codes/.cache/huggingface"

for seed in `seq 0 2`
do
    echo "Processing seed: ${seed}"
    
    # Loop through sub-model indices from start to end
    for model_index in `seq $5 $6`
    do
        echo "Evaluating sub-model ${model_index} with seed ${seed}"
        
        # Sub-model path
        
        CUDA_VISIBLE_DEVICES=$1 python 04_eval.py \
            --seed=$seed \
            --e_seed=$2 \
            --model_config_name_or_path="config.json" \
            --model_path="./saved/lds-val/ddpm-sub-${model_index}-${seed}" \
            --dataset_type=$3 \
            --dataset_name_or_path="../../../Resources/Datasets/CIFAR10" \
            --index_path="./data/indices/$4" \
            --gen_path="./saved/$3/gen" \
            --dataloader_num_workers=8 \
            --resolution=32 --center_crop \
            --train_batch_size=256 \
            --save_path="./saved/lds-val/ddpm-sub-${index}-${seed}" \
            
    done
done




