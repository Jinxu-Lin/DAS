echo "gpu_ids: $1"
echo "e_seed: $2"
echo "setting: $3"

echo "index: $4"

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
        model_path="./saved/$3/lds-val/ddpm-sub-${model_index}-${seed}"
        
        CUDA_VISIBLE_DEVICES=$1 python 04_eval.py \
            --dataset_name="cifar10" \
            --dataloader_num_workers=8 \
            --model_config_name_or_path="config.json" \
            --resolution=32 --center_crop \
            --train_batch_size=256 \
            --index_path=./data/indices/$3/$4 \
            --gen_path=./saved/$3/gen \
            --model_path=${model_path} \
            --output_dir=./saved/$3/lds-val \
            --model_index=${model_index} \
            --e_seed=$2 \
            --seed=${seed}
    done
done




