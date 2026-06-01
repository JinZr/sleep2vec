cd /home/notebook/code/personal/S9063410/sleep2vec
source /root/miniconda3/bin/activate wrist2vec

GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
DEVICES=$(seq 0 $((GPUS_PER_NODE-1)) | tr '\n' ' ')

echo "Devices: $DEVICES"

WANDB_MODE=offline python -m wrist2vec_flex.pretrain \
    --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_250hz_pretrain_resnet1d.yaml \
    --pretrain-data-index /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask_split.csv \
    --pretrain-preset-path /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pretrain_data/bp_index_mask_split_merge_preset_30.pickle \
    --version-name multilight_250Hz \
    --epochs 30 \
    --lr 5e-5 \
    --batch-size 1280 \
    --devices $DEVICES \
    --num-workers 64 \
    --val-num-workers 16 \
    --allow-missing-channels \
    --min-channels 2 \
    2>&1 | tee -a "training_terminal_out.txt"