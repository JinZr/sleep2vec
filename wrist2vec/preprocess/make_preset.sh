cd /home/notebook/code/personal/S9063410/sleep2vec
source /opt/conda/bin/activate sleep2vec

python wrist2vec/preprocess/save_dataset_presets.py \
    --index /home/notebook/data/personal/S9063410/bp_data_multilight/index_mask_split.csv \
    --config configs/write2vec/wrist2vec_multichannel_ppg_accgyro_pretrain_resnet1d.yaml \
    --output-template /home/notebook/data/personal/S9063410/bp_data_multilight/pretrain_data/{dataset}_{split}_preset_{tokens}{meta_suffix}.pickle \
    --n-tokens 30