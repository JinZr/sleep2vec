cd /home/notebook/code/personal/S9063410/sleep2vec
source /opt/conda/bin/activate sleep2vec

python wrist2vec/preprocess/save_dataset_presets.py \
    --index /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pwv+bp_index_mask_split.csv \
    --config configs/write2vec/wrist2vec_pwv+bp_pretrain_resnet1d.yaml \
    --output-template /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pretrain_data/{dataset}_{split}_preset_{tokens}{meta_suffix}.pickle \
    --overwrite \
    --num-workers 32 \
    --n-tokens 30

python -m wrist2vec.preprocess.merge_dataset_presets \
  --inputs /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pretrain_data/pwv+bp_index_mask_split_train_preset_30.pickle /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pretrain_data/pwv+bp_index_mask_split_val_preset_30.pickle /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pretrain_data/pwv+bp_index_mask_split_test_preset_30.pickle \
  --output /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pretrain_data/pwv+bp_index_mask_split_merge_preset_30.pickle