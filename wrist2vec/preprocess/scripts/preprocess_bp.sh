cd /home/notebook/code/personal/S9063410/sleep2vec
source /opt/conda/bin/activate sleep2vec

# cat wrist2vec/preprocess/scripts/preprocess_bp.sh
# python wrist2vec/preprocess/ppg_formant_multilight.py \
#     --out-root /home/notebook/data/personal/S9063410/pwv+bp_data_multilight \
#     --metadata-csv /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index.csv \
#     --target-fs 250 \
#     --bp-low 0.5 \
#     --bp-high 40.0 \
#     --workers 32

# python wrist2vec/preprocess/ppg_metadata.py \
#     --index-csv /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index.csv \
#     --roster-csv /home/notebook/data/personal/S9063410/bp_data_one_channel/加入研究信息整合名单260422.csv \
#     --out-csv /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask.csv \
#     --workers 32

# python wrist2vec/preprocess/split_index_by_subject.py \
#     --input /home/notebook/data/personal/S9063410/bp_data_one_channel/index_mask.csv \
#     --output /home/notebook/data/personal/S9063410/bp_data_one_channel/index_mask_split.csv


python wrist2vec/preprocess/save_dataset_presets.py \
    --index /home/notebook/data/personal/S9063410/bp_data_one_channel/index_mask_split.csv \
    --config configs/write2vec/wrist2vec_onechannel_ppg_accgyro_pretrain_resnet1d.yaml \
    --output-template /home/notebook/data/personal/S9063410/bp_data_one_channel/pretrain_data/{dataset}_{split}_preset_{tokens}{meta_suffix}.pickle \
    --overwrite \
    --num-workers 32 \
    --n-tokens 30


python -m wrist2vec.preprocess.merge_dataset_presets \
  --inputs /home/notebook/data/personal/S9063410/bp_data_one_channel/pretrain_data/index_mask_split_train_preset_30.pickle /home/notebook/data/personal/S9063410/bp_data_one_channel/pretrain_data/index_mask_split_val_preset_30.pickle /home/notebook/data/personal/S9063410/bp_data_one_channel/pretrain_data/index_mask_split_test_preset_30.pickle \
  --output /home/notebook/data/personal/S9063410/bp_data_one_channel/pretrain_data/index_mask_split_merge_preset_30.pickle