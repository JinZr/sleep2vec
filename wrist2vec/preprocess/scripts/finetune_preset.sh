cd /home/notebook/code/personal/S9063410/sleep2vec
source /opt/conda/bin/activate sleep2vec

INDEX=/home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask_split.csv
OUT=/home/notebook/data/personal/S9063410/pwv+bp_data_multilight/finetune_presets
DATASET_NAME=bp_multilight
mkdir -p "$OUT"

python -m wrist2vec.preprocess.save_dataset_presets \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_250hz_pretrain_resnet1d.yaml \
  --index "$INDEX" \
  --dataset-name $DATASET_NAME \
  --n-tokens 30 \
  --split train val test \
  --meta-data-names sex age bmi \
  --channels ppg_green ppg_infrared gyro_vm \
  --output-template "$OUT/{dataset}_{split}_preset_{tokens}{meta_suffix}.pickle" \
  --no-allow-missing-channels \
  --overwrite \
  --num-workers 32


python -m wrist2vec.preprocess.merge_dataset_presets \
  --inputs "$OUT"/"$DATASET_NAME"_train_preset_30_sex.pickle "$OUT"/"$DATASET_NAME"_val_preset_30_sex.pickle "$OUT"/"$DATASET_NAME"_test_preset_30_sex.pickle \
  --output "$OUT"/"$DATASET_NAME"_merged_preset_30_sex.pickle

python -m wrist2vec.preprocess.merge_dataset_presets \
  --inputs "$OUT"/"$DATASET_NAME"_train_preset_30_age.pickle "$OUT"/"$DATASET_NAME"_val_preset_30_age.pickle "$OUT"/"$DATASET_NAME"_test_preset_30_age.pickle \
  --output "$OUT"/"$DATASET_NAME"_merged_preset_30_age.pickle

python -m wrist2vec.preprocess.merge_dataset_presets \
  --inputs "$OUT"/"$DATASET_NAME"_train_preset_30_bmi.pickle "$OUT"/"$DATASET_NAME"_val_preset_30_bmi.pickle "$OUT"/"$DATASET_NAME"_test_preset_30_bmi.pickle \
  --output "$OUT"/"$DATASET_NAME"_merged_preset_30_bmi.pickle

# --split train val test \
# --meta-data-names sex age bmi \