cd /home/notebook/code/personal/S9063410/sleep2vec
source /opt/conda/bin/activate sleep2vec

GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
DEVICES=$(seq 0 $((GPUS_PER_NODE-1)) | tr '\n' ' ')

export WANDB_MODE=offline

echo "Devices: $DEVICES"

# PRETRAIN_CKPT="log-wrist2vec-pretrain/multilight_250Hz-roformer-unsupervised/checkpoints/epoch=epoch=9-step=step=12870.ckpt"
# PRESET_DIR="/home/notebook/data/personal/S9063410/pwv+bp_data_multilight/finetune_presets"
# version_name="wrist-multilight_250Hz-10thepoch"

mkdir -p results

# mkdir -p log-wrist2vec-finetune/${version_name}-age
# python -m wrist2vec.finetune \
#   --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_250hz_finetune_reg_resnet1d.yaml \
#   --label-name age \
#   --finetune-data-index /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask_split.csv \
#   --finetune-preset-path "$PRESET_DIR/bp_multilight_merged_preset_30_age.pickle" \
#   --pretrained-backbone-path "$PRETRAIN_CKPT" \
#   --results-csv-path results/wrist_age.csv \
#   --version-name ${version_name}-age \
#   --epochs 20 --lr 1e-5 --batch-size 256 --devices $DEVICES \
#   2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-age/training_terminal_out.txt"

# mkdir -p log-wrist2vec-finetune/${version_name}-sex
# python -m wrist2vec.finetune \
#   --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_250hz_finetune_cls_resnet1d.yaml \
#   --label-name sex \
#   --finetune-data-index /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask_split.csv \
#   --finetune-preset-path "$PRESET_DIR/bp_multilight_merged_preset_30_sex.pickle" \
#   --pretrained-backbone-path "$PRETRAIN_CKPT" \
#   --results-csv-path results/wrist_sex.csv \
#   --version-name ${version_name}-sex \
#   --epochs 10 --lr 1e-5 --batch-size 512 --devices $DEVICES \
#   2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-sex/training_terminal_out.txt"

# mkdir -p log-wrist2vec-finetune/${version_name}-bmi
# python -m wrist2vec.finetune \
#   --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_250hz_finetune_reg_resnet1d.yaml \
#   --label-name bmi \
#   --finetune-data-index /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask_split.csv \
#   --finetune-preset-path "$PRESET_DIR/bp_multilight_merged_preset_30_bmi.pickle" \
#   --pretrained-backbone-path "$PRETRAIN_CKPT" \
#   --results-csv-path results/wrist_bmi.csv \
#   --version-name ${version_name}-bmi \
#   --epochs 10 --lr 1e-5 --batch-size 512 --devices $DEVICES \
#   2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-bmi/training_terminal_out.txt"



PRETRAIN_CKPT="log-wrist2vec-pretrain/multilight_250Hz-roformer-unsupervised/checkpoints/epoch=epoch=19-step=step=25740.ckpt"
PRESET_DIR="/home/notebook/data/personal/S9063410/pwv+bp_data_multilight/finetune_presets"
version_name="wrist-multilight_250Hz-20thepoch"

mkdir -p log-wrist2vec-finetune/${version_name}-age
python -m wrist2vec.finetune \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_250hz_finetune_reg_resnet1d.yaml \
  --label-name age \
  --finetune-data-index /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask_split.csv \
  --finetune-preset-path "$PRESET_DIR/bp_multilight_merged_preset_30_age.pickle" \
  --pretrained-backbone-path "$PRETRAIN_CKPT" \
  --results-csv-path results/wrist_age.csv \
  --version-name ${version_name}-age \
  --epochs 20 --lr 1e-5 --batch-size 256 --devices $DEVICES \
  2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-age/training_terminal_out.txt"

mkdir -p log-wrist2vec-finetune/${version_name}-sex
python -m wrist2vec.finetune \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_250hz_finetune_cls_resnet1d.yaml \
  --label-name sex \
  --finetune-data-index /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask_split.csv \
  --finetune-preset-path "$PRESET_DIR/bp_multilight_merged_preset_30_sex.pickle" \
  --pretrained-backbone-path "$PRETRAIN_CKPT" \
  --results-csv-path results/wrist_sex.csv \
  --version-name ${version_name}-sex \
  --epochs 10 --lr 1e-5 --batch-size 512 --devices $DEVICES \
  2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-sex/training_terminal_out.txt"

mkdir -p log-wrist2vec-finetune/${version_name}-bmi
python -m wrist2vec.finetune \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_250hz_finetune_reg_resnet1d.yaml \
  --label-name bmi \
  --finetune-data-index /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask_split.csv \
  --finetune-preset-path "$PRESET_DIR/bp_multilight_merged_preset_30_bmi.pickle" \
  --pretrained-backbone-path "$PRETRAIN_CKPT" \
  --results-csv-path results/wrist_bmi.csv \
  --version-name ${version_name}-bmi \
  --epochs 10 --lr 1e-5 --batch-size 512 --devices $DEVICES \
  2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-bmi/training_terminal_out.txt"



PRETRAIN_CKPT="log-wrist2vec-pretrain/multilight_250Hz-roformer-unsupervised/checkpoints/epoch=epoch=29-step=step=38610.ckpt"
PRESET_DIR="/home/notebook/data/personal/S9063410/pwv+bp_data_multilight/finetune_presets"
version_name="wrist-multilight_250Hz-30thepoch"

mkdir -p log-wrist2vec-finetune/${version_name}-age
python -m wrist2vec.finetune \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_250hz_finetune_reg_resnet1d.yaml \
  --label-name age \
  --finetune-data-index /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask_split.csv \
  --finetune-preset-path "$PRESET_DIR/bp_multilight_merged_preset_30_age.pickle" \
  --pretrained-backbone-path "$PRETRAIN_CKPT" \
  --results-csv-path results/wrist_age.csv \
  --version-name ${version_name}-age \
  --epochs 20 --lr 1e-5 --batch-size 256 --devices $DEVICES \
  2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-age/training_terminal_out.txt"

mkdir -p log-wrist2vec-finetune/${version_name}-sex
python -m wrist2vec.finetune \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_250hz_finetune_cls_resnet1d.yaml \
  --label-name sex \
  --finetune-data-index /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask_split.csv \
  --finetune-preset-path "$PRESET_DIR/bp_multilight_merged_preset_30_sex.pickle" \
  --pretrained-backbone-path "$PRETRAIN_CKPT" \
  --results-csv-path results/wrist_sex.csv \
  --version-name ${version_name}-sex \
  --epochs 10 --lr 1e-5 --batch-size 512 --devices $DEVICES \
  2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-sex/training_terminal_out.txt"

mkdir -p log-wrist2vec-finetune/${version_name}-bmi
python -m wrist2vec.finetune \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_250hz_finetune_reg_resnet1d.yaml \
  --label-name bmi \
  --finetune-data-index /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index_mask_split.csv \
  --finetune-preset-path "$PRESET_DIR/bp_multilight_merged_preset_30_bmi.pickle" \
  --pretrained-backbone-path "$PRETRAIN_CKPT" \
  --results-csv-path results/wrist_bmi.csv \
  --version-name ${version_name}-bmi \
  --epochs 10 --lr 1e-5 --batch-size 512 --devices $DEVICES \
  2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-bmi/training_terminal_out.txt"