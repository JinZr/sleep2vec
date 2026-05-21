cd /home/notebook/code/personal/S9063410/sleep2vec
source /opt/conda/bin/activate sleep2vec

GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
DEVICES=$(seq 0 $((GPUS_PER_NODE-1)) | tr '\n' ' ')

export WANDB_MODE=offline

echo "Devices: $DEVICES"

PRETRAIN_CKPT="log-wrist2vec-pretrain/multilight-correct_split-roformer-unsupervised/checkpoints/epoch=epoch=9-step=step=12880.ckpt"
PRESET_DIR="/home/notebook/data/personal/S9063410/bp_data_multilight/finetune_presets"
version_name="wrist-multilight-10epoch"

mkdir -p results

python -m wrist2vec.finetune \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_finetune_reg_resnet1d.yaml \
  --label-name age \
  --finetune-data-index /home/notebook/data/personal/S9063410/bp_data_multilight/index_mask_correct_split.csv \
  --finetune-preset-path "$PRESET_DIR/wrist_multilight_cs_merged_preset_30_age.pickle" \
  --pretrained-backbone-path "$PRETRAIN_CKPT" \
  --results-csv-path results/wrist_age.csv \
  --version-name ${version_name}-age \
  --epochs 20 --lr 1e-5 --batch-size 256 --devices $DEVICES \
  2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-age/training_terminal_out.txt"

python -m wrist2vec.finetune \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_finetune_cls_resnet1d.yaml \
  --label-name sex \
  --finetune-data-index /home/notebook/data/personal/S9063410/bp_data_multilight/index_mask_correct_split.csv \
  --finetune-preset-path "$PRESET_DIR/wrist_multilight_cs_merged_preset_30_sex.pickle" \
  --pretrained-backbone-path "$PRETRAIN_CKPT" \
  --results-csv-path results/wrist_sex.csv \
  --version-name ${version_name}-sex \
  --epochs 10 --lr 1e-5 --batch-size 512 --devices $DEVICES \
  2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-sex/training_terminal_out.txt"

python -m wrist2vec.finetune \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_finetune_reg_resnet1d.yaml \
  --label-name bmi \
  --finetune-data-index /home/notebook/data/personal/S9063410/bp_data_multilight/index_mask_correct_split.csv \
  --finetune-preset-path "$PRESET_DIR/wrist_multilight_cs_merged_preset_30_bmi.pickle" \
  --pretrained-backbone-path "$PRETRAIN_CKPT" \
  --results-csv-path results/wrist_bmi.csv \
  --version-name ${version_name}-bmi \
  --epochs 10 --lr 1e-5 --batch-size 512 --devices $DEVICES \
  2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-bmi/training_terminal_out.txt"



PRETRAIN_CKPT="log-wrist2vec-pretrain/multilight-correct_split-roformer-unsupervised/checkpoints/epoch=epoch=19-step=step=25760.ckpt"
PRESET_DIR="/home/notebook/data/personal/S9063410/bp_data_multilight/finetune_presets"
version_name="wrist-multilight-20epoch"

python -m wrist2vec.finetune \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_finetune_reg_resnet1d.yaml \
  --label-name age \
  --finetune-data-index /home/notebook/data/personal/S9063410/bp_data_multilight/index_mask_correct_split.csv \
  --finetune-preset-path "$PRESET_DIR/wrist_multilight_cs_merged_preset_30_age.pickle" \
  --pretrained-backbone-path "$PRETRAIN_CKPT" \
  --results-csv-path results/wrist_age.csv \
  --version-name ${version_name}-age \
  --epochs 20 --lr 1e-5 --batch-size 256 --devices $DEVICES \
  2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-age/training_terminal_out.txt"

python -m wrist2vec.finetune \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_finetune_cls_resnet1d.yaml \
  --label-name sex \
  --finetune-data-index /home/notebook/data/personal/S9063410/bp_data_multilight/index_mask_correct_split.csv \
  --finetune-preset-path "$PRESET_DIR/wrist_multilight_cs_merged_preset_30_sex.pickle" \
  --pretrained-backbone-path "$PRETRAIN_CKPT" \
  --results-csv-path results/wrist_sex.csv \
  --version-name ${version_name}-sex \
  --epochs 10 --lr 1e-5 --batch-size 512 --devices $DEVICES \
  2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-sex/training_terminal_out.txt"

python -m wrist2vec.finetune \
  --config configs/write2vec/wrist2vec_multilight_ppg_accgyro_finetune_reg_resnet1d.yaml \
  --label-name bmi \
  --finetune-data-index /home/notebook/data/personal/S9063410/bp_data_multilight/index_mask_correct_split.csv \
  --finetune-preset-path "$PRESET_DIR/wrist_multilight_cs_merged_preset_30_bmi.pickle" \
  --pretrained-backbone-path "$PRETRAIN_CKPT" \
  --results-csv-path results/wrist_bmi.csv \
  --version-name ${version_name}-bmi \
  --epochs 10 --lr 1e-5 --batch-size 512 --devices $DEVICES \
  2>&1 | tee -a "log-wrist2vec-finetune/${version_name}-bmi/training_terminal_out.txt"