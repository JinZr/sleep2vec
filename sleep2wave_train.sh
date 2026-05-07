#!/usr/bin/env bash
set -euo pipefail

mkdir -p data checkpoints

python -m sleep2wave.preprocess.merge_dataset_presets \
  --inputs \
    index_iclr2/presets_gen/hsp_psg_pretrain.pkl \
    index_iclr2/presets_gen/mesa_psg_pretrain.pkl \
    index_iclr2/presets_gen/mros_psg_pretrain.pkl \
    index_iclr2/presets_gen/shhs_psg_pretrain.pkl \
    index_iclr2/presets_gen/wsc_psg_pretrain.pkl \
  --output index_iclr2/presets_gen/sleep2wave_medium_preset.pkl

ln -sf "$(pwd)/index_iclr2/presets_gen/sleep2wave_medium_preset.pkl" \
  data/sleep2wave_medium_preset.pkl

python -m sleep2wave.train_autoencoder \
  --config configs/sleep2wave/sleep2wave_autoencoder_medium.yaml \
  --version-name ae-medium-v1 \
  --accelerator gpu \
  --devices 4,5 \
  --precision 16 \
  --num-workers 8

ln -sf "$(pwd)/outputs/sleep2wave_autoencoder_medium/ae-medium-v1/checkpoints/last.ckpt" \
  checkpoints/sleep2wave_autoencoder_medium.ckpt

for phase in 1 2 3 4 5; do
  python -m sleep2wave.train_diffusion \
    --config configs/sleep2wave/sleep2wave_diffusion_medium_phase${phase}.yaml \
    --version-name phase${phase} \
    --accelerator gpu \
    --devices 4,5 \
    --precision 16 \
    --num-workers 8 \
    --seed 0
done
