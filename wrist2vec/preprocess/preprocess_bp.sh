cd /home/notebook/code/personal/S9063410/sleep2vec
source /opt/conda/bin/activate sleep2vec

cat wrist2vec/preprocess/preprocess_bp.sh
python wrist2vec/preprocess/ppg_formant_multilight.py \
    --out-root /home/notebook/data/personal/S9063410/pwv+bp_data_multilight \
    --metadata-csv /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/bp_index.csv \
    --target-fs 250 \
    --bp-low 0.5 \
    --bp-high 40.0 \
    --workers 32