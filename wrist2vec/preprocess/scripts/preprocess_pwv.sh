cd /home/notebook/code/personal/S9063410/sleep2vec
source /opt/conda/bin/activate sleep2vec

# python wrist2vec/preprocess/pwv_formant_multilight.py \
#     --workers 32

python wrist2vec/preprocess/pwv_metadata.py \
    --index-csv /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pwv_index.csv \
    --demographics-csv /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pwv_label/demographics.csv \
    --out-csv /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pwv_index_mask.csv \
    --workers 32

python wrist2vec/preprocess/split_index_by_subject.py \
    --input /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pwv_index_mask.csv \
    --output /home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pwv_index_mask_split.csv