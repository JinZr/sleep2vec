from data.psg_pretrain_dataset import PSGPretrainDataset

kwargs = {
    "batch_size": 16,
    "shuffle": True,
}

meta_data_name = None
# allergiesorsinusproblems,asthma,bronchitis,cerebrovasculardisease,
# chronicobstructivepulmonarydiseasecopd,coronaryheartdisease,diabetes,
# heartfailure,hypertension,restlesslegsyndromerls
# meta_data_name = 'hypertension'
meta_data_names = [
    "hypertension",
    # 'bronchitis',
    # 'cerebrovasculardisease',
    # 'chronicobstructivepulmonarydiseasecopd',
    # 'heartfailure',
    # 'restlesslegsyndromerls'
]

# for split in ['test']:
for meta_data_name in meta_data_names:
    for split in ["test", "val", "train"]:

        # n_tokens = 120
        n_tokens = 1535

        meta_data_suffix = "" if not meta_data_name else f"_{meta_data_name}"

        save_preset_path = f"/data/ywx/BIOT/data/shhs_{split}_preset_{n_tokens}{meta_data_suffix}.pickle"
        # save_preset_path = f"/data/ywx/BIOT/5dataset_{split}_preset_{n_tokens}{meta_data_suffix}.pickle"
        load_preset_path = None

        dataset = PSGPretrainDataset(
            [
                "heartbeat",
                "breath",
                "eeg_original",
                "ecg_original",
                "eog_original",
                "emg_original",
                "spo2",
                "resp_original",
                "resp_nasal_original",
            ],
            save_preset_path,
            load_preset_path,
            index=[
                # 'index/mros_psg_pretrain.csv',
                # 'index/mesa_psg_pretrain.csv',
                # 'index/wsc_psg_pretrain.csv',
                # 'index/shhs_psg_pretrain.csv',
                # 'index/hsp_psg_pretrain.csv',
                "index/shhs_d_merged_with_diseases.csv",
            ],
            meta_data_names=[meta_data_name] if meta_data_name else [],
            split=split,
            max_tokens=n_tokens,
            stride_tokens=0 if n_tokens == 1535 else n_tokens,  # 0 for truncation
            mask_rate=0.0,
            use_legacy_body_movement=False,
            allow_missing_channels=True,
            min_channels=2,
            **kwargs,
        )

        print(f"Dataset size: {len(dataset)}")
