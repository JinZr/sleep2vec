from __future__ import annotations

from pathlib import Path
import sys

from agent_tool_test_helpers import config_payload, survival_config_payload, write_survival_sidecars, write_yaml

from agent_tools.configs import config_summary


def test_config_summary_extracts_channels_task_backend_and_monitor(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration\nx.npz,train,60\n")
    payload = config_payload(index)
    payload["model"]["head"]["dropout"] = 0.2
    payload["model"]["head"]["hidden_dim"] = 1024
    payload["model"]["head"]["channel_agg"] = {"name": "gated_scalar", "kwargs": {}}
    payload["model"]["head"]["temporal_agg"] = {"name": "attn", "kwargs": {"heads": 2}}
    payload["finetune"]["freeze_tokenizer"] = True
    payload["finetune"]["layer_mix"] = {
        "enabled": True,
        "shared_across_modalities": False,
        "layer_indices": [15, 16],
    }
    payload["finetune"]["lora"] = {"freeze_backbone_and_insert_lora": True, "insert_lora": False}
    payload["model_averaging"] = {"name": "ema", "params": {"enabled": True}}
    config = write_yaml(tmp_path / "config.yaml", payload)

    summary = config_summary(config)

    assert summary["data_backend"] == "npz"
    assert summary["model"]["channels"][0]["name"] == "ppg"
    assert summary["model"]["cls"]["downstream"] == "tokens"
    assert summary["model"]["head_details"]["dropout"] == 0.2
    assert summary["model"]["head_details"]["hidden_dim"] == 1024
    assert summary["model"]["head_details"]["channel_agg"]["name"] == "gated_scalar"
    assert summary["model"]["head_details"]["temporal_agg"]["name"] == "attn"
    assert summary["model"]["layer_mix"]["layer_indices"] == [15, 16]
    assert summary["model"]["freeze"]["freeze_tokenizer"] is True
    assert summary["model"]["model_averaging"]["present"] is True
    assert summary["finetune"]["task"]["monitor"] == "val_ahi_pearson"
    assert summary["preset_build"]["required_channels"] == ["ppg", "ahi", "stage5"]


def test_config_summary_validates_survival_sidecars(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,eid\nx.npz,train,60,001\n")
    config = write_yaml(
        tmp_path / "survival.yaml",
        survival_config_payload(index, write_survival_sidecars(tmp_path)),
    )

    summary = config_summary(config)
    survival = summary["finetune"]["survival"]

    assert survival["valid"] is True
    assert survival["key_column"] == "eid"
    assert survival["covariates"] == []
    assert survival["covariate_embedding_dim"] == 16
    assert survival["disease_count"] == 2
    assert survival["sidecar_key_count"] == 2
    assert survival["issues"] == []


def test_config_summary_reports_survival_covariates(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,eid\nx.npz,train,60,001\n")
    payload = survival_config_payload(index, write_survival_sidecars(tmp_path))
    payload["finetune"]["survival"].update({"covariates": ["age", "sex"], "covariate_embedding_dim": 8})
    config = write_yaml(tmp_path / "survival_covariates.yaml", payload)

    survival = config_summary(config)["finetune"]["survival"]

    assert survival["covariates"] == ["age", "sex"]
    assert survival["covariate_embedding_dim"] == 8


def test_config_summary_validates_survival_sidecars_without_torch(tmp_path: Path, monkeypatch):
    import builtins

    sys.modules.pop("data.survival", None)
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,eid\nx.npz,train,60,001\n")
    config = write_yaml(
        tmp_path / "survival.yaml",
        survival_config_payload(index, write_survival_sidecars(tmp_path)),
    )

    survival = config_summary(config)["finetune"]["survival"]

    assert survival["valid"] is True
    assert survival["issues"] == []


def test_config_summary_reports_placeholder_survival_paths(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,eid\nx.npz,train,60,001\n")
    payload = survival_config_payload(
        index,
        {
            "disease_columns_index": "/path/to/disease_columns.txt",
            "event_time_index": "/path/to/event_time.csv",
            "is_event_index": "/path/to/is_event.csv",
            "has_label_index": "/path/to/has_label.csv",
        },
    )
    config = write_yaml(tmp_path / "survival_placeholder.yaml", payload)

    survival = config_summary(config)["finetune"]["survival"]

    assert survival["valid"] is False
    assert len(survival["issues"]) == 4
    assert any("disease_columns_index" in issue for issue in survival["issues"])


def test_config_summary_reports_survival_output_dim_mismatch(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,eid\nx.npz,train,60,001\n")
    config = write_yaml(
        tmp_path / "survival_bad_dim.yaml",
        survival_config_payload(index, write_survival_sidecars(tmp_path), output_dim=3),
    )

    survival = config_summary(config)["finetune"]["survival"]

    assert survival["valid"] is False
    assert any("output_dim" in issue for issue in survival["issues"])
