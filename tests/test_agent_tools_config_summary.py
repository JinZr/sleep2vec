from __future__ import annotations

from pathlib import Path

from agent_tool_test_helpers import config_payload, write_yaml
import yaml

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
