from __future__ import annotations

from pathlib import Path

import yaml


def config_payload(index_path: Path) -> dict:
    return {
        "model": {
            "backbone": {"name": "roformer", "hidden_size": 8},
            "projection": {"name": "simclr", "enabled": True},
            "cls": {"embedding_type": "bert", "downstream": "tokens"},
            "channels": [{"name": "ppg", "input_dim": 8, "tokenizer": {"name": "linear", "out_dim": 8}}],
            "head": {"name": "classification"},
        },
        "data": {
            "backend": "npz",
            "max_tokens": 4,
            "data_channel_names": ["ppg"],
            "finetune_data_index": str(index_path),
            "finetune_preset_path": None,
        },
        "finetune": {
            "task": {
                "type": "classification",
                "output_dim": 30,
                "is_seq": True,
                "monitor": "val_ahi_pearson",
                "monitor_mod": "max",
            }
        },
        "preset_build": {"required_channels": ["ppg", "ahi", "stage5"], "min_channels": 3},
    }


def write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload))
    return path


def write_finetune_recipe(tmp_path: Path, *, include_label: bool = True, variant: str = "sleep2vec") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,ppg_mask,ah_event_mask,stage_mask\nx.npz,train,60,1,1,1\n")
    config = write_yaml(tmp_path / "config.yaml", config_payload(index))
    inputs = {"config": str(config), "pretrained_backbone_path": None}
    if include_label:
        inputs["label_name"] = "ahi"
    recipe = {
        "name": "unit_finetune",
        "task": "finetune",
        "variant": variant,
        "inputs": inputs,
        "runtime": {"devices": [0]},
        "artifacts": {"results_csv_path": str(tmp_path / "results.csv"), "version_name": "unit"},
        "evaluation_policy": {
            "selection_metric": "val_ahi_pearson",
            "selection_mode": "max",
            "selection_split": "val",
            "external_test_locked": True,
            "test_after_fit": False,
        },
        "decisions": {
            "task": {"value": "finetune", "source": "explicit_recipe"},
            "pretrained_backbone_path": {
                "value": None,
                "source": "explicit_recipe",
                "meaning": "train from scratch",
            },
            "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
            "overwrite_policy": {"value": False, "source": "explicit_recipe"},
        },
    }
    return write_yaml(tmp_path / "recipe.yaml", recipe)
