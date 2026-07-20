from __future__ import annotations

import pytest

from agent_tools.models import REPO_ROOT
from agent_tools.plan_hparam import validate_final_eval_config_bytes


@pytest.mark.parametrize(
    ("variant", "config_path"),
    [
        ("sleep2vec2", "configs/sleep2vec2/sleep2vec_dense_finetune_cls.yaml"),
        ("sleep2expert", "configs/sleep2expert/moe/sleep2expert_phase_moe_finetune_cls.yaml"),
        ("sex_age_baseline", "configs/sex_age_baseline/cox.yaml"),
    ],
)
def test_final_eval_config_bytes_use_variant_loader(variant: str, config_path: str):
    validate_final_eval_config_bytes({"variant": variant}, (REPO_ROOT / config_path).read_bytes())

    with pytest.raises(ValueError):
        validate_final_eval_config_bytes({"variant": variant}, b"{}\n")
