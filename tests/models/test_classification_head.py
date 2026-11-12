import importlib

import pytest
import torch
import torch.nn as nn


@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
@pytest.mark.parametrize("is_sequence", [False, True])
def test_concat_projects_each_channel_and_preserves_order(package_name: str, is_sequence: bool):
    module = importlib.import_module(f"{package_name}.downstreams.channel_aggregation")
    aggregator = module.build_channel_aggregator("concat", feature_dim=4, n_mods=2, projection_dim=2)
    with torch.no_grad():
        aggregator.projection.weight.copy_(torch.eye(4)[:2])
        aggregator.projection.bias.zero_()
    first = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    if not is_sequence:
        first = first[:, 0]
    second = first + 24

    fused, has_sequence = aggregator([first, second])

    assert has_sequence is is_sequence
    torch.testing.assert_close(fused, torch.cat([first[..., :2], second[..., :2]], dim=-1))


@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
@pytest.mark.parametrize(
    ("num_layers", "expected_dims"),
    [
        (1, [(8, 5)]),
        (2, [(8, 6), (6, 5)]),
        (3, [(8, 6), (6, 6), (6, 5)]),
    ],
)
def test_classification_head_num_layers(package_name: str, num_layers: int, expected_dims: list[tuple[int, int]]):
    module = importlib.import_module(f"{package_name}.downstreams.heads.classification")
    head = module.build_classification_head(
        target="stage5",
        feature_dim=4,
        n_mods=2,
        output_dim=5,
        agg="concat",
        hidden_dim=6,
        dropout=0.1,
        num_layers=num_layers,
    )

    linears = [layer for layer in head.mlp if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == expected_dims
    output = head([torch.ones(2, 3, 4), torch.ones(2, 3, 4)])
    assert output.shape == (2, 3, 5)


@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
@pytest.mark.parametrize("num_layers", [0, 4, True, 2.0, "3"])
def test_classification_head_rejects_invalid_num_layers(package_name: str, num_layers):
    module = importlib.import_module(f"{package_name}.downstreams.heads.classification")

    with pytest.raises(ValueError, match="num_layers must be 1, 2, or 3"):
        module.ClassificationHead(
            feature_dim=4,
            n_mods=2,
            n_classes=5,
            agg="concat",
            hidden_dim=6,
            num_layers=num_layers,
        )
