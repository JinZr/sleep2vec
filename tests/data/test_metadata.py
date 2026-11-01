from importlib import import_module
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("data_package", ["data", "sleep2vec2.data", "sleep2expert.data"])
def test_process_metadata_preserves_fractional_age(data_package):
    values = [45.9, "60.25", 42, None, "nan", float("nan"), float("inf"), "-inf"]
    samples = [SimpleNamespace(metadata={"age": value}) for value in values]

    metadata = import_module(f"{data_package}.metadata").process_metadata(samples, [])

    torch.testing.assert_close(metadata["age"], torch.tensor([45.9, 60.25, 42.0, -1.0, -1.0, -1.0, -1.0, -1.0]))
