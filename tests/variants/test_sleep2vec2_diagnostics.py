import pytest
import torch

from sleep2vec2.diagnostics import TensorDiagnostic, TensorDiagnosticOptions


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_diagnostic_eigenvalues_support_mixed_precision(dtype, capsys):
    values = torch.arange(24, dtype=torch.float32).reshape(4, 6).div(24).to(dtype)
    original = values.clone()
    diagnostic = TensorDiagnostic(TensorDiagnosticOptions(), "mixed_precision")
    diagnostic.accumulate(values)
    accumulators = [stats["eigs"][0].tensor.clone() for stats in diagnostic.stats]

    diagnostic.print_diagnostics()

    output = capsys.readouterr().out
    assert "eigs" in output
    assert "Error getting eigenvalues" not in output
    assert torch.equal(values, original)
    for stats, accumulator in zip(diagnostic.stats, accumulators):
        assert stats["eigs"][0].tensor.dtype == dtype
        assert torch.equal(stats["eigs"][0].tensor, accumulator)
