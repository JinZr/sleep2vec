import importlib

import pytest
import torch


@pytest.mark.parametrize("package", ["sleep2vec", "sleep2vec2", "sleep2expert"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_diagnostic_eigenvalues_support_mixed_precision(package, dtype, capsys):
    diagnostics = importlib.import_module(f"{package}.diagnostics")
    values = torch.arange(24, dtype=torch.float32).reshape(4, 6).div(24).to(dtype)
    original = values.clone()
    diagnostic = diagnostics.TensorDiagnostic(diagnostics.TensorDiagnosticOptions(), "mixed_precision")
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


@pytest.mark.parametrize("package", ["sleep2vec", "sleep2vec2", "sleep2expert"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tuple_output", [False, True])
def test_attached_diagnostics_record_outputs_and_gradients(package, dtype, tuple_output, capsys):
    diagnostics = importlib.import_module(f"{package}.diagnostics")

    class Model(torch.nn.Module):
        def forward(self, values):
            outputs = (values * 2, values * 3)
            return outputs if tuple_output else outputs[0]

    model = Model()
    diagnostic = diagnostics.attach_diagnostics(model)
    values = torch.ones(4, 6, dtype=dtype, requires_grad=True)
    outputs = model(values)
    loss = sum(output.sum() for output in outputs) if tuple_output else outputs.sum()
    loss.backward()

    for suffix in ("[0]", "[1]") if tuple_output else ("",):
        assert f"<top-level>.output{suffix}" in diagnostic.diagnostics
        assert f"<top-level>.grad{suffix}" in diagnostic.diagnostics
    assert torch.equal(values.grad, torch.full_like(values, 5 if tuple_output else 2))

    diagnostic.print_diagnostics()

    output = capsys.readouterr().out
    assert "<top-level>.output" in output
    assert "<top-level>.grad" in output
    assert "Error getting eigenvalues" not in output
