from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ENTRYPOINTS = (
    Path("sleep2vec/pretrain.py"),
    Path("sleep2vec/finetune.py"),
    Path("sleep2vec/adapt.py"),
    Path("sleep2vec2/pretrain.py"),
    Path("sleep2vec2/finetune.py"),
    Path("sleep2vec2/adapt.py"),
    Path("sleep2expert/pretrain.py"),
    Path("sleep2expert/finetune.py"),
    Path("sleep2expert/adapt.py"),
)


def _parse(relative_path: Path) -> ast.Module:
    return ast.parse((REPO_ROOT / relative_path).read_text())


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _is_add_argument_call(node: ast.AST, argument_name: str) -> bool:
    if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "add_argument":
        return False
    return any(isinstance(arg, ast.Constant) and arg.value == argument_name for arg in node.args)


@pytest.mark.parametrize("relative_path", TRAINING_ENTRYPOINTS)
def test_training_entrypoints_define_accumulate_grad_batches_cli(relative_path: Path):
    calls = [
        node
        for node in ast.walk(_parse(relative_path))
        if _is_add_argument_call(node, "--accumulate-grad-batches")
    ]

    assert len(calls) == 1
    arg_type = _keyword(calls[0], "type")
    default = _keyword(calls[0], "default")

    assert isinstance(arg_type, ast.Name)
    assert arg_type.id == "int"
    assert isinstance(default, ast.Constant)
    assert default.value == 1


@pytest.mark.parametrize("relative_path", TRAINING_ENTRYPOINTS)
def test_training_entrypoints_pass_accumulate_grad_batches_to_trainer(relative_path: Path):
    for node in ast.walk(_parse(relative_path)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "trainer_kwargs" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name):
            continue
        if node.value.func.id != "dict":
            continue

        value = _keyword(node.value, "accumulate_grad_batches")
        if value is None:
            continue

        assert isinstance(value, ast.Attribute)
        assert value.attr == "accumulate_grad_batches"
        assert isinstance(value.value, ast.Name)
        assert value.value.id == "args"
        return

    pytest.fail(f"{relative_path} does not pass accumulate_grad_batches into trainer_kwargs")
