from __future__ import annotations

import hashlib
import importlib

import pytest

from agent_tools import python_programs, transport

SOURCE_HASHES = {
    "managed_scheduler.cli_preflight": "5620962bcbd5db619b4f934773ad9e3d67e2c6ddc1edc212fac7cda872cb0cd1",
    "managed_scheduler.process_launch": "0ad40681fac86dc521274271c68cfa0661e24176fc1d45d6ea25c553f15a1634",
    "managed_scheduler.runtime_identity": "ad21db85b664d221a18430c9e053c7626c4055036fd71b653090d87ee334ff96",
    "plan_rendering.commit_status": "7112cb97e0d1ac2bce6fef0c4c453a1dd7adc307eaca04ae7cd66d52416531c9",
    "plan_rendering.verify_input_snapshots": "8c07f2508676126a0362ad9bd080db823a3649d587c00dcf90051149b1fa1701",
    "run_evidence.process_probe": "e2756bdde3a1a1c801e9d1ceab4656fea92c2cdddd5b266f185d98e615b12057",
}

COMMAND_HASHES = {
    "managed_scheduler.cli_preflight": "f29964331539fb594e8736b27be9aa4005980f493df994833f00a5364c837418",
    "managed_scheduler.process_launch": "1c04cb9a82c8e4b39b74e0e2788023c410c814f685190220e1321829e32dcc9f",
    "managed_scheduler.runtime_identity": "dca67842bd1be75e69edce2cf03a1562a6c6d525bdae55f600b3ce33435e02be",
    "plan_rendering.commit_status": "5c20b6ed04d665521d6bd3e3bd917892e6f0548a1a71d0bf227ff43884d4eb7f",
    "plan_rendering.verify_input_snapshots": "71e4b5683071833f63e1abbf8e578b58cd99debb9731a67fd2c77f3677c7a4b7",
    "run_evidence.process_probe": "f19223a263e07ffd5d46e110423f8d1dfa4317249940d5cec3debd7a1f8154a8",
}


@pytest.mark.parametrize(("name", "expected_sha256"), SOURCE_HASHES.items())
def test_python_program_source_bytes_and_syntax(name, expected_sha256):
    source = python_programs.source(name)

    assert hashlib.sha256(source.encode()).hexdigest() == expected_sha256
    compile(source, name, "exec")


@pytest.mark.parametrize(("name", "expected_sha256"), COMMAND_HASHES.items())
def test_python_program_command_bytes(name, expected_sha256):
    command = transport.python_program_command(name, "argument with spaces", 17)

    assert hashlib.sha256(command.encode()).hexdigest() == expected_sha256
    assert command == f"python3 -c {transport.sh(python_programs.source(name))} 'argument with spaces' 17"


def test_unknown_python_program_name_is_rejected():
    with pytest.raises(KeyError):
        python_programs.source("run_evidence.unknown")


@pytest.mark.parametrize(
    "module_name",
    [
        "agent_tools.experiment_io",
        "agent_tools.experiment_workspace",
        "agent_tools.run_evidence",
        "agent_tools.managed_scheduler",
        "agent_tools.plan_rendering",
    ],
)
def test_python_program_owners_import(module_name):
    importlib.import_module(module_name)
