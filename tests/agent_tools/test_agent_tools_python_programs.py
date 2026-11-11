from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest

from agent_tools import process_identity, python_programs

SOURCE_HASHES = {
    "managed_scheduler.cli_preflight": "5620962bcbd5db619b4f934773ad9e3d67e2c6ddc1edc212fac7cda872cb0cd1",
    "managed_scheduler.process_launch": "7080cc8c29513ba594951e2f65304986433088ce94b0c418092aeec34c89221e",
    "managed_scheduler.runtime_identity": "ad21db85b664d221a18430c9e053c7626c4055036fd71b653090d87ee334ff96",
    "plan_rendering.commit_status": "7112cb97e0d1ac2bce6fef0c4c453a1dd7adc307eaca04ae7cd66d52416531c9",
    "plan_rendering.verify_input_snapshots": "8c07f2508676126a0362ad9bd080db823a3649d587c00dcf90051149b1fa1701",
}


@pytest.mark.parametrize(("name", "expected_sha256"), SOURCE_HASHES.items())
def test_python_program_source_bytes_and_syntax(name, expected_sha256):
    source = python_programs.source(name)

    assert hashlib.sha256(source.encode()).hexdigest() == expected_sha256
    compile(source, name, "exec")


def test_launch_wrapper_embeds_the_imported_process_identity_source():
    # The launch-time start token and the one run_evidence compares later must come from the same bytes.
    module_source = Path(process_identity.__file__).read_text(encoding="utf-8")

    assert module_source in python_programs.source("managed_scheduler.process_launch")


@pytest.mark.parametrize("name", ["run_evidence.unknown", "run_evidence.process_probe"])
def test_unknown_python_program_name_is_rejected(name):
    with pytest.raises(KeyError):
        python_programs.source(name)


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
