from __future__ import annotations

import hashlib
import importlib

import pytest

from agent_tools import python_programs, transport

SOURCE_HASHES = {
    "experiment_io.conditional_atomic_replace_text": "350a1adcc8b0b7ee44609886e2a9447f6ab48381a6365b85d9f5cb9ad70eee5a",
    "experiment_io.list_managed_subdirectories": "e8b83f0cef4f3cf46b84542276f7ffaf7045acaf4f409c9ad2d0333178b39c8b",
    "experiment_io.path_exists": "ae3b150f5e8e90493d30a016323529509d00519ba9505185f293d4481564fe17",
    "experiment_io.read_managed_files": "acf0d66282bd5bb74c506d165ee1bd0e88f219c78e9678e75951c6e385538002",
    "experiment_io.read_managed_output_texts": "5fd7bd1da8cf83fb26034f9cd6eaa55f6548f620294765adf36e1c4486360c2e",
    "experiment_io.read_text": "98a9a6036e73c9e43cc87ec31103a1185f0a51a62dcdf510c77c3cbe63d39f64",
    "experiment_io.remote_dir_nonempty": "d9604eaedee251e2b5a8ef5f055344f0e51c976c4c9c2420df7ffb4c0343bc77",
    "experiment_io.validate_managed_output_paths": "59198ca38ad76b6ee88bf33d524c59f2a47cde862a56931bbcac170033581df9",
    "experiment_workspace.write_run_matrix_if_current": (
        "82220ec43daaa0e9244a6a30fddabfba5a6ee7e7a5406bcd1d42a74bca38c0b9"
    ),
    "hparam_postprocess.verify_checkpoint_sha256": "edff342d976933beabffc01bfc98080c49cdbe673d1a9ec8e8958989fef569ee",
    "managed_scheduler.cli_preflight": "5620962bcbd5db619b4f934773ad9e3d67e2c6ddc1edc212fac7cda872cb0cd1",
    "managed_scheduler.process_launch": "47c3ab7745ce2f24f431540b08b5bd73ce1396413a0b8beab6b95d6aa5d15755",
    "managed_scheduler.runtime_identity": "1bd7d8eb05d6767c75b60192b6b8acb3682a99df8e4e1b98909472e1b7d9e2ed",
    "plan_rendering.commit_status": "7112cb97e0d1ac2bce6fef0c4c453a1dd7adc307eaca04ae7cd66d52416531c9",
    "plan_rendering.verify_input_snapshots": "8c07f2508676126a0362ad9bd080db823a3649d587c00dcf90051149b1fa1701",
    "run_evidence.checkpoint_file_sha256": "96caa2e935be9c7b31c1da2571dc6f94847545281d3cd70589d4979eefd78744",
    "run_evidence.log_tail": "1d2a2315d84c58ae257637186fce6976085eca03fcec4b69294dd2b996c57561",
    "run_evidence.log_tail_and_age": "27b2b825b3337cbc87a404214c73b80371a3d9999ac5e061d8a08efe0fd2c1e2",
    "run_evidence.process_probe": "e2756bdde3a1a1c801e9d1ceab4656fea92c2cdddd5b266f185d98e615b12057",
    "run_evidence.process_stop": "779fead9d94d0aca781168c74f3f639a33cd4f4aca17d2db4f9cc477996e6f79",
    "run_evidence.read_pid_text": "f206628bfc983a7d923624bb341c77156d9242804f8e24f179196ca3343e17cf",
    "run_evidence.runtime_artifacts": "3d03f0f32d450801d0179eb5280081765e414abca2d716b8ab80b1e42eabd36e",
    "runtime_sync.sync": "18690564fc6fc39bdf912c39bc45fa334c860e537acff74768f8dd3072598bfa",
    "slurm.worker_bootstrap": "f9a83db38ee5b9c55922c47bda684bebd909d7231f576e0d5d5616d7beb5780e",
}

COMMAND_HASHES = {
    "experiment_io.conditional_atomic_replace_text": "2448666cea677a03b2e2401758c3ac3ff72752fb108c4bcae83f4468e6aecbc1",
    "experiment_io.list_managed_subdirectories": "87f1c489bbff353f89d4418104abbb691fe6b2ba5ee9245815eb0ae1369e2921",
    "experiment_io.path_exists": "1d4f605c0e2155db58da12c08f298901dc23d636bc9dd87fd24f97939a4e33e6",
    "experiment_io.read_managed_files": "43836ec1178eee36b1b54788554a977047552565245e8d54413370e8d57d6da0",
    "experiment_io.read_managed_output_texts": "d1445b6b99b4c0441495c35e2f26dfbbe469ad3be83446882407c9c1fac8cfd0",
    "experiment_io.read_text": "6a0e31af6006881d55b1109f58f0dfea2c332e503c6315741261569eb9b4edc0",
    "experiment_io.remote_dir_nonempty": "bea5ad87ff7a0921736c954d84dec11f66a8d3079db464bb00a3b6328a0c60fc",
    "experiment_io.validate_managed_output_paths": "565198db56507fa2b584cd6c8b055d685ca2db9e47bd3d47114055d6f81ee69e",
    "experiment_workspace.write_run_matrix_if_current": (
        "89160398c17ccd32df9d2949fe4e6273de5f0ccc62dbc1da514c4aa9fbe9f7eb"
    ),
    "hparam_postprocess.verify_checkpoint_sha256": "0423ef2081129f2e665a1128ccacb092120596d35dd57f8093782ebc3a737869",
    "managed_scheduler.cli_preflight": "f29964331539fb594e8736b27be9aa4005980f493df994833f00a5364c837418",
    "managed_scheduler.process_launch": "a67bc254b96f99d4bfef8126cfef593712abf6912a2d85b0229fc390d0173c4a",
    "managed_scheduler.runtime_identity": "04d157788ea0c9dff28038d9804a5f1c48dfef36d909b456a30430f43ce863b8",
    "plan_rendering.commit_status": "5c20b6ed04d665521d6bd3e3bd917892e6f0548a1a71d0bf227ff43884d4eb7f",
    "plan_rendering.verify_input_snapshots": "71e4b5683071833f63e1abbf8e578b58cd99debb9731a67fd2c77f3677c7a4b7",
    "run_evidence.checkpoint_file_sha256": "54d7f2d05c2e5d5bbd4adc0cd78201abef73ca8ec794f4b8d213f11b55dca385",
    "run_evidence.log_tail": "ad16cebbe78345cf00616c27e62bfa8fed385b7506b2ef0d301774d89b86c2b0",
    "run_evidence.log_tail_and_age": "7af545c333e25088acda73ff9fd753dcd5ec60c1dc0b052d22b6b7431a025f07",
    "run_evidence.process_probe": "f19223a263e07ffd5d46e110423f8d1dfa4317249940d5cec3debd7a1f8154a8",
    "run_evidence.process_stop": "14308dbf62b516eecec8cf28c777a20a72b7745d77841fb446663aa1bcb97310",
    "run_evidence.read_pid_text": "703861bd3a4230efb30fb4801465c2ec0b2902004519f7bf1428d35ec2405519",
    "run_evidence.runtime_artifacts": "855ce681a6a858d31dc68eb930cbef4e9d832ca1e75459f4e0c47ef41eb7e5ba",
    "runtime_sync.sync": "f8a0f442da33bcc09393b66667872cc01a264beea274ca1f463118a8095f77e9",
    "slurm.worker_bootstrap": "3a5d44fd6ac4fb9ee6a4f75adc949d54cfed044ae1a6922e2fea871a3c7c9fce",
}


@pytest.mark.parametrize(("name", "expected_sha256"), SOURCE_HASHES.items())
def test_python_program_source_bytes_and_syntax(name, expected_sha256):
    source = python_programs.source(name)

    assert hashlib.sha256(source.encode()).hexdigest() == expected_sha256
    compile(source, name, "exec")


@pytest.mark.parametrize(("name", "expected_sha256"), COMMAND_HASHES.items())
def test_remote_python_program_command_bytes(name, expected_sha256):
    command = transport.remote_python_program_command(name, "argument with spaces", 17)

    assert hashlib.sha256(command.encode()).hexdigest() == expected_sha256
    assert command == transport.remote_python_command(python_programs.source(name), "argument with spaces", 17)


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
        "agent_tools.hparam_postprocess",
    ],
)
def test_python_program_owners_import(module_name):
    importlib.import_module(module_name)
