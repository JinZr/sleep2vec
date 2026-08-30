from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from agent_tool_test_helpers import survival_config_payload, write_finetune_recipe, write_survival_sidecars, write_yaml
import pandas as pd
import pytest
import yaml

from agent_tools import cli, managed_scheduler, plan_context, plans
from agent_tools.configs import config_summary
from agent_tools.domain import index_csv
from agent_tools.models import REPO_ROOT


def _sidecar_recipe(tmp_path: Path, kind: str, *, variant: str = "sleep2vec", hparam: bool = False) -> Path:
    recipe = write_finetune_recipe(tmp_path, variant=variant)
    index = tmp_path / "index.csv"
    index.write_text(
        "path,split,duration,eid,ppg_mask,age,sex\n"
        "x.npz,train,60,doctor-private-001,1,50,0\n"
        "y.npz,val,60,doctor-private-002,1,60,1\n"
    )
    sidecars = write_survival_sidecars(tmp_path)
    for field in ("event_time_index", "is_event_index", "has_label_index"):
        path = Path(sidecars[field])
        path.write_text(path.read_text().replace("001", "doctor-private-001").replace("002", "doctor-private-002"))
    if variant == "sex_age_baseline":
        payload = yaml.safe_load((REPO_ROOT / "configs/sex_age_baseline/cox.yaml").read_text())
        payload["data"]["finetune_data_index"] = str(index)
        payload["finetune"]["survival"].update(sidecars)
        payload["finetune"]["task"].update(output_dim=2, monitor="val_loss", monitor_mod="min")
    else:
        payload = survival_config_payload(index, sidecars)
    if kind == "multilabel":
        payload["finetune"].pop("survival")
        payload["finetune"]["task"]["type"] = "multilabel_classification"
        payload["finetune"]["multilabel"] = {
            "key_column": "eid",
            "disease_columns_index": sidecars["disease_columns_index"],
            "label_index": sidecars["is_event_index"],
            "has_label_index": sidecars["has_label_index"],
        }
    write_yaml(tmp_path / "config.yaml", payload)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["label_name"] = "incident_cox" if kind == "survival" else "disease_label"
    payload["evaluation_policy"].update(selection_metric="val_loss", selection_mode="min")
    if variant == "sex_age_baseline":
        payload["decisions"]["required_channels"] = {"value": [], "source": "explicit_recipe"}
    write_yaml(recipe, payload)
    if not hparam:
        return recipe
    return write_yaml(
        tmp_path / "hparam.yaml",
        {
            "name": "doctor_sidecar_reuse",
            "task": "hparam_tune",
            "variant": variant,
            "base_recipe": str(recipe),
            "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-3]}},
            "evaluation_policy": {
                **payload["evaluation_policy"],
                "final_eval_split": "test",
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


@pytest.fixture
def csv_reads(monkeypatch):
    calls = Counter()
    read_csv = pd.read_csv

    def counted(path, *args, **kwargs):
        calls[Path(path).name] += 1
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", counted)
    return calls


@pytest.fixture
def runtime_probes(monkeypatch):
    calls = []

    def run(execution, command):
        calls.append((execution, command))
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "host": "test-runtime",
                    "python": "/test/python",
                    "python_version": "3.10.20",
                    "pytorch_lightning_version": "2.6.1",
                }
            ),
            "",
        )

    monkeypatch.setattr(managed_scheduler, "run_execution_command", run)
    return calls


def _without_reuse(*args, **kwargs):
    kwargs.pop("validated_summary", None)
    return index_csv.index_summary(*args, **kwargs)


@pytest.mark.parametrize("kind", ["survival", "multilabel"])
@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2", "sleep2expert", "sex_age_baseline"])
@pytest.mark.parametrize("hparam", [False, True])
def test_doctor_reuses_complete_validation_without_changing_output(
    tmp_path, monkeypatch, capsys, csv_reads, runtime_probes, kind, variant, hparam
):
    recipe = _sidecar_recipe(tmp_path, kind, variant=variant, hparam=hparam)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with monkeypatch.context() as uncached:
        uncached.setattr(plan_context, "index_summary", _without_reuse)
        previous_exit = cli.main(["doctor", "--recipe", str(recipe)])
    previous_output = capsys.readouterr()
    previous_counts = dict(csv_reads)
    assert previous_exit == 0, previous_output.out
    assert previous_counts == (
        {"event_time.csv": 3, "is_event.csv": 2, "has_label.csv": 2, "index.csv": 1}
        if kind == "survival"
        else {"is_event.csv": 3, "has_label.csv": 2, "index.csv": 1}
    )

    csv_reads.clear()
    assert cli.main(["doctor", "--recipe", str(recipe)]) == previous_exit
    assert capsys.readouterr() == previous_output
    assert csv_reads == {name: 1 for name in previous_counts}
    assert len(runtime_probes) == (2 if hparam else 0)
    assert "Doctor phase: consultation" in previous_output.err
    assert "Doctor phase: publish outputs" in previous_output.err
    assert "doctor-private-" not in previous_output.out + previous_output.err
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("kind", ["survival", "multilabel"])
def test_evaluate_recipe_retains_one_validation_view_and_refreshes_next_call(tmp_path, monkeypatch, csv_reads, kind):
    recipe_path = _sidecar_recipe(tmp_path, kind)
    if kind == "survival":
        from data import survival as loader

        name = "load_survival_label_table"
    else:
        from data import multilabel as loader

        name = "load_multilabel_label_table"
    load = getattr(loader, name)
    mutations = []

    def mutate_after_validation(*args, **kwargs):
        labels = load(*args, **kwargs)
        if not mutations:
            mutations.append(True)
            for filename in ("event_time.csv", "is_event.csv", "has_label.csv"):
                path = tmp_path / filename
                path.write_text(path.read_text().replace("doctor-private-001", "doctor-private-003"))
        return labels

    monkeypatch.setattr(loader, name, mutate_after_validation)
    recipe, cfg, report = plans.evaluate_recipe(recipe_path)
    assert report.exit_code == 0
    assert csv_reads == (
        {"event_time.csv": 1, "is_event.csv": 1, "has_label.csv": 1, "index.csv": 1}
        if kind == "survival"
        else {"is_event.csv": 1, "has_label.csv": 1, "index.csv": 1}
    )
    public_cfg = {key: value for key, value in cfg.items() if key != "_source_config_bytes"}
    public_payload = json.dumps([recipe, public_cfg, asdict(report)])
    assert "validated_sidecar_keys" not in public_payload
    assert "doctor-private-" not in public_payload

    csv_reads.clear()
    _recipe, _cfg, report = plans.evaluate_recipe(recipe_path)
    assert report.exit_code == 1
    assert any(
        "missing from sidecars" in issue.message and "doctor-private-001" in issue.message for issue in report.issues
    )
    assert all(count == 1 for count in csv_reads.values())


@pytest.mark.parametrize("kind", ["survival", "multilabel"])
def test_valid_empty_sidecar_keys_are_not_treated_as_missing_evidence(tmp_path, csv_reads, kind):
    recipe_path = _sidecar_recipe(tmp_path, kind)
    for filename in ("event_time.csv", "is_event.csv", "has_label.csv"):
        (tmp_path / filename).write_text("eid,d1,d2\n")

    _recipe, cfg, report = plans.evaluate_recipe(recipe_path)

    assert cfg["finetune"][kind]["valid"] is True
    assert cfg["finetune"][kind]["sidecar_key_count"] == 0
    assert report.exit_code == 1
    assert any("2 missing" in issue.message for issue in report.issues)
    assert all(count == 1 for count in csv_reads.values())


@pytest.mark.parametrize(
    "state,expected_status,expected_exit",
    [("pass", "PASS", 0), ("warn", "WARN", 0), ("needs", "NEEDS_USER_INPUT", 2), ("fail", "FAIL", 1)],
)
def test_doctor_preserves_decision_output_and_blocked_probe_boundary(
    tmp_path, monkeypatch, capsys, runtime_probes, state, expected_status, expected_exit
):
    recipe_path = _sidecar_recipe(tmp_path, "survival", hparam=True)
    base_path = tmp_path / "recipe.yaml"
    base = yaml.safe_load(base_path.read_text())
    if state == "warn":
        payload = yaml.safe_load(recipe_path.read_text())
        payload["execution"] = {
            "target": "ssh",
            "host": "test-runtime",
            "path_validation": "defer",
            "python": "/test/python",
            "runtime_commit": "a" * 40,
            "workdir": "/test/runtime",
        }
        write_yaml(recipe_path, payload)
    elif state == "needs":
        base["inputs"].pop("label_name")
    elif state == "fail":
        index = tmp_path / "index.csv"
        index.write_text(index.read_text().replace("doctor-private-001", "doctor-private-003"))
    write_yaml(base_path, base)

    with monkeypatch.context() as uncached:
        uncached.setattr(plan_context, "index_summary", _without_reuse)
        _recipe, _cfg, previous_report = plans.evaluate_recipe(recipe_path)
        previous_exit = cli.main(["doctor", "--recipe", str(recipe_path)])
    previous_output = capsys.readouterr()
    _recipe, _cfg, report = plans.evaluate_recipe(recipe_path)
    assert asdict(report) == asdict(previous_report)
    assert cli.main(["doctor", "--recipe", str(recipe_path)]) == previous_exit == expected_exit
    output = capsys.readouterr()
    assert output == previous_output
    assert f"Status: {expected_status}" in output.out
    assert f"Doctor finished: exit_code={expected_exit}" in output.err
    assert len(runtime_probes) == (2 if expected_exit == 0 else 0)
    assert ("Doctor phase: runtime diagnostics" in output.err) is (expected_exit == 0)


@pytest.mark.parametrize("ineligible", ["missing_snapshot", "blocking_config", "invalid_sidecars", "missing_collector"])
def test_index_summary_reuse_requires_complete_local_evidence(tmp_path, monkeypatch, csv_reads, ineligible):
    recipe_path = _sidecar_recipe(tmp_path, "survival")
    recipe, cfg, _report = plans.evaluate_recipe(recipe_path)
    collector = {"survival": {"doctor-private-001", "doctor-private-002"}}
    if ineligible == "missing_snapshot":
        cfg.pop("_source_config_bytes")
    elif ineligible == "blocking_config":
        cfg["blocking_issues"] = ["structural config failure"]
    elif ineligible == "invalid_sidecars":
        cfg["finetune"]["survival"]["valid"] = False
    else:
        collector = None
    csv_reads.clear()
    calls = []

    def summarize(*args, **kwargs):
        calls.append(kwargs.get("validated_summary"))
        return index_csv.index_summary(*args, **kwargs)

    monkeypatch.setattr(plan_context, "index_summary", summarize)
    payload = plan_context.context_index_summary(recipe, cfg, validated_sidecar_keys=collector)
    assert calls == [None]
    assert payload["blocking_issues"] == []
    assert csv_reads == {"event_time.csv": 2, "is_event.csv": 1, "has_label.csv": 1, "index.csv": 1}


def test_sleep2stat_does_not_pass_its_config_to_index_summary(tmp_path, monkeypatch):
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration\nx.npz,train,60\n")
    cfg = {"is_sleep2stat": True, "sleep2stat": {"data": {"index": [str(index)]}}, "_source_config_bytes": b""}
    calls = []

    def summarize(*args, **kwargs):
        calls.append(kwargs)
        return index_csv.index_summary(*args, **kwargs)

    monkeypatch.setattr(plan_context, "index_summary", summarize)
    payload = plan_context.context_index_summary(
        {"task": "sleep2stat", "inputs": {"config": "unused.yaml"}}, cfg, validated_sidecar_keys={}
    )
    assert payload["blocking_issues"] == []
    assert calls[0]["config"] is None
    assert calls[0].get("validated_summary") is None


def test_forced_provider_failure_keeps_unbound_index_summary_behavior(tmp_path, monkeypatch):
    recipe_path = _sidecar_recipe(tmp_path, "survival")
    payload = yaml.safe_load(recipe_path.read_text())
    payload["variant"] = "sex_age_baseline"
    write_yaml(recipe_path, payload)
    calls = []

    def summarize(*args, **kwargs):
        calls.append(kwargs.get("validated_summary"))
        return index_csv.index_summary(*args, **kwargs)

    monkeypatch.setattr(plan_context, "index_summary", summarize)
    _recipe, cfg, report = plans.evaluate_recipe(recipe_path)
    assert report.exit_code != 0
    assert cfg["blocking_issues"]
    assert calls == []  # The rejected provider has no index path; do not recover one from another provider.


def test_standalone_index_summary_still_performs_its_own_validation(tmp_path, csv_reads):
    _sidecar_recipe(tmp_path, "survival")

    payload = index_csv.index_summary([tmp_path / "index.csv"], config=tmp_path / "config.yaml")

    assert payload["blocking_issues"] == []
    assert csv_reads == {"event_time.csv": 2, "is_event.csv": 1, "has_label.csv": 1, "index.csv": 1}


def test_summary_collects_keys_only_after_complete_loader_success(tmp_path):
    _sidecar_recipe(tmp_path, "survival")
    (tmp_path / "has_label.csv").write_text("eid,d1,d2\ndoctor-private-001,1,1\n")
    collector = {}

    cfg = config_summary(tmp_path / "config.yaml", validated_sidecar_keys=collector)

    assert cfg["finetune"]["survival"]["valid"] is False
    assert collector == {}


@pytest.mark.parametrize("kind", ["survival", "multilabel"])
def test_failed_full_sidecar_validation_is_not_reused(tmp_path, monkeypatch, csv_reads, kind):
    recipe_path = _sidecar_recipe(tmp_path, kind)
    (tmp_path / "has_label.csv").write_text("eid,d1,d2\ndoctor-private-001,1,1\n")
    calls = []

    def summarize(*args, **kwargs):
        calls.append(kwargs.get("validated_summary"))
        return index_csv.index_summary(*args, **kwargs)

    monkeypatch.setattr(plan_context, "index_summary", summarize)
    _recipe, cfg, report = plans.evaluate_recipe(recipe_path)

    assert report.exit_code == 2
    assert cfg["finetune"][kind]["valid"] is False
    assert calls == [None]
    assert csv_reads == (
        {"event_time.csv": 2, "is_event.csv": 2, "has_label.csv": 2, "index.csv": 1}
        if kind == "survival"
        else {"is_event.csv": 2, "has_label.csv": 2, "index.csv": 1}
    )


def test_doctor_keeps_blocked_output_templates_unchanged(tmp_path, monkeypatch, capsys, runtime_probes):
    recipe_path = _sidecar_recipe(tmp_path, "survival", hparam=True)
    base_path = tmp_path / "recipe.yaml"
    base = yaml.safe_load(base_path.read_text())
    base["inputs"].pop("label_name")
    write_yaml(base_path, base)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    previous_dir = tmp_path / "previous"
    current_dir = tmp_path / "current"

    with monkeypatch.context() as uncached:
        uncached.setattr(plan_context, "index_summary", _without_reuse)
        assert cli.main(["doctor", "--recipe", str(recipe_path), "--output-dir", str(previous_dir)]) == 2
    previous_output = capsys.readouterr()
    assert cli.main(["doctor", "--recipe", str(recipe_path), "--output-dir", str(current_dir)]) == 2
    current_output = capsys.readouterr()

    assert current_output.out.replace(str(current_dir), str(previous_dir)) == previous_output.out
    assert current_output.err.replace(str(current_dir), str(previous_dir)) == previous_output.err
    previous_files = {path.name: path.read_bytes() for path in previous_dir.iterdir()}
    current_files = {path.name: path.read_bytes() for path in current_dir.iterdir()}
    assert current_files == previous_files
    assert set(current_files) == {"questions.json", "questions.md", "decisions.yaml"}
    assert "doctor-private-" not in repr(current_files)
    assert {path: path.read_bytes() for path in before} == before
    assert runtime_probes == []


@pytest.mark.parametrize("kind", ["survival", "multilabel"])
def test_reuse_keeps_runtime_relative_paths_and_requested_splits(tmp_path, csv_reads, kind):
    recipe_path = _sidecar_recipe(tmp_path, kind)
    config_path = tmp_path / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["data"]["finetune_data_index"] = "index.csv"
    for field, value in config["finetune"][kind].items():
        if field.endswith("_index"):
            config["finetune"][kind][field] = Path(value).name
    write_yaml(config_path, config)
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["execution"] = {"target": "local", "workdir": str(tmp_path)}
    write_yaml(recipe_path, recipe)
    index = tmp_path / "index.csv"
    index.write_text(index.read_text() + "z.npz,test,60,unloaded-test-key,1,55,0\n")

    _recipe, _cfg, report = plans.evaluate_recipe(recipe_path)

    assert report.exit_code == 0
    assert csv_reads
    assert all(count == 1 for count in csv_reads.values())


@pytest.mark.parametrize("kind", ["survival", "multilabel"])
def test_doctor_still_validates_sidecar_rows_absent_from_main_index(tmp_path, kind):
    recipe_path = _sidecar_recipe(tmp_path, kind)
    for filename in ("event_time.csv", "is_event.csv", "has_label.csv"):
        path = tmp_path / filename
        value = "bad" if filename == ("event_time.csv" if kind == "survival" else "is_event.csv") else "1"
        path.write_text(path.read_text() + f"not-in-main-index,{value},1\n")

    _recipe, cfg, report = plans.evaluate_recipe(recipe_path)

    assert report.exit_code == 2
    assert cfg["finetune"][kind]["valid"] is False
    assert cfg["finetune"][kind]["issues"]


def test_build_context_keeps_its_independent_validation(tmp_path, csv_reads):
    _sidecar_recipe(tmp_path, "survival")

    plans.build_context(
        task="finetune",
        config=tmp_path / "config.yaml",
        output_dir=tmp_path / "context",
        label_name="incident_cox",
        variant="sleep2vec",
    )

    assert csv_reads == {"event_time.csv": 3, "is_event.csv": 2, "has_label.csv": 2, "index.csv": 1}
